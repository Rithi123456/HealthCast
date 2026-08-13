import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# HealthCast — Production Reliability-Gated Dashboard
# ============================================================
#
# This app is intentionally driven by V3 artifacts.
#
# V3 production rule:
#   1. Rolling validation must pass (>= 3/4 windows)
#   2. Final untouched test must beat the strongest baseline
#
# If a state does not pass BOTH checks, its ML forecast is hidden.
# The app never falls back to displaying an unsupported prediction.
# ============================================================

st.set_page_config(
    page_title="HealthCast",
    page_icon="📈",
    layout="wide",
)

st.title("HealthCast — 7-Day COVID Forecast & Healthcare Stress")
st.caption(
    "Reliability-gated state-level forecasting. "
    "Forecasts are shown only for models that pass both rolling validation "
    "and the untouched chronological test."
)

BASE_DIR = Path(__file__).resolve().parent

# V3 is the current experiment/production candidate.
# V1/V2 are intentionally not used by this dashboard.
ARTIFACT_DIR = BASE_DIR / "artifacts_v3"

if not ARTIFACT_DIR.exists():
    st.error(
        f"V3 artifact directory not found: {ARTIFACT_DIR}\n\n"
        "Run the V3 notebook first."
    )
    st.stop()

METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"
DATA_PATH = ARTIFACT_DIR / "state_data.csv"

if not METADATA_PATH.exists():
    st.error(f"Missing V3 metadata: {METADATA_PATH}")
    st.stop()

if not DATA_PATH.exists():
    st.error(f"Missing V3 state data: {DATA_PATH}")
    st.stop()

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)


# ============================================================
# Metadata helpers
# ============================================================

def clean_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def get_state_meta(state_code):
    states_meta = metadata.get("states", {})
    value = states_meta.get(state_code, {})
    return value if isinstance(value, dict) else {}


FORECAST_DAYS = int(metadata.get("forecast_horizon", metadata.get("forecast_days", 7)))
FEATURES = metadata.get(
    "feature_columns",
    metadata.get("features", []),
)

GATE_RULE = metadata.get("gate_rule", {})
ROLLING_REQUIRED = int(GATE_RULE.get("rolling_windows_required", 3))
ROLLING_TESTED = int(GATE_RULE.get("rolling_windows_tested", 4))

if not FEATURES:
    st.error(
        "V3 metadata does not contain feature columns. "
        "The artifact set is incomplete."
    )
    st.stop()


# ============================================================
# Load data — supports V3 generated state_data.csv
# and the older raw Confirmed format.
# ============================================================

df_raw = pd.read_csv(DATA_PATH)

# V3 notebook saves state_daily with Date as the index.
if "Date" in df_raw.columns:
    df = df_raw.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

# Also support raw state_data.csv if someone points the artifacts
# directory at an older dataset.
elif "Date_YMD" in df_raw.columns:
    df = df_raw.copy()
    df["Date"] = pd.to_datetime(df["Date_YMD"], errors="coerce")
    if "Status" in df.columns:
        df = df[
            df["Status"].astype(str).str.strip().str.lower().eq("confirmed")
        ].copy()
    df = df.sort_values("Date").reset_index(drop=True)

elif "Status" in df_raw.columns and "Date" in df_raw.columns:
    df = df_raw.copy()
    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce",
        dayfirst=True,
    )
    df = df[
        df["Status"].astype(str).str.strip().str.lower().eq("confirmed")
    ].copy()
    df = df.sort_values("Date").reset_index(drop=True)

else:
    st.error(
        "Unsupported state_data.csv format. Expected either V3 daily data "
        "with a Date column or the older raw Status/Date format."
    )
    st.stop()


# Convert state columns to numeric.
candidate_states = [
    s for s in ["TN", "KA", "MH", "DL", "KL"]
    if s in df.columns
]

for state_code in candidate_states:
    df[state_code] = pd.to_numeric(
        df[state_code],
        errors="coerce",
    )


if not candidate_states:
    st.error("No supported state columns found in state_data.csv.")
    st.stop()


# ============================================================
# V3 feature reconstruction
# ============================================================

def make_v3_feature_row(history):
    """
    Recreate the V3 feature vector at the latest known date.

    V3 features:
      lag_1, lag_2, lag_3, lag_7, lag_14,
      rolling_avg_7, rolling_avg_14, rolling_std_7,
      growth_rate, level_ratio_7, level_ratio_14,
      trend_slope_7, trend_slope_14
    """
    s = (
        pd.Series(history, dtype=float)
        .ffill()
        .bfill()
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if len(s) < 14:
        raise ValueError(
            f"Need at least 14 days of history; got {len(s)}."
        )

    lag_1 = s.iloc[-1]
    lag_2 = s.iloc[-2]
    lag_3 = s.iloc[-3]
    lag_7 = s.iloc[-7]
    lag_14 = s.iloc[-14]

    rolling_avg_7 = s.iloc[-7:].mean()
    rolling_avg_14 = s.iloc[-14:].mean()
    rolling_std_7 = s.iloc[-7:].std()

    if lag_2 == 0:
        growth_rate = 0.0
    else:
        growth_rate = (lag_1 - lag_2) / lag_2

    growth_rate = float(np.clip(growth_rate, -5.0, 5.0))

    level_ratio_7 = (
        lag_1 / rolling_avg_7
        if rolling_avg_7 != 0
        else 0.0
    )

    level_ratio_14 = (
        lag_1 / rolling_avg_14
        if rolling_avg_14 != 0
        else 0.0
    )

    def calc_slope(values):
        values = np.asarray(values, dtype=float)
        x = np.arange(len(values), dtype=float)
        return float(np.polyfit(x, values, 1)[0])

    trend_slope_7 = calc_slope(s.iloc[-7:])
    trend_slope_14 = calc_slope(s.iloc[-14:])

    values = {
        "lag_1": lag_1,
        "lag_2": lag_2,
        "lag_3": lag_3,
        "lag_7": lag_7,
        "lag_14": lag_14,
        "rolling_avg_7": rolling_avg_7,
        "rolling_avg_14": rolling_avg_14,
        "rolling_std_7": rolling_std_7,
        "growth_rate": growth_rate,
        "level_ratio_7": level_ratio_7,
        "level_ratio_14": level_ratio_14,
        "trend_slope_7": trend_slope_7,
        "trend_slope_14": trend_slope_14,
    }

    missing = [c for c in FEATURES if c not in values]

    if missing:
        raise ValueError(
            "The V3 metadata requests unsupported feature(s): "
            + ", ".join(missing)
        )

    return pd.DataFrame(
        [[values[c] for c in FEATURES]],
        columns=FEATURES,
    )


def inverse_relative_prediction(relative_prediction, current_value):
    """
    V3 relative target:
        log1p(future) - log1p(current)

    Therefore:
        future = expm1(predicted_relative + log1p(current))
    """
    current_value = max(float(current_value), 0.0)

    pred = np.expm1(
        np.asarray(relative_prediction, dtype=float)
        + np.log1p(current_value)
    )

    return np.maximum(pred, 0.0)


def forecast_state(history, model, target_mode):
    """Run one direct 7-day prediction for a production-approved state."""
    feature_row = make_v3_feature_row(history)

    raw = np.asarray(
        model.predict(feature_row)[0],
        dtype=float,
    )

    if target_mode == "relative":
        return inverse_relative_prediction(
            raw,
            pd.Series(history).iloc[-1],
        ).tolist()

    # Absolute-target fallback for completeness.
    return np.maximum(raw, 0.0).tolist()


# ============================================================
# Reliability gate
# ============================================================

def evaluate_state_gate(state_code):
    meta = get_state_meta(state_code)

    model_mae = meta.get(
        "model_mae",
        meta.get("mae"),
    )

    persistence_mae = meta.get("persistence_mae")
    seasonal_mae = meta.get("seasonal_naive_mae")

    strongest_baseline = meta.get(
        "strongest_baseline_mae"
    )

    if strongest_baseline is None:
        values = [
            x for x in [persistence_mae, seasonal_mae]
            if x is not None
        ]
        strongest_baseline = min(values) if values else None

    beats = bool(
        meta.get(
            "beats_strongest_baseline",
            meta.get("production_gate", False),
        )
    )

    rolling_pass = bool(
        meta.get("rolling_pass", False)
    )

    rolling_wins = int(
        meta.get(
            "windows_won",
            meta.get("validation_windows_won", 0),
        )
    )

    rolling_tested = int(
        meta.get(
            "windows_tested",
            meta.get("validation_windows_tested", 0),
        )
    )

    production_gate = bool(
        meta.get(
            "production_gate",
            beats and rolling_pass,
        )
    )

    return {
        "model_mae": model_mae,
        "persistence_mae": persistence_mae,
        "seasonal_naive_mae": seasonal_mae,
        "strongest_baseline_mae": strongest_baseline,
        "beats_baseline": beats,
        "rolling_pass": rolling_pass,
        "rolling_wins": rolling_wins,
        "rolling_tested": rolling_tested,
        "production_gate": production_gate,
    }


# ============================================================
# Load only production-approved models.
#
# Missing model files are NORMAL when a state failed the gate.
# The app must not crash because of that.
# ============================================================

state_models = {}

for state_code in candidate_states:
    state_meta = get_state_meta(state_code)

    model_path = ARTIFACT_DIR / f"state_model_{state_code}.pkl"

    gate = evaluate_state_gate(state_code)

    if not gate["production_gate"]:
        continue

    if not model_path.exists():
        # Metadata says PASS but model is missing.
        # Treat this as a deployment integrity failure and do not forecast.
        continue

    try:
        state_models[state_code] = joblib.load(model_path)
    except Exception:
        # Never allow a corrupted/incompatible model to crash the dashboard.
        continue


# ============================================================
# Dashboard
# ============================================================

available_states = candidate_states

state = st.selectbox(
    "Select State",
    available_states,
)

gate = evaluate_state_gate(state)

st.subheader("Model Reliability")

c1, c2, c3 = st.columns(3)

with c1:
    value = gate["model_mae"]
    st.metric(
        "Model MAE",
        "N/A" if value is None else f"{float(value):,.1f}",
    )

with c2:
    value = gate["strongest_baseline_mae"]
    st.metric(
        "Strongest Baseline MAE",
        "N/A" if value is None else f"{float(value):,.1f}",
    )

with c3:
    model_mae = gate["model_mae"]
    baseline_mae = gate["strongest_baseline_mae"]

    if (
        model_mae is not None
        and baseline_mae is not None
        and baseline_mae != 0
    ):
        improvement = (
            1 - float(model_mae) / float(baseline_mae)
        ) * 100

        st.metric(
            "vs Strongest Baseline",
            f"{improvement:+.1f}%",
        )
    else:
        st.metric(
            "vs Strongest Baseline",
            "N/A",
        )

st.write(
    f"Rolling validation: **{gate['rolling_wins']}/"
    f"{gate['rolling_tested']}** windows won "
    f"(required: {ROLLING_REQUIRED}/{ROLLING_TESTED})."
)

g1, g2, g3 = st.columns(3)

with g1:
    if gate["beats_baseline"]:
        st.success("Final test: PASS")
    else:
        st.error("Final test: FAIL")

with g2:
    if gate["rolling_pass"]:
        st.success("Rolling validation: PASS")
    else:
        st.error("Rolling validation: FAIL")

with g3:
    if gate["production_gate"]:
        st.success("Production gate: PASS")
    else:
        st.error("Production gate: FAIL")


# ============================================================
# Forecast section
# ============================================================

forecast_allowed = (
    gate["production_gate"]
    and state in state_models
)

if not forecast_allowed:

    reasons = []

    if not gate["beats_baseline"]:
        reasons.append("untouched final test")

    if not gate["rolling_pass"]:
        reasons.append("rolling validation")

    if gate["production_gate"] and state not in state_models:
        reasons.append("missing/incompatible production model artifact")

    if not reasons:
        reasons.append("production reliability gate")

    st.error(
        "Forecast withheld."
    )

    st.warning(
        "The ML forecast is intentionally hidden because the model "
        "does not currently have sufficient evidence to be presented "
        "as a reliable production forecast."
    )

    st.caption(
        "Failed criterion: " + ", ".join(reasons) + "."
    )

else:

    history = (
        df[state]
        .ffill()
        .bfill()
        .astype(float)
        .values
    )

    state_meta = get_state_meta(state)

    target_mode = state_meta.get(
        "selected_target_mode",
        "absolute",
    )

    try:
        forecast = forecast_state(
            history,
            state_models[state],
            target_mode,
        )
    except Exception as exc:
        st.error(
            "The production artifact could not generate a forecast. "
            "The forecast has been withheld."
        )
        st.caption(f"Artifact compatibility error: {exc}")
        forecast = None

    if forecast is not None:

        st.subheader("7-Day Forecast")

        forecast_df = pd.DataFrame({
            "Future Day": [
                f"Day +{i}"
                for i in range(1, FORECAST_DAYS + 1)
            ],
            "Predicted Cases": [
                round(float(x), 1)
                for x in forecast[:FORECAST_DAYS]
            ],
        })

        st.dataframe(
            forecast_df,
            use_container_width=True,
            hide_index=True,
        )

        fig, ax = plt.subplots()

        ax.plot(
            range(1, len(forecast) + 1),
            forecast,
            marker="o",
        )

        ax.set_xlabel("Future Day")
        ax.set_ylabel("Predicted Daily Cases")
        ax.set_title(
            f"Direct 7-Day Forecast — {state}"
        )
        ax.grid(True)

        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Forecast Trend")

        if forecast[-1] > forecast[0]:
            trend = "Increasing"
        elif forecast[-1] < forecast[0]:
            trend = "Decreasing"
        else:
            trend = "Stable"

        st.write(
            f"Day +1 → Day +{FORECAST_DAYS}: **{trend}**"
        )


# ============================================================
# Current observed trend
# ============================================================

st.subheader("Recent Observed Cases")

recent = (
    df[["Date", state]]
    .dropna()
    .tail(30)
    .copy()
)

fig, ax = plt.subplots()

ax.plot(
    recent["Date"],
    recent[state],
    marker="o",
)

ax.set_xlabel("Date")
ax.set_ylabel("Daily Confirmed Cases")
ax.set_title(
    f"Recent Observed Daily Cases — {state}"
)

ax.tick_params(axis="x", rotation=45)
ax.grid(True)

st.pyplot(fig)
plt.close(fig)


# ============================================================
# State reliability ranking
#
# IMPORTANT:
# Only production-approved states can enter this table.
# If zero states pass, the table is intentionally empty.
# ============================================================

st.subheader("State Risk Ranking — Production-Passed Models Only")

ranking = []

for state_code in candidate_states:

    gate_info = evaluate_state_gate(state_code)

    if not gate_info["production_gate"]:
        continue

    if state_code not in state_models:
        continue

    try:
        history = (
            df[state_code]
            .ffill()
            .bfill()
            .astype(float)
            .values
        )

        state_meta = get_state_meta(state_code)

        target_mode = state_meta.get(
            "selected_target_mode",
            "absolute",
        )

        preds = forecast_state(
            history,
            state_models[state_code],
            target_mode,
        )

        ranking.append({
            "State": state_code,
            "Day +7 Cases": int(round(preds[-1])),
            "Model MAE": round(
                float(gate_info["model_mae"]),
                1,
            ),
        })

    except Exception:
        # A failed forecast never enters the ranking.
        continue


if ranking:

    ranking_df = pd.DataFrame(ranking)

    ranking_df = ranking_df.sort_values(
        "Day +7 Cases",
        ascending=False,
    ).reset_index(drop=True)

    ranking_df.insert(
        0,
        "Rank",
        np.arange(1, len(ranking_df) + 1),
    )

    st.dataframe(
        ranking_df,
        use_container_width=True,
        hide_index=True,
    )

else:

    st.info(
        "No state currently passes the V3 production reliability gate. "
        "Therefore no ML forecast-based ranking is shown."
    )


# ============================================================
# Footer
# ============================================================

st.divider()

st.caption(
    "HealthCast uses chronological evaluation and a production reliability "
    "gate. A model that fails the untouched test or rolling validation is "
    "not presented as a reliable forecast."
)
