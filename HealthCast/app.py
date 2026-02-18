import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.title("HealthCast — Real-Time Healthcare Stress Forecast")

# --------------------------
# Load Model
# --------------------------
model = joblib.load("covid_forecast_model.pkl")

# --------------------------
# Load State Data
# --------------------------
df = pd.read_csv("state_data.csv")
df["Date"] = pd.to_datetime(df["Date"])

states = ["TN", "KA", "MH", "DL", "KL"]

state_beds = {
    "TN": 120000,
    "KA": 110000,
    "MH": 250000,
    "DL": 75000,
    "KL": 90000,
}

# --------------------------
# Feature Builder
# --------------------------
def build_features(series):
    df_feat = pd.DataFrame({"cases": series})

    df_feat["lag_1"] = df_feat["cases"].shift(1)
    df_feat["lag_2"] = df_feat["cases"].shift(2)
    df_feat["lag_3"] = df_feat["cases"].shift(3)
    df_feat["lag_7"] = df_feat["cases"].shift(7)
    df_feat["lag_14"] = df_feat["cases"].shift(14)

    df_feat["rolling_avg_7"] = df_feat["cases"].rolling(7).mean()
    df_feat["rolling_avg_14"] = df_feat["cases"].rolling(14).mean()
    df_feat["rolling_std_7"] = df_feat["cases"].rolling(7).std()

    df_feat["growth_rate"] = df_feat["cases"].pct_change()

    df_feat = df_feat.fillna(method="bfill").fillna(0)
    return df_feat

# --------------------------
# Recursive Forecast
# --------------------------
def forecast_state(series):
    feat_df = build_features(series)
    last = feat_df.iloc[-1]

    features = last[
        ["lag_1","lag_2","lag_3","lag_7","lag_14",
         "rolling_avg_7","rolling_avg_14","rolling_std_7","growth_rate"]
    ].values.reshape(1,-1)

    preds = []

    for _ in range(7):
        next_pred = model.predict(features)[0]
        preds.append(float(next_pred))

        lag_1, lag_2, lag_3, lag_7, lag_14, r7, r14, rs7, gr = features[0]

        new_features = [
            next_pred, lag_1, lag_2, lag_3, lag_7,
            r7, r14, rs7, gr
        ]

        features = np.array(new_features).reshape(1,-1)

    return preds

# --------------------------
# UI
# --------------------------
state = st.selectbox("Select State", states)

series = df[state].values
forecast = forecast_state(series)

# --------------------------
# Plot Forecast
# --------------------------
st.subheader("7-Day Forecast")

fig, ax = plt.subplots()
ax.plot(range(1,8), forecast, marker="o")
ax.set_xlabel("Future Day")
ax.set_ylabel("Predicted Cases")
ax.set_title(f"Forecast for {state}")
st.pyplot(fig)

# --------------------------
# Stress
# --------------------------
beds = state_beds[state]
stress = (forecast[-1] * 0.05) / beds

def stress_level(s):
    if s < 0.001: return "Very Low"
    elif s < 0.005: return "Low"
    elif s < 0.02: return "Rising"
    elif s < 0.05: return "High"
    else: return "Critical"

st.subheader("Healthcare Stress (Day 7)")
st.write("Cases:", int(forecast[-1]))
st.write("Stress Index:", round(stress,6))
st.write("Risk Level:", stress_level(stress))


# --------------------------
# Risk Ranking (All States)
# --------------------------
st.subheader("State Risk Ranking (Day 7)")

ranking = []

for stt in states:
    series = df[stt].values
    preds = forecast_state(series)
    beds = state_beds[stt]
    stress_val = (preds[-1] * 0.05) / beds
    ranking.append((stt, preds[-1], stress_val))

# Sort by stress descending
ranking = sorted(ranking, key=lambda x: x[2], reverse=True)

for i, (stt, cases, stress_val) in enumerate(ranking, 1):
    st.write(
        f"{i}. {stt} | Cases={int(cases)} | Stress={round(stress_val,6)} | {stress_level(stress_val)}"
    )


# --------------------------
# Stress Trend (Last 3 Days Forecast)
# --------------------------
st.subheader("Stress Trend")

last3 = forecast[-3:]

if last3[2] > last3[1] > last3[0]:
    trend = "Increasing"
elif last3[2] < last3[1] < last3[0]:
    trend = "Decreasing"
else:
    trend = "Stable"

st.write("Trend:", trend)


