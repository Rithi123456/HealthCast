# HealthCast — Reliability-Gated COVID-19 Forecasting & Healthcare Stress Dashboard

HealthCast is a state-level COVID-19 forecasting project that explores whether machine-learning models can produce useful 7-day forecasts of daily confirmed cases for **Tamil Nadu (TN), Karnataka (KA), Maharashtra (MH), Delhi (DL), and Kerala (KL)**.

The project goes beyond simply training a model and displaying its prediction. It uses **chronological backtesting, rolling-origin validation, simple forecasting baselines, and a production reliability gate** so that an unsupported model prediction is not presented to the user as a trustworthy forecast.

> **Important result:** the current experiments show that the ML models do **not** outperform the strongest simple baseline on the untouched final test period. HealthCast therefore intentionally withholds ML forecasts rather than displaying misleading numbers.

---

## Why this project exists

COVID-19 case data is a time series, so a model can appear to perform well on historical validation windows while failing when the underlying data-generating regime changes.

HealthCast was designed to answer a more useful question than:

> "Can I get a good validation score?"

The project asks:

> **"Does the ML model outperform a simple baseline on genuinely unseen future data, consistently enough to justify showing its forecast in a production dashboard?"**

That distinction is the central idea of the project.

---

# Project Architecture

```text
Raw State-Level COVID Data
            │
            ▼
     Data Preparation
            │
            ├── Daily confirmed cases
            ├── Missing-value handling
            └── Chronological ordering
            │
            ▼
   Time-Series Feature Engineering
            │
            ├── Lag 1 / 2 / 3 / 7 / 14
            ├── Rolling mean 7 / 14
            ├── Rolling standard deviation
            ├── Growth rate
            ├── Level ratios
            └── Recent trend slopes
            │
            ▼
       Model Experiments
            │
      ┌─────┴─────┐
      │           │
     V1          V2
      │           │
      ▼           ▼
 Random Forest   Multiple ML models
                  + robust validation
            │
            ▼
            V3
   Regime-Aware Forecasting
            │
            ├── Recent training windows
            │   60 / 90 / 120 / 180 days
            │
            ├── Absolute target
            │
            └── Relative/log-change target
            │
            ▼
   Rolling-Origin Validation
            │
            ▼
   Model Selection Per State
            │
            ▼
    Untouched Final Test
            │
            ▼
      Reliability Gate
            │
      ┌─────┴─────┐
      │           │
     PASS        FAIL
      │           │
 Forecast      Forecast
 shown         withheld
```

---

# Forecasting Methodology

## 1. Daily time-series construction

The source data contains state-wise COVID records. The confirmed-case records are converted into daily state-level series.

The five states currently evaluated are:

- Tamil Nadu (TN)
- Karnataka (KA)
- Maharashtra (MH)
- Delhi (DL)
- Kerala (KL)

---

## 2. Time-series features

HealthCast uses information available at the forecast origin to construct features such as:

- Lag 1 day
- Lag 2 days
- Lag 3 days
- Lag 7 days
- Lag 14 days
- 7-day rolling average
- 14-day rolling average
- 7-day rolling standard deviation
- Recent growth rate
- Current level / 7-day average
- Current level / 14-day average
- 7-day trend slope
- 14-day trend slope

No future observations are used to construct forecasting features.

---

# Model Experiments

## V1 — Baseline ML Forecasting

The first version used state-specific Random Forest forecasting with chronological evaluation.

The final test showed that the ML forecasts were substantially worse than a simple persistence baseline.

This established the first important problem:

> A model can produce predictions without producing useful predictions.

---

## V2 — Robust Forecasting

V2 expanded the experiment to multiple model families and added rolling validation.

The experiment included:

- Random Forest
- HistGradientBoosting
- Ridge
- Rolling-origin validation
- Persistence baseline
- Seasonal-naive baseline
- Final untouched chronological test

V2 demonstrated an important failure pattern:

> Some configurations performed well during rolling validation but failed on the final unseen regime.

This indicated that validation performance alone was insufficient.

---

# V3 — Regime-Aware Forecasting

V3 was created specifically to investigate the regime-shift problem.

Instead of treating the entire historical series as equally relevant, V3 tested recent-history training windows:

```text
60 days
90 days
120 days
180 days
```

V3 also compared two target formulations:

### Absolute target

Predict future daily cases directly.

### Relative target

Predict the change relative to the current level using:

```text
log(1 + future_cases) - log(1 + current_cases)
```

The predicted relative change is then transformed back into predicted case counts.

The V3 experiment tested:

- Random Forest
- HistGradientBoosting
- Absolute target
- Relative target
- 60 / 90 / 120 / 180-day recent windows

---

# Validation Strategy

HealthCast uses **chronological evaluation** rather than random train/test splitting.

## Rolling-origin validation

Four historical validation windows are evaluated.

A candidate configuration must win against the strongest simple baseline in at least:

```text
3 out of 4 windows
```

to pass rolling validation.

This prevents a model from being selected because of one unusually favorable validation period.

---

# Baselines

Machine-learning forecasts are compared against simple forecasting methods.

## Persistence baseline

The next seven days are predicted as the latest known value.

```text
Forecast(t+1 ... t+7) = Last Known Value
```

## Seasonal-naive baseline

The forecast uses the corresponding previous weekly pattern.

The strongest of these baselines is used as the comparison point for the production gate.

This is important because a complicated ML model should not be considered useful if a trivial forecasting strategy performs better.

---

# Production Reliability Gate

HealthCast does **not** automatically display the model prediction.

A state can enter production only when BOTH conditions are satisfied:

```text
Rolling validation
       │
       ├── At least 3/4 windows won
       │
       ▼
Untouched final test
       │
       ├── ML model beats strongest baseline
       │
       ▼
Production Gate = PASS
       │
       ▼
Forecast may be displayed
```

If either condition fails:

```text
Production Gate = FAIL
       │
       ▼
Forecast withheld
```

This prevents the dashboard from presenting an unreliable model-generated number as if it were a trustworthy forecast.

---

# Current V3 Results

The current V3 experiment produced the following final-test results:

| State | Selected V3 configuration | V3 MAE | Strongest Baseline MAE | Final Test | Production Gate |
|---|---|---:|---:|---|---|
| TN | 180-day + Random Forest + relative target | 6,018.5 | 2,289.4 | ❌ Fail | ❌ Fail |
| KA | 180-day + Random Forest + relative target | 7,177.8 | 3,224.9 | ❌ Fail | ❌ Fail |
| MH | No production candidate | — | 3,513.0 | ❌ Fail | ❌ Fail |
| DL | No production candidate | — | 1,225.7 | ❌ Fail | ❌ Fail |
| KL | No production candidate | — | 3,514.7 | ❌ Fail | ❌ Fail |

TN and KA passed the rolling validation stage:

```text
TN → 3/4 windows
KA → 3/4 windows
```

However, both failed the untouched final test.

This is the most important result of the current project.

---

# What the Results Mean

The experiments revealed a consistent pattern:

```text
Historical rolling validation
        ↓
Some ML configurations look strong
        ↓
Final unseen COVID regime
        ↓
ML performance deteriorates
        ↓
Simple baseline wins
```

This strongly suggests that **temporal regime shift** is a major limitation for the current forecasting setup.

The project therefore does not claim that the ML model is production-accurate.

Instead, it demonstrates a complete model-evaluation workflow that can identify when a model should **not** be trusted.

---

# Streamlit Dashboard

The Streamlit application is reliability-gated.

The dashboard displays:

- Selected state
- Model MAE
- Strongest baseline MAE
- Relative performance against the baseline
- Rolling-validation result
- Final-test result
- Production-gate status
- Recent observed case trend

When a state fails the production gate, the app explicitly displays:

```text
Forecast withheld.
```

The forecast is not generated for presentation merely because a model file exists.

Similarly, the state risk ranking is shown only for production-approved models.

With the current V3 results, no state passes the production gate, so no ML forecast-based risk ranking is displayed.

---

# Why Withholding the Forecast Is a Feature

A common mistake in ML projects is:

```text
Train model
   ↓
Get prediction
   ↓
Put prediction in dashboard
   ↓
Call it forecasting
```

HealthCast deliberately avoids this.

Instead:

```text
Train
  ↓
Validate
  ↓
Compare against baselines
  ↓
Test on untouched future data
  ↓
Evaluate reliability
  ↓
Only then display forecast
```

If the model fails, the dashboard says so.

This makes the application more honest and operationally safer than displaying a prediction simply because the model produced one.

---

# Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib
- Streamlit
- Jupyter Notebook
- Git / GitHub

---

# Project Structure

```text
HealthCast/
│
├── app.py
│
├── data/
│   └── state_data.csv
│
├── notebooks/
│   ├── HealthCast_V1_Baseline.ipynb
│   ├── HealthCast_V2_Robust_Forecasting_FINAL.ipynb
│   └── HealthCast_V3_Regime_Aware_Forecasting.ipynb
│
├── artifacts/
│   └── V1 artifacts
│
├── artifacts_v2/
│   └── V2 artifacts
│
├── artifacts_v3/
│   ├── model_metadata.json
│   └── state_data.csv
│
├── requirements.txt
│
└── README.md
```

---

# Running the Application

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Streamlit:

```bash
streamlit run app.py
```

The application reads the current V3 production metadata from:

```text
artifacts_v3/model_metadata.json
```

and uses the reliability gate stored in that metadata.

---

# Reproducing V3

Open:

```text
notebooks/HealthCast_V3_Regime_Aware_Forecasting.ipynb
```

Run the notebook chronologically.

The notebook produces:

```text
artifacts_v3/
├── model_metadata.json
└── state_data.csv
```

Production model files are created only for states that pass the production gate.

---

# Key Learning Outcomes

This project demonstrates practical understanding of:

- Time-series forecasting
- Chronological train/test splitting
- Rolling-origin validation
- Feature engineering for time series
- Baseline forecasting
- Model comparison
- Distribution/regime shift
- Model reliability evaluation
- Production gating
- Preventing misleading model outputs
- Streamlit deployment
- Model artifact management

The most important lesson is:

> **A model with a good validation score is not automatically a good forecasting model.**

A model should be evaluated against simple baselines and tested on genuinely unseen future data before its predictions are presented to users.

---

# Project Status

**Status: Completed experimental forecasting system**

The current HealthCast version is intentionally conservative.

No state currently passes the complete V3 production gate, so the dashboard withholds ML forecasts.

This is an evaluation result, not an error condition.

Future work would require a justified change in forecasting formulation, data sources, or problem definition rather than simply trying additional models until a favorable score appears.
