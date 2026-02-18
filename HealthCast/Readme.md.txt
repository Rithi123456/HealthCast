# HealthCast — Healthcare Stress Forecasting System

HealthCast is a machine learning–based time-series forecasting system that predicts short-term infection trends and converts them into healthcare capacity stress indicators. The system provides state-wise risk assessment and decision insights through an interactive Streamlit dashboard.

---

## Problem

Healthcare systems often fail to anticipate sudden surges in infections. Raw case counts alone do not reflect healthcare pressure. HealthCast bridges this gap by forecasting future infections and translating them into actionable healthcare stress signals.

---

## Solution

HealthCast analyzes historical infection data to:

- Forecast short-term case trends using Machine Learning
- Estimate healthcare demand (Beds, ICU, Ventilators)
- Compute Healthcare Stress Index
- Classify risk levels (Very Low → Critical)
- Rank states based on predicted healthcare pressure
- Provide visual insights through an interactive dashboard

---

## Key Features

- Time-series forecasting using Random Forest
- Automated feature engineering (lags, rolling trends, volatility)
- Recursive multi-step prediction
- Forecast stabilization to prevent unrealistic spikes
- Healthcare stress modeling from predicted cases
- Risk classification and ranking
- Interactive Streamlit dashboard
- Real-time feature extraction from latest data

---

## Machine Learning Approach

**Model Used:** RandomForestRegressor  

**Features Used:**
- Lag features (1,2,3,7,14 days)
- Rolling averages (7,14 days)
- Rolling standard deviation (volatility)
- Growth rate (spread acceleration)

**Why Recursive Forecasting?**  
To generate multi-day predictions using previous predicted values.

**Why Stabilization?**  
To prevent exploding forecasts and unrealistic jumps.

---

## Healthcare Stress Index

Stress is calculated as:




Risk Levels:

| Stress | Level |
|--------|-------|
| < 0.001 | Very Low |
| < 0.005 | Low |
| < 0.02 | Rising |
| < 0.05 | High |
| >= 0.05 | Critical |

---

## Dashboard

The Streamlit dashboard shows:

- State-wise forecast visualization
- Healthcare stress indicator
- Risk classification
- Risk ranking
- Trend detection

Run dashboard:

```bash
streamlit run app.py

