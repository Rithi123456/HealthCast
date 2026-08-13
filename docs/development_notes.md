# Notes — HealthCast Notebook Fix & Upgrade

## Bugs found in the original notebook (verified by reading the actual file, not assumed)

1. **Leakage in outlier capping (Cell 3, original).**
   `df_cases["daily_cases"].quantile(0.99)` was computed on the full dataset
   before the train/test split. Test-period values influenced a threshold
   applied to training data.

2. **National model reused for state forecasts (Cells 6–10, original).**
   One RandomForestRegressor trained on national aggregate case counts was
   applied to forecast individual states (TN, KA, MH, DL, KL) pulled from a
   different source at a different scale. No statistical basis for that
   transfer.

3. **Train/inference feature mismatch (Cells 4 vs 9, original).**
   `growth_rate` was unclipped during training but clipped to ±20% during
   recursive forecasting — the model was fed inference-time inputs from a
   distribution it never trained on.

## What was fixed

| Bug | Fix applied |
|---|---|
| #1 | Cap now fit on train slice only (`time_split_and_cap()`), applied to full series after. |
| #2 | Separate RandomForest trained per state, on that state's own history, same feature pipeline as national model. |
| #3 | Single `GROWTH_CLIP` constant (0.30) applied identically at training and inference. |

## What was added (upgrade, not just bugfix)

- Naive-baseline MAE/RMSE reported next to every model metric.
- Backtesting per state: forecast vs. real held-out last 7 days, `beats_naive` flag reported explicitly.
- Missing state-day values forward/back-filled instead of `fillna(0)`.
- API fetches wrapped in try/except with local CSV caching fallback under `data_cache/`.
- Model metadata (feature list, clip value, outlier caps) saved as JSON alongside pickled models under `artifacts/`.
- Output sanity clamp (`SANITY_BOUND_PCT`) kept for the operational forecast only, explicitly excluded from backtest evaluation so accuracy numbers stay honest.

## What was NOT fixed (documented, not solved)

- **Bed-capacity figures (Section 7)** are the same undated placeholder numbers from the original notebook. No real source was substituted — flagged with a `TODO` in-code and called out in both README and notebook Section 10. Do not use for real capacity planning.
- No external features added (vaccination, policy, mobility, variants) — model is still case-count-history-only.
- Per-state sample sizes are still small; some states do not beat the naive baseline in backtesting on real data (unknown until you run it — synthetic test data showed mixed results, which is expected/correct behavior, not a bug).

## Verification performed before delivery

- `nbformat.validate()` passed — structurally valid `.ipynb`.
- Every code cell parsed with `ast.parse()` — zero syntax errors.
- Full pipeline logic (all cells except the two live API-fetch cells) executed end-to-end against synthetic data — zero runtime errors.
- Could not execute against real data: sandbox has no network access to `disease.sh` or `covid19india.org`. Run locally to get real numbers before trusting any output.
