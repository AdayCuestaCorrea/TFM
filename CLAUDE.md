# CLAUDE.md — TFM: Madrid public transport demand forecasting

Technical context for Claude Code. Not an end-user README.

Project root: `C:\MisCosas\Universidad\TFM`. Python 3.13.5 via `.\venv\Scripts\python.exe`
(or `.\venv\Scripts\Activate.ps1`). Pins in `requirements.txt`. Tests: `python -m pytest`.
LaTeX: `latexmk -pdf -interaction=nonstopmode -halt-on-error TFT.tex` from `docs/LaTeX/`.

**Current-phase spec is the plan the user attached.** Do not copy phase diaries into this
file. Forensic history of completed phases lives in `docs/CLAUDE_PHASE_LOG.md` — read that
file only if the task needs it. Do not `@`-import it (Claude Code would load it at startup).

New phase = new Claude Code session. Execute the plan in **this** session (parent), not a
coding subagent. Use `explore` only to search.

---

## Goal

Predict daily total public transport demand in Madrid (`total`) with a **two-stage residual
hybrid**: Stage 1 LSTM on the series; Stage 2 XGBoost on `e = y - ŷ_LSTM` with exogenous
features; `ŷ_final = ŷ_LSTM + ê_XGBoost`.

The Surribas-Sayago paper is **feature-engineering inspiration only** (lags, Fourier,
exogenous branch). It is **not** the architecture to replicate. Sequential residual
correction ≠ parallel CNN+LSTM+MLP fusion.

---

## Standing empirical result

The hybrid **does not beat** SARIMAX or `xgboost_alone` on val or test. Diagnostic phases
(7, 8, …) train **controls, not candidates**: nothing enters chapter 5's master comparison,
`dashboard_data.MODEL_NAMES`, or `models/`. A strong diagnostic score is a finding to
discuss, never a new headline model.

---

## OOF stacking (hard)

LSTM predictions used to train XGBoost on residuals **must be out-of-fold** (walk-forward
or time-series K-fold), **never in-sample**. In-sample residuals are optimistically small
and unlike inference residuals. Reject any Stage-2 fit on a single in-sample LSTM.

Seed: `GLOBAL_SEED = 42` in `src/utils/seed.py` — `random`, `numpy`, TensorFlow. Every LSTM
run logs fold id, date ranges, loss curve, MAE/RMSE on the held-out fold.

---

## Leakage (hard)

1. Lags/rolling on `total` or operators: `shift(1)` **before** rolling. Never unshifted
   `rolling`.
2. Splits are chronological only. No shuffled `train_test_split`, no plain `KFold`.
3. Scalers fit **only on train**; `transform` val/test. Never `fit_transform` on all rows.
4. No global rolling / Fourier-fitting / scaling on the full series before the split
   (deterministic Fourier basis is the sanctioned exception; see tests).
5. LSTM → XGBoost residuals: OOF only.

---

## Data (paths and traps)

Daily, 2023-01-01 → 2026-08-02, 1310 rows, no gaps/nulls at unification (re-verify on
ingestion). Raw files stay unmodified in `data/raw/`:

- `CRTM_Evolucion_demanda_diaria.xlsx` — sheet `diaria`, **`header=1`** (Excel row 2).
  `header=2` eats 2023-01-01. Keep `metro`, `emt`, `carretera`, `cercanias` even though
  `total` is the only target.
- `open-meteo-40.39N3.68W666m.csv` — `read_csv(..., header=2)` (or skiprows=3). Timezone
  metadata says `Europe/Berlin`.
- `300082-1-calendario_laboral-csv.csv` — `sep=";"`, `encoding="utf-8-sig"`, dates
  `format="%d/%m/%Y"`. `Dia_semana` is unaccented. Carry `domingo festivo` in encoding;
  **fail loudly** on unexpected `day_type` values (observed: laborable, sabado, domingo,
  festivo; domingo festivo count 0).

Column maps live in `src/ingestion/`. Do not invent a second mapping here.

---

## Conventions

- `src/`: pure functions, type hints, no import-time I/O, no absolute paths.
- `notebooks/`: EDA only; reusable logic goes to `src/`.
- Comments in English, rationale (why), not narration (what).
- `tests/`: pytest; leakage rules are the priority.

```
data/raw/       immutable
data/interim/   unification
data/processed/ modeling tables
src/ingestion/  loaders + unify
src/features/   lags, rolling, Fourier, calendar, weather
src/models/     baselines, lstm, hybrid_residual
src/evaluation/ metrics, figures, tables
src/utils/      seed, paths
reports/figures/
docs/LaTeX/
tests/
```

Acceptance (pipeline, not a phase diary): EDA + no missing dates; NaN report for
lags/rolling; baselines and LSTM and hybrid on the same chronological splits; hybrid
compared to LSTM-alone **and** XGBoost-alone on the same features.

**Splits** — import `src.utils.splits.chronological_split`; do not recompute locally.
Train 2023-01-01 → 2025-07-05 (917 days, 889 after 28-day warm-up, all inside train);
val 2025-07-06 → 2026-01-17 (196); test 2026-01-18 → 2026-08-02 (197). Same-day weather
never enters the model matrix (105 lagged/rolling weather features only). Same-day
operator columns are the target arithmetically — lagged operators only. Residual Stage 2
trains on OOF folds 2–5 (552 rows); fold 1 was degenerate and excluded after a coverage
check. XGBoost-alone trains on the full 889-row split.

---

## Tooling landmines (still true)

- kaleido **1.x** (`kaleido==1.3.0`). 0.2.1 hangs with plotly 6.x.
- Do **not** export PNGs from a Jupyter kernel (kaleido deadlock). Use
  `python -m src.evaluation.export_figures`.
- Do not set `pio.renderers.default = "notebook_connected"`.
