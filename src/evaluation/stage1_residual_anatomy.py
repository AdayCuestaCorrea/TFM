"""Diagnostic anatomy of the Stage 1 LSTM residual — why the hybrid path loses.

DIAGNOSTIC CONTROL, NOT A CANDIDATE MODEL. Nothing here trains, refits or re-predicts
anything, and no artifact, variant or metric produced in this phase may enter the master
comparison table of chapter 5 (`tabla_comparacion_test` / `tabla_comparacion_val`) or be
presented as competing for best model. `dashboard_data.MODEL_NAMES`,
`full_comparison.parquet` and `sensitivity_comparison.parquet` are left untouched; a test
(`test_phase7_adds_no_model_to_the_master_comparison`) pins that mechanically. The only
purpose of this module is explanatory: chapter 5 §5.6.1 asserts that Stage 1 is the
bottleneck, and this module measures it.

This module is computation only, no plotting — mirroring the
`residual_diagnostics.py` / `dashboard_data.py` split. The Plotly builders that consume it
live in `memoria_figures.py`.

THREE ANALYSES:
  1. Residual structure of Stage 1 — ACF/PACF, Ljung-Box, weekday profile, periodogram,
     calendar-group breakdown, and residual-vs-lagged-weather correlation (raw and
     partialled on the annual Fourier terms). Acceptance criteria are fixed as module
     constants below, before any result is seen, so the verdict cannot drift.
  2. The error chain lstm_alone -> hybrid -> xgboost_alone, quantified: how much of the
     124k test-MAE gap between a univariate LSTM and a direct XGBoost does the residual
     stage actually close. This is NOT a decomposition (three separate models, not
     additive contributions); the table is primary and the figure is a stepped bar.
  3. Sample-size context, difficulty-controlled — per-fold OOF error expressed as a skill
     ratio against seasonal_naive scored on the identical fold block, so a harder
     evaluation period does not masquerade as a smaller-sample effect.

FRAMING (confirmatory, per the plan's objection O3). A univariate LSTM that never sees the
calendar will show period-7 residual structure; that is close to guaranteed. The
contribution is quantifying how much, not discovering whether. Ljung-Box and any
weekday-group significance test are anti-conservative here (no meaningful parameter count
for the LSTM; autocorrelated residuals violate independence), so effect size is the
primary criterion and p-values are secondary with the caveat stated inline.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal, stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import acf, pacf

from src.evaluation.dashboard_data import _require, load_all_predictions
from src.evaluation.day_type_breakdown import BRIDGE_COL, DAY_TYPE_COL, breakdown
from src.evaluation.metrics import all_metrics, mae
from src.evaluation.residual_diagnostics import _summarise
from src.features.build_features import FEATURES_DAILY_FILE
from src.features.weather_features import WEATHER_COLS
from src.utils.paths import PROCESSED_DIR

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

PRIMARY_SERIES = "val_test"   # final model, 393 contiguous days, ONE model (objection O2)
SECONDARY_SERIES = "oof"      # folds 2-5, 552 days, FOUR models — caveated in the caption
MAX_LAG = 35                  # five weekly cycles
FDR_Q = 0.10
ACF_ALPHA = 0.05              # Bartlett band

# Window length of the Stage 1 LSTM (CLAUDE.md Phase 4). Used only to turn a training-row
# count into a training-sequence count for the per-fold table; nothing is re-fit.
LSTM_WINDOW = 28

LSTM_VAL_TEST_FILE = PROCESSED_DIR / "lstm_val_test_predictions.parquet"
OOF_FILE = PROCESSED_DIR / "lstm_oof_predictions.parquet"
OOF_FOLD_LOG_FILE = PROCESSED_DIR / "lstm_oof_fold_log.parquet"
BASELINE_FILE = PROCESSED_DIR / "baseline_predictions.parquet"

# Fold 1 is degenerate (CLAUDE.md Phase 4: OOF std ratio 0.199x, correlation 0.300 with
# actuals) and was excluded from Phase 5's residual training set. The secondary OOF series
# here follows that same exclusion, so it is folds 2-5 only.
DEGENERATE_OOF_FOLD = 1

# The error chain, worst model first. lstm_alone and hybrid share Stage 1; xgboost_alone
# is the single-stage control on the identical feature matrix.
CHAIN = ["lstm_alone", "hybrid", "xgboost_alone"]

# lunes -> domingo. The raw column is unaccented ASCII (CLAUDE.md Phase 1).
WEEKDAY_ORDER = ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"]

# Weather feature suffixes promoted in Phase 3: lags 1/2/3/7 plus the 7-day rolling mean.
WEATHER_LAG_KINDS = ["lag_1", "lag_2", "lag_3", "lag_7", "roll_mean_7"]
# Partial correlations control for these four deterministic annual-cycle terms, which are
# already in the feature matrix and carry the seasonal signal shared by temperature and
# demand (objection O5).
ANNUAL_FOURIER_COLS = [
    "feat_fourier_annual_k1_sin",
    "feat_fourier_annual_k1_cos",
    "feat_fourier_annual_k2_sin",
    "feat_fourier_annual_k2_cos",
]

# --------------------------------------------------------------------------------------
# Acceptance criteria — fixed HERE, before any result is seen (objection O3, O7).
#
# A result landing between the "structure remains" and "near white noise" thresholds is
# reported as "weak/ambiguous structure": a legitimate outcome, not a failure to argue
# away. None of these is worded as evidence that the hybrid should have won — weather
# signal surviving in the residual means Stage 2 captured it only partially, since Stage 2
# received those features and still lost (objection O7).
# --------------------------------------------------------------------------------------

# Q1 — temporal memory
ACF_STRUCTURE_MIN = 0.20        # |ACF| at some lag 1..MAX_LAG >= this AND outside the band
ACF_WHITE_MAX = 0.10           # all |ACF| < this ...
BAND_EXCEEDANCE_WHITE_MAX = 2  # ... and <= this many lags outside the band (~5% of 35)

# Q2 — weekly seasonality (all three sub-conditions required for "structure")
ACF7_STRUCTURE_MIN = 0.20
ACF7_WHITE_MAX = 0.10
SPECTRAL_PEAK_RATIO_MIN = 3.0  # period-7 power share >= this * mean bin share
WEEKDAY_SPREAD_STRUCTURE_MIN = 0.30  # (max-min of weekday mean residual) / sigma
WEEKDAY_SPREAD_WHITE_MAX = 0.10

# Q3 — calendar concentration
CALENDAR_RATIO_STRUCTURE_MIN = 2.0   # mean |resid| festivo / laborable-ordinary
CALENDAR_RATIO_WHITE_MAX = 1.3

# Q4 — weather signal
WEATHER_PARTIAL_RHO_MIN = 0.15       # |partial rho| threshold for a "real" association
WEATHER_MIN_SIGNIFICANT = 5          # >= this many BH-significant features => structure

VERDICT_STRUCTURE = "estructura persiste"
VERDICT_WHITE = "ruido cuasi-blanco"
VERDICT_AMBIGUOUS = "estructura debil/ambigua"


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


def _features() -> pd.DataFrame:
    return pd.read_parquet(_require(FEATURES_DAILY_FILE))


def load_stage1_residuals(series: str = PRIMARY_SERIES) -> pd.DataFrame:
    """Stage 1 residuals as a contiguous, date-sorted frame.

    Args:
        series: PRIMARY_SERIES ('val_test') — the final model's 393 contiguous val+test
            days, one model, the honest input for ACF/spectral analysis. SECONDARY_SERIES
            ('oof') — folds 2-5 of the walk-forward OOF scheme (552 days), produced by
            four different fold models with level discontinuities at fold boundaries; use
            it only as a caveated cross-check.

    Returns:
        DataFrame [date, y_true, y_pred, residual, split, fold], sorted by date.
        `residual = y_true - y_pred`, so a positive residual is an under-prediction and
        the Stage 2 correction adds. `fold` is NaN for the val_test series.
    """
    if series == PRIMARY_SERIES:
        df = pd.read_parquet(_require(LSTM_VAL_TEST_FILE)).sort_values("date")
        out = pd.DataFrame(
            {
                "date": pd.to_datetime(df["date"].to_numpy()),
                "y_true": df["y_true"].to_numpy(dtype="float64"),
                "y_pred": df["y_pred_lstm"].to_numpy(dtype="float64"),
                "split": df["split"].to_numpy(),
                "fold": np.nan,
            }
        )
    elif series == SECONDARY_SERIES:
        df = pd.read_parquet(_require(OOF_FILE))
        df = df[(df["has_oof"]) & (df["fold"] != DEGENERATE_OOF_FOLD)].sort_values("date")
        out = pd.DataFrame(
            {
                "date": pd.to_datetime(df["date"].to_numpy()),
                "y_true": df["y_true"].to_numpy(dtype="float64"),
                "y_pred": df["y_pred_oof"].to_numpy(dtype="float64"),
                "split": "train",
                "fold": df["fold"].to_numpy(dtype="int64"),
            }
        )
    else:
        raise ValueError(
            f"load_stage1_residuals: series must be {PRIMARY_SERIES!r} or "
            f"{SECONDARY_SERIES!r}, got {series!r}"
        )

    out["residual"] = out["y_true"] - out["y_pred"]
    return out.reset_index(drop=True)


def _residual_array(series: str) -> np.ndarray:
    return load_stage1_residuals(series)["residual"].to_numpy(dtype="float64")


# --------------------------------------------------------------------------------------
# Analysis 1 — residual structure
# --------------------------------------------------------------------------------------


def autocorrelation_frame(x: np.ndarray, max_lag: int = MAX_LAG) -> pd.DataFrame:
    """ACF/PACF table for an arbitrary series — the disk-free core of the analysis.

    Returns:
        DataFrame [lag, acf, pacf, ci_low, ci_high, outside_band] for lags 1..max_lag.
        `ci_low`/`ci_high` are the Bartlett band around zero (recentred off the estimate);
        a lag is `outside_band` when |acf| exceeds the band half-width, which widens with
        lag, hence stored per lag rather than as a scalar.
    """
    acf_vals, acf_confint = acf(x, nlags=max_lag, alpha=ACF_ALPHA, fft=False)
    pacf_vals, _ = pacf(x, nlags=max_lag, alpha=ACF_ALPHA)

    # statsmodels centres confint on the estimate; recentre on zero for a band.
    band = acf_confint[:, 1] - acf_vals

    rows = []
    for lag in range(1, max_lag + 1):
        half = float(band[lag])
        rows.append(
            {
                "lag": lag,
                "acf": float(acf_vals[lag]),
                "pacf": float(pacf_vals[lag]),
                "ci_low": -half,
                "ci_high": half,
                "outside_band": abs(float(acf_vals[lag])) > half,
            }
        )
    return pd.DataFrame(rows)


def residual_autocorrelation(
    series: str = PRIMARY_SERIES, max_lag: int = MAX_LAG
) -> pd.DataFrame:
    """`autocorrelation_frame` applied to the Stage 1 residual of `series`."""
    return autocorrelation_frame(_residual_array(series), max_lag)


def ljung_box(
    series: str = PRIMARY_SERIES, lags: tuple[int, ...] = (7, 14, 28)
) -> pd.DataFrame:
    """Ljung-Box portmanteau statistic at the given lags.

    Reported as DESCRIPTIVE only (objection O4): the test's degrees-of-freedom correction
    is undefined for an LSTM with no meaningful parameter count, and on a residual that is
    visibly autocorrelated it will reject trivially. No claim in the memoria rests on
    these p-values; they accompany the effect-size measures, not replace them.

    Returns:
        DataFrame [lag, lb_stat, lb_pvalue].
    """
    x = _residual_array(series)
    table = acorr_ljungbox(x, lags=list(lags), return_df=True)
    return pd.DataFrame(
        {
            "lag": list(lags),
            "lb_stat": table["lb_stat"].to_numpy(dtype="float64"),
            "lb_pvalue": table["lb_pvalue"].to_numpy(dtype="float64"),
        }
    )


def weekday_residual_profile(series: str = PRIMARY_SERIES) -> pd.DataFrame:
    """Mean/median/spread of the residual by day of week.

    Returns:
        DataFrame [day_of_week_es, n, mean_residual, median_residual, sd,
        mean_abs_residual, mean_over_sigma], ordered lunes..domingo. `mean_over_sigma` is
        the mean residual as a fraction of the whole-series residual sigma — the effect
        size that carries the weekly-seasonality claim, since the significance tests are
        anti-conservative here (objection O4).
    """
    resid = load_stage1_residuals(series)
    feats = _features()[["date", "day_of_week_es"]]
    merged = resid.merge(feats, on="date", how="left")
    sigma = float(merged["residual"].std(ddof=1))

    rows = []
    for day in WEEKDAY_ORDER:
        sub = merged.loc[merged["day_of_week_es"] == day, "residual"]
        if sub.empty:
            continue
        rows.append(
            {
                "day_of_week_es": day,
                "n": int(sub.size),
                "mean_residual": float(sub.mean()),
                "median_residual": float(sub.median()),
                "sd": float(sub.std(ddof=1)),
                "mean_abs_residual": float(sub.abs().mean()),
                "mean_over_sigma": float(sub.mean() / sigma) if sigma else 0.0,
            }
        )
    return pd.DataFrame(rows)


def periodogram_frame(x: np.ndarray) -> pd.DataFrame:
    """Periodogram of the demeaned series — the disk-free core.

    Returns:
        DataFrame [period_days, frequency, power, power_share], sorted by descending
        power. The zero frequency is dropped. `power_share` sums to 1 across all bins, so
        a period-7 peak can be compared against the mean bin share (1 / n_bins).
    """
    freq, power = signal.periodogram(x, fs=1.0, detrend="constant")

    nonzero = freq > 0
    freq, power = freq[nonzero], power[nonzero]
    share = power / power.sum() if power.sum() else np.zeros_like(power)

    out = pd.DataFrame(
        {
            "period_days": 1.0 / freq,
            "frequency": freq,
            "power": power,
            "power_share": share,
        }
    )
    return out.sort_values("power", ascending=False).reset_index(drop=True)


def residual_periodogram(series: str = PRIMARY_SERIES) -> pd.DataFrame:
    """`periodogram_frame` applied to the Stage 1 residual of `series`."""
    return periodogram_frame(_residual_array(series))


def _period7_ratio(periodogram: pd.DataFrame) -> float:
    """Power share of the bin nearest period 7, as a multiple of the mean bin share."""
    idx = (periodogram["period_days"] - 7.0).abs().idxmin()
    peak_share = float(periodogram.loc[idx, "power_share"])
    mean_share = 1.0 / len(periodogram)
    return peak_share / mean_share if mean_share else 0.0


def residual_by_calendar_group(series: str = PRIMARY_SERIES) -> pd.DataFrame:
    """Residual magnitude by calendar regime.

    Reuses `day_type_breakdown.breakdown` for the day-type rows (positionally aligned, so
    the merge is reset_index'd first) and `residual_diagnostics._summarise` for the
    holiday-type supplement — neither grouping is reimplemented here.

    Returns:
        DataFrame [dimension, group, n, mean_abs_residual, mean_abs_pct], where
        `dimension` is 'day_type' or 'holiday_type'. `mean_abs_residual` is the group MAE
        of the Stage 1 residual (== mean |residual|, since residual = y_true - y_pred).
    """
    resid = load_stage1_residuals(series)
    all_feats = _features()
    holiday_cols = [c for c in all_feats.columns if c.startswith("feat_holiday_type_")]
    feats = all_feats[["date", DAY_TYPE_COL, BRIDGE_COL, *holiday_cols]]
    merged = resid.merge(feats, on="date", how="left").reset_index(drop=True)

    day_type = breakdown(merged["y_true"], merged["y_pred"], merged)
    day_type = day_type.rename(columns={"MAE": "mean_abs_residual", "MAPE": "mean_abs_pct"})
    day_type.insert(0, "dimension", "day_type")

    # Holiday-type supplement. _summarise expects abs_residual / abs_pct_error columns.
    holiday = merged.copy()
    holiday["residual"] = holiday["y_true"] - holiday["y_pred"]
    holiday["abs_residual"] = holiday["residual"].abs()
    holiday["abs_pct_error"] = holiday["abs_residual"] / holiday["y_true"] * 100.0
    label = pd.Series("none (not a holiday)", index=holiday.index, name="holiday_type")
    for col in [c for c in holiday.columns if c.startswith("feat_holiday_type_")]:
        pretty = col.replace("feat_holiday_type_", "").replace("_", " ")
        label = label.mask(holiday[col] == 1, pretty)
    hol = _summarise(holiday, label, "holiday_type").rename(
        columns={
            "mean_abs_error": "mean_abs_residual",
            "mean_abs_pct": "mean_abs_pct",
        }
    )
    hol = hol[["holiday_type", "n", "mean_abs_residual", "mean_abs_pct"]].rename(
        columns={"holiday_type": "group"}
    )
    hol.insert(0, "dimension", "holiday_type")

    cols = ["dimension", "group", "n", "mean_abs_residual", "mean_abs_pct"]
    return pd.concat([day_type[cols], hol[cols]], ignore_index=True)


def _calendar_concentration_ratio(calendar: pd.DataFrame) -> float:
    """mean |resid| on festivo / mean |resid| on laborable-ordinary."""
    by_group = calendar.set_index("group")["mean_abs_residual"]
    return float(by_group["festivo"] / by_group["laborable, ordinary"])


def _rank_residualise(values: np.ndarray, controls: np.ndarray) -> np.ndarray:
    """Return `rank(values)` with the linear fit on `rank(controls)` (+ intercept) removed."""
    ry = stats.rankdata(values)
    design = np.column_stack([np.ones(len(ry)), stats.rankdata(controls, axis=0)])
    beta, *_ = np.linalg.lstsq(design, ry, rcond=None)
    return ry - design @ beta


def residual_weather_correlation(
    series: str = PRIMARY_SERIES, q: float = FDR_Q
) -> pd.DataFrame:
    """Spearman correlation of the residual with each lagged weather feature.

    Reported raw AND partialled on the four annual Fourier terms, because both temperature
    and demand carry the annual cycle and a raw correlation would largely re-measure it
    (objection O5). Benjamini-Hochberg FDR at `q` controls the 105-way multiplicity. The
    GAP between the raw and partial coefficient is itself the finding.

    Same-day weather is never included: every column here is a `_lag_{1,2,3,7}` or
    `_roll_mean_7` feature, re-asserting the Phase 3 guard at the diagnostic layer.

    Returns:
        DataFrame [feature, weather_var, lag_kind, spearman_raw, p_raw,
        spearman_partial, p_partial, p_adj_bh, significant], sorted by descending
        |spearman_partial|. `significant` is the BH decision at `q`.
    """
    resid = load_stage1_residuals(series)
    feats = _features()
    merged = resid.merge(feats, on="date", how="left")

    weather_features = [
        f"feat_{var}_{kind}" for var in WEATHER_COLS for kind in WEATHER_LAG_KINDS
    ]
    weather_features = [c for c in weather_features if c in merged.columns]
    if any("_lag_0" in c or c in {f"feat_{v}" for v in WEATHER_COLS} for c in weather_features):
        raise ValueError("residual_weather_correlation: a same-day weather column leaked in")

    y = merged["residual"].to_numpy(dtype="float64")
    controls = merged[ANNUAL_FOURIER_COLS].to_numpy(dtype="float64")
    ry_resid = _rank_residualise(y, controls)
    n = len(y)
    df_partial = n - 2 - controls.shape[1]

    rows = []
    for col in weather_features:
        x = merged[col].to_numpy(dtype="float64")
        rho_raw, p_raw = stats.spearmanr(y, x)

        rx_resid = _rank_residualise(x, controls)
        rho_partial = float(np.corrcoef(ry_resid, rx_resid)[0, 1])
        # t-test on the partial correlation with the controls' df removed.
        if abs(rho_partial) >= 1.0:
            p_partial = 0.0
        else:
            t_stat = rho_partial * np.sqrt(df_partial / (1.0 - rho_partial**2))
            p_partial = float(2.0 * stats.t.sf(abs(t_stat), df_partial))

        var = col[len("feat_"):]
        for kind in WEATHER_LAG_KINDS:
            if var.endswith(f"_{kind}"):
                weather_var, lag_kind = var[: -len(kind) - 1], kind
                break
        else:  # pragma: no cover - guarded by the construction of weather_features
            weather_var, lag_kind = var, ""

        rows.append(
            {
                "feature": col,
                "weather_var": weather_var,
                "lag_kind": lag_kind,
                "spearman_raw": float(rho_raw),
                "p_raw": float(p_raw),
                "spearman_partial": rho_partial,
                "p_partial": p_partial,
            }
        )

    table = pd.DataFrame(rows)
    reject, p_adj, _, _ = multipletests(table["p_partial"], alpha=q, method="fdr_bh")
    table["p_adj_bh"] = p_adj
    table["significant"] = reject
    return table.sort_values(
        "spearman_partial", key=lambda s: s.abs(), ascending=False
    ).reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Analysis 2 — the error chain, quantified (NOT a decomposition; objection O6)
# --------------------------------------------------------------------------------------


def error_chain_table(split: str = "test") -> pd.DataFrame:
    """MAE/RMSE/MAPE/R2 for lstm_alone -> hybrid -> xgboost_alone, with step deltas.

    Built from `dashboard_data.load_all_predictions` + `metrics.all_metrics`, i.e. the
    exact code path behind chapter 5, so it cannot disagree with the master table.

    Returns:
        DataFrame [Modelo, MAE, RMSE, MAPE, R2, delta_MAE_abs, delta_MAE_pct,
        gap_share_pct]. `delta_*` are step-over-previous-row (negative = the step reduces
        MAE). `gap_share_pct` expresses each step as a share of the FULL lstm_alone ->
        xgboost_alone MAE gap; the non-zero rows sum to 100. The deliverable sentence:
        Stage 2 closes X% of that gap, and (100-X)% still separates the corrected hybrid
        from a direct XGBoost on the same features.
    """
    preds = load_all_predictions()
    subset = preds[preds["split"] == split]

    scores = {}
    for model in CHAIN:
        pair = subset[subset["model"] == model]
        if pair.empty:
            raise ValueError(f"error_chain_table: no {model!r} predictions on {split!r}")
        scores[model] = all_metrics(pair["y_true"], pair["y_pred"])

    full_gap = scores[CHAIN[0]]["MAE"] - scores[CHAIN[-1]]["MAE"]

    rows = []
    prev_mae = None
    for model in CHAIN:
        m = scores[model]
        delta_abs = 0.0 if prev_mae is None else m["MAE"] - prev_mae
        delta_pct = 0.0 if not prev_mae else 100.0 * delta_abs / prev_mae
        gap_share = 0.0 if (full_gap == 0 or delta_abs == 0) else -delta_abs / full_gap * 100.0
        rows.append(
            {
                "Modelo": model,
                "MAE": m["MAE"],
                "RMSE": m["RMSE"],
                "MAPE": m["MAPE"],
                "R2": m["R2"],
                "delta_MAE_abs": delta_abs,
                "delta_MAE_pct": delta_pct,
                "gap_share_pct": gap_share,
            }
        )
        prev_mae = m["MAE"]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Analysis 3a — per-fold sample-size evidence, difficulty-controlled (objection O1)
# --------------------------------------------------------------------------------------


def fold_size_vs_error() -> pd.DataFrame:
    """Per-fold OOF error vs training size, with a difficulty-controlled skill ratio.

    Raw fold MAE is confounded: fold size grows monotonically but fold MAE does not
    (1,461,905 -> 282,122 -> then back up to ~500k), because each fold also predicts a
    different, differently-hard period. `skill_ratio = mae / mae_seasonal_naive` on the
    IDENTICAL fold block removes that confound — if folds 4-5 are simply harder periods,
    the naive baseline degrades there too and the ratio flattens.

    This is 5 points, non-monotonic, and still confounded with which period each fold
    predicts. It is indicative context, NOT a learning curve (that is
    `lstm_learning_curve.py`).

    Returns:
        DataFrame [fold, train_rows, train_sequences, oof_rows, oof_start, oof_end, mae,
        mae_seasonal_naive_same_block, skill_ratio, epochs_run, best_epoch].
    """
    fold_log = pd.read_parquet(_require(OOF_FOLD_LOG_FILE)).set_index("fold")
    oof = pd.read_parquet(_require(OOF_FILE)).sort_values("date").reset_index(drop=True)
    baselines = pd.read_parquet(_require(BASELINE_FILE))[["date", "total", "seasonal_naive"]]

    rows = []
    for fold in sorted(f for f in oof["fold"].unique() if f != -1):
        block = oof[oof["fold"] == fold]
        # Position of the fold block's first row within the 889-row training series is the
        # size of the training slice that produced it (expanding-window scheme, oof.py).
        train_rows = int(block.index.min())
        naive = block[["date"]].merge(baselines, on="date", how="left")
        naive_mae = mae(naive["total"], naive["seasonal_naive"])
        fold_mae = float(fold_log.loc[fold, "mae"])
        rows.append(
            {
                "fold": int(fold),
                "train_rows": train_rows,
                "train_sequences": train_rows - LSTM_WINDOW,
                "oof_rows": int(len(block)),
                "oof_start": block["date"].min(),
                "oof_end": block["date"].max(),
                "mae": fold_mae,
                "mae_seasonal_naive_same_block": float(naive_mae),
                "skill_ratio": fold_mae / float(naive_mae),
                "epochs_run": int(fold_log.loc[fold, "epochs_run"]),
                "best_epoch": int(fold_log.loc[fold, "best_epoch"]),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Acceptance-criterion verdicts
# --------------------------------------------------------------------------------------


def evaluate_acceptance_criteria(series: str = PRIMARY_SERIES) -> pd.DataFrame:
    """Apply the pre-registered criteria mechanically to `series`.

    Returns:
        DataFrame [question, statistic, value, threshold_structure, threshold_white,
        verdict], one row per question. `verdict` is one of VERDICT_STRUCTURE,
        VERDICT_WHITE, VERDICT_AMBIGUOUS. Thresholds are the module constants, echoed into
        the table so a test can pin that they are not hard-coded twice.
    """
    acf_table = residual_autocorrelation(series)
    weekday = weekday_residual_profile(series)
    periodogram = residual_periodogram(series)
    calendar = residual_by_calendar_group(series)
    weather = residual_weather_correlation(series)

    sigma = float(load_stage1_residuals(series)["residual"].std(ddof=1))

    # Q1 — temporal memory
    max_abs_acf = float(acf_table["acf"].abs().max())
    max_lag_idx = acf_table["acf"].abs().idxmax()
    max_lag_outside = bool(acf_table.loc[max_lag_idx, "outside_band"])
    n_outside = int(acf_table["outside_band"].sum())
    if max_abs_acf >= ACF_STRUCTURE_MIN and max_lag_outside:
        q1 = VERDICT_STRUCTURE
    elif (acf_table["acf"].abs() < ACF_WHITE_MAX).all() and n_outside <= BAND_EXCEEDANCE_WHITE_MAX:
        q1 = VERDICT_WHITE
    else:
        q1 = VERDICT_AMBIGUOUS

    # Q2 — weekly seasonality
    acf7 = float(acf_table.loc[acf_table["lag"] == 7, "acf"].iloc[0])
    peak7_ratio = _period7_ratio(periodogram)
    weekday_spread = float(
        (weekday["mean_residual"].max() - weekday["mean_residual"].min()) / sigma
    )
    structure_q2 = (
        abs(acf7) >= ACF7_STRUCTURE_MIN
        and peak7_ratio >= SPECTRAL_PEAK_RATIO_MIN
        and weekday_spread >= WEEKDAY_SPREAD_STRUCTURE_MIN
    )
    white_q2 = (
        abs(acf7) < ACF7_WHITE_MAX
        and peak7_ratio < SPECTRAL_PEAK_RATIO_MIN
        and weekday_spread < WEEKDAY_SPREAD_WHITE_MAX
    )
    q2 = VERDICT_STRUCTURE if structure_q2 else VERDICT_WHITE if white_q2 else VERDICT_AMBIGUOUS

    # Q3 — calendar concentration
    ratio = _calendar_concentration_ratio(calendar)
    if ratio >= CALENDAR_RATIO_STRUCTURE_MIN:
        q3 = VERDICT_STRUCTURE
    elif ratio < CALENDAR_RATIO_WHITE_MAX:
        q3 = VERDICT_WHITE
    else:
        q3 = VERDICT_AMBIGUOUS

    # Q4 — weather signal
    n_sig = int(
        (weather["significant"] & (weather["spearman_partial"].abs() >= WEATHER_PARTIAL_RHO_MIN)).sum()
    )
    if n_sig >= WEATHER_MIN_SIGNIFICANT:
        q4 = VERDICT_STRUCTURE
    elif n_sig == 0:
        q4 = VERDICT_WHITE
    else:
        q4 = VERDICT_AMBIGUOUS

    return pd.DataFrame(
        [
            {
                "question": "Memoria temporal (ACF/PACF)",
                "statistic": f"max |ACF| lags 1-{MAX_LAG} = {max_abs_acf:.3f} "
                f"(lag {int(acf_table.loc[max_lag_idx, 'lag'])}, "
                f"{'fuera' if max_lag_outside else 'dentro'} de banda); "
                f"{n_outside}/{MAX_LAG} lags fuera de banda",
                "value": max_abs_acf,
                "threshold_structure": ACF_STRUCTURE_MIN,
                "threshold_white": ACF_WHITE_MAX,
                "verdict": q1,
            },
            {
                "question": "Estacionalidad semanal",
                "statistic": f"ACF(7) = {acf7:.3f}; pico espectral periodo-7 = "
                f"{peak7_ratio:.1f}x cuota media; dispersion de medias por dia = "
                f"{weekday_spread:.2f} sigma",
                "value": abs(acf7),
                "threshold_structure": ACF7_STRUCTURE_MIN,
                "threshold_white": ACF7_WHITE_MAX,
                "verdict": q2,
            },
            {
                "question": "Concentracion en calendario",
                "statistic": f"|residuo| festivo / laborable-ordinario = {ratio:.2f}x",
                "value": ratio,
                "threshold_structure": CALENDAR_RATIO_STRUCTURE_MIN,
                "threshold_white": CALENDAR_RATIO_WHITE_MAX,
                "verdict": q3,
            },
            {
                "question": "Senal meteorologica (parcial, BH)",
                "statistic": f"{n_sig} variables con |rho parcial| >= "
                f"{WEATHER_PARTIAL_RHO_MIN} y significativas tras BH (q={FDR_Q})",
                "value": float(n_sig),
                "threshold_structure": float(WEATHER_MIN_SIGNIFICANT),
                "threshold_white": 0.0,
                "verdict": q4,
            },
        ]
    )


# --------------------------------------------------------------------------------------
# Aggregate
# --------------------------------------------------------------------------------------


def summarise() -> dict[str, pd.DataFrame]:
    """Every table this module produces, for the report and the tests."""
    return {
        "autocorrelation": residual_autocorrelation(),
        "ljung_box": ljung_box(),
        "weekday_profile": weekday_residual_profile(),
        "periodogram": residual_periodogram(),
        "calendar_group": residual_by_calendar_group(),
        "weather_correlation": residual_weather_correlation(),
        "error_chain_test": error_chain_table("test"),
        "error_chain_val": error_chain_table("val"),
        "fold_size": fold_size_vs_error(),
        "acceptance": evaluate_acceptance_criteria(),
    }


def main() -> None:
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)

    print("=" * 88)
    print("STAGE 1 RESIDUAL ANATOMY — diagnostic control, NOT a candidate model")
    print("=" * 88)
    resid = load_stage1_residuals()
    print(
        f"primary series : {PRIMARY_SERIES}  "
        f"({resid['date'].min():%Y-%m-%d} -> {resid['date'].max():%Y-%m-%d}, "
        f"{len(resid)} contiguous days, one model)"
    )
    print(f"residual sigma : {resid['residual'].std(ddof=1):,.0f}  "
          f"(signal sigma {resid['y_true'].std(ddof=1):,.0f})")

    tables = summarise()

    print("\n--- ACF / PACF (lags 1, 7, 14, 21, 28) ---")
    acf_table = tables["autocorrelation"]
    print(acf_table[acf_table["lag"].isin([1, 7, 14, 21, 28])].to_string(index=False))

    print("\n--- Ljung-Box (descriptive; see O4) ---")
    print(tables["ljung_box"].to_string(index=False))

    print("\n--- Weekday residual profile ---")
    print(tables["weekday_profile"].to_string(index=False))

    print("\n--- Periodogram (top 5 bins by power) ---")
    print(tables["periodogram"].head(5).to_string(index=False))

    print("\n--- Residual by calendar group ---")
    print(tables["calendar_group"].to_string(index=False))

    print("\n--- Residual vs lagged weather (top 10 by |partial rho|) ---")
    print(tables["weather_correlation"].head(10).to_string(index=False))
    n_sig = int(tables["weather_correlation"]["significant"].sum())
    print(f"BH-significant partial correlations (q={FDR_Q}): {n_sig} / "
          f"{len(tables['weather_correlation'])}")

    for split in ("test", "val"):
        print(f"\n--- Error chain - {split.upper()} split (NOT a decomposition) ---")
        print(tables[f"error_chain_{split}"].to_string(index=False))
    chain = tables["error_chain_test"].set_index("Modelo")
    closed = chain.loc["hybrid", "gap_share_pct"]
    print(
        f"\nDeliverable: Stage 2 closes {closed:.1f}% of the lstm_alone -> xgboost_alone "
        f"test-MAE gap; {100 - closed:.1f}% "
        f"({chain.loc['xgboost_alone', 'delta_MAE_abs']:,.0f} MAE) still separates the "
        "corrected hybrid from a direct XGBoost."
    )

    print("\n--- Per-fold size vs error (5 points, non-monotonic, confounded - O1) ---")
    print(tables["fold_size"].to_string(index=False))

    print("\n" + "=" * 88)
    print("ACCEPTANCE-CRITERION VERDICTS (thresholds fixed before any result was seen)")
    print("=" * 88)
    print(tables["acceptance"].to_string(index=False))


if __name__ == "__main__":
    main()
