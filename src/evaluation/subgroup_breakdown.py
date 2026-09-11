"""Phase 9 — subgroup and period breakdown of the hybrid's performance (§5.11).

DIAGNOSTIC RE-ANALYSIS, NOT A CANDIDATE MODEL. Nothing here trains, refits or re-predicts
anything: every number comes from `data/processed/*_predictions.parquet` via
`dashboard_data.load_all_predictions` and is scored by `metrics.all_metrics`, the exact
path behind chapter 5. No artifact of this phase enters the master comparison,
`dashboard_data.MODEL_NAMES`, `full_comparison.parquet`, `sensitivity_comparison.parquet`
or `models/`; `test_phase9_adds_no_model_to_the_master_comparison` pins that mechanically.

THE QUESTION (from the supervisor). "El híbrido puede no ganar globalmente, pero tener
algún comportamiento diferente durante festivos, puentes, fines de semana o periodos de
demanda anómala." This module searches that systematically and reports it honestly.

THE DOMINANT RISK IS P-HACKING, not incompleteness. Searching subgroups until one favours
the hybrid is p-hacking regardless of who asked for the search. The design below is fixed
BEFORE any MAE, ranking or Δ was computed (only the n-counts were), so that "the hybrid
wins in no subgroup" is as publishable as any positive finding:

  * `SUBGROUPS` and every threshold are module constants with pinned n-counts (§1).
  * Test is the verdict split; val is an independent replication requirement; a subgroup
    counts as a win only if the hybrid ranks first on BOTH (`VERDICT_SCOPE` /
    `REPLICATION_SCOPE`). This converts a ~9-way search into one requiring replication.
  * Uncertainty is a moving-block paired bootstrap with `best_model`/`best_hybrid`
    re-selected inside every replicate (the double selection is priced in, not ignored).
  * Benjamini-Hochberg at `FDR_Q` over the inferential family, with the positive
    dependence (PRDS) among overlapping subgroups stated rather than assumed.
  * `demanda_anomala` uses observed `total_t`: legitimate for post-hoc reporting,
    illegitimate as a routing rule — §5.11 says so, so no reader concludes "deploy the
    hybrid on anomalous days".

`demanda_anomala` — VERDICT SCOPE RESOLUTION (pre-registered, stated before any result).
This subgroup has n=16 on test and n=34 on val, so it falls below `MIN_N_INFERENTIAL` on
the verdict split and its pooled `val_test` view reaches n=50. It receives
`VERDICT_NON_INFER` on the test verdict scope, exactly like any other subgroup below the
threshold — NO exception, even though it is the one subgroup the supervisor named
explicitly. Its pooled `val_test` result is reported as a pre-registered SECONDARY,
descriptive scope with the full O6 caveat (SARIMAX order, XGBoost hyperparameters and the
ensemble weights were all selected on val), never carrying a verdict or a headline. The
pooled scope was available and was DELIBERATELY NOT promoted to a verdict scope for this
subgroup: choosing per-subgroup which split decides is a researcher degree of freedom, and
adopting it for the one subgroup the supervisor asked about is precisely where it would be
least defensible. `evaluate_subgroup_criteria` therefore reads `VERDICT_SCOPE` uniformly
for every subgroup and no code path assigns a per-subgroup verdict scope.

Computation only, no plotting — the Plotly builders live in `memoria_figures.py`, mirroring
the `stage1_residual_anatomy.py` / `memoria_figures.py` split.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from src.evaluation.dashboard_data import (
    MODEL_NAMES,
    SMALL_SAMPLE_THRESHOLD,
    _require,
    load_all_predictions,
)
from src.evaluation.day_type_breakdown import BRIDGE_COL, DAY_TYPE_COL
from src.evaluation.metrics import all_metrics
from src.features.build_features import FEATURES_DAILY_FILE
from src.utils.paths import PROCESSED_DIR
from src.utils.seed import GLOBAL_SEED
from src.utils.splits import chronological_split, split_masks

# --------------------------------------------------------------------------------------
# 1. Pre-registered design — fixed before any MAE / ranking / Δ was computed
# --------------------------------------------------------------------------------------

# 1.2 — anomalous demand. Day t is anomalous iff |total_t - total_{t-ANOMALY_LAG}| exceeds
# ANOMALY_K robust deviations, with the scale (MAD) estimated on the TRAIN split ONLY so
# the definition is not fitted to evaluation data. The lag-7 (weekday-matched) rule is
# pre-registered; a rolling-median-28 variant was prototyped on train and rejected because
# weekly seasonality dominates its deviation, making it a weekend detector that would
# silently duplicate `domingo` (objection O1). The rejected variant is not reported.
ANOMALY_LAG = 7
ANOMALY_K = 3.0

# 1.3 — minimum n for an inferential (win/loss) claim. Justified by single-day leverage: at
# n=20 one day carries <=5% of the group mean; at n=5, 20%; at n=3, 33%. Any subgroup below
# this is computed and tabulated with its n, marked non-inferential, and excluded from
# every win/loss claim. `SMALL_SAMPLE_THRESHOLD` (10, imported) is the lower floor at which
# a point MAE is not even tabulated as a ranking.
MIN_N_INFERENTIAL = 20

# 1.4 — scopes and the verdict split. Test is the verdict split; val is the replication
# requirement; `val_test` (the pooled 393 contiguous days) is a pre-registered SECONDARY
# scope, used descriptively for subgroups that fail MIN_N_INFERENTIAL on test. Its cost
# (O6): SARIMAX order, XGBoost hyperparameters and the ensemble weights were selected on
# val, so pooled results are labelled secondary and never carry a headline.
SCOPES = ["test", "val", "val_test"]
VERDICT_SCOPE = "test"
REPLICATION_SCOPE = "val"

# 3.1 — moving-block paired bootstrap. Block length 7 preserves the weekly dependence that
# makes a naive paired t-test overstate significance. Re-selection of best_model /
# best_hybrid happens inside every replicate (O3).
BLOCK_LENGTH = 7
N_BOOTSTRAP = 2000
USABLE_RESAMPLE_MIN = 0.90  # suppress a CI below this fraction of usable replicates

# 3.2 — Benjamini-Hochberg. Same call and same q as §5.9's weather analysis.
FDR_Q = 0.10

# 4 — period-level rolling view. 28-day window, inherited from the LSTM's W=28 and the
# rolling-feature window used throughout; not newly tuned.
ROLLING_WINDOW = 28

# The two model families whose Δ is the focal statistic. `best_model` is the argmin over
# competitors; `best_hybrid` the argmin over these two.
HYBRID_MODELS = ["hybrid", "hybrid_weighted"]
COMPETITOR_MODELS = [m for m in MODEL_NAMES if m not in HYBRID_MODELS]

# 1.1 — the pre-registered subgroups. Built from columns already in
# `features_daily.parquet` (`day_type`, `feat_is_bridge_day`, `feat_is_weekend`,
# `holiday_name`) — no new data source. The key set is FROZEN.
SUBGROUPS: dict[str, str] = {
    "laborable_ordinario": "Laborable sin puente",
    "laborable_puente": "Laborable en puente detectado",
    "sabado": "Sábado",
    "domingo": "Domingo",
    "festivo": "Festivo (day_type)",
    "fin_de_semana": "Sábado o domingo",
    "entre_semana": "Laborable o festivo (lunes-viernes de calendario)",
    "calendario_irregular": "Festivo o puente",
    "calendario_ordinario": "Ni festivo ni puente",
    "verano_jul_ago": "Julio y agosto",
    "resto_del_anio": "Resto del año",
    "periodo_navidad": "22 dic - 6 ene",
    "periodo_semana_santa": "Jueves Santo +/- 4 días",
    "demanda_anomala": "|total_t - total_{t-7}| > 3 MAD_train",
    "demanda_regular": "Demanda no anómala",
}

# O5 — the 393-day evaluation window contains exactly one Christmas (val only, test n=0)
# and one Easter (test only, val n=0). No definition fixes this; it is a property of the
# split. These two are reported descriptively and marked VERDICT_EXCLUDED: no win/loss
# claim about Christmas or Easter is possible in this thesis.
EXCLUDED_SUBGROUPS = frozenset({"periodo_navidad", "periodo_semana_santa"})

# n-counts pinned during planning (the ONLY quantities computed before the design froze).
# A drift in any subgroup definition is caught by
# `test_subgroup_sizes_match_the_preregistered_counts`.
PREREGISTERED_N: dict[str, dict[str, int]] = {
    "laborable_ordinario": {"val": 121, "test": 133},
    "laborable_puente": {"val": 12, "test": 3},
    "sabado": {"val": 26, "test": 27},
    "domingo": {"val": 28, "test": 29},
    "festivo": {"val": 9, "test": 5},
    "fin_de_semana": {"val": 54, "test": 56},
    "entre_semana": {"val": 142, "test": 141},
    "calendario_irregular": {"val": 21, "test": 8},
    "calendario_ordinario": {"val": 175, "test": 189},
    "verano_jul_ago": {"val": 57, "test": 33},
    "resto_del_anio": {"val": 139, "test": 164},
    "periodo_navidad": {"val": 16, "test": 0},
    "periodo_semana_santa": {"val": 0, "test": 10},
    "demanda_anomala": {"val": 34, "test": 16},
    "demanda_regular": {"val": 162, "test": 181},
}

# 3.3 — the reading, pre-stated as verdict constants. Only VERDICT_WIN may be described in
# prose as the hybrid winning.
VERDICT_WIN = "gana con evidencia"          # ranks 1st on test AND val, n>=MIN_N on both,
#                                             BH-significant on test
VERDICT_RANK_ONLY = "primero sin evidencia"  # ranks 1st but fails replication or BH
VERDICT_NO_WIN = "no gana"
VERDICT_NON_INFER = "no inferencial (n insuficiente)"
VERDICT_EXCLUDED = "no analizable (sin solape val-test)"

SUBGROUP_BREAKDOWN_FILE: Path = PROCESSED_DIR / "subgroup_breakdown.parquet"
SUBGROUP_ROLLING_MAE_FILE: Path = PROCESSED_DIR / "subgroup_rolling_mae.parquet"

_SCOPE_SPLITS = {"test": ("test",), "val": ("val",), "val_test": ("val", "test")}


# --------------------------------------------------------------------------------------
# Subgroup membership
# --------------------------------------------------------------------------------------


def _features() -> pd.DataFrame:
    return pd.read_parquet(_require(FEATURES_DAILY_FILE)).sort_values("date").reset_index(
        drop=True
    )


def _train_mad(deltas: pd.Series, train_mask: np.ndarray) -> float:
    """Robust scale (1.4826 * MAD) of the lag-7 difference over TRAIN rows only.

    The mask is applied here, before the median, so no validation or test row can enter
    the anomaly scale (CLAUDE.md leakage rule 3, applied to a definition rather than a
    scaler). `test_anomaly_scale_is_estimated_on_train_only` guards this both in source
    text and numerically.
    """
    train_deltas = deltas[train_mask].dropna()
    return float(1.4826 * (train_deltas - train_deltas.median()).abs().median())


def subgroup_masks(features: pd.DataFrame | None = None) -> pd.DataFrame:
    """`[date, <one bool column per SUBGROUPS key>]` over all 1310 days.

    Every column is a deterministic function of calendar/holiday features already in
    `features_daily.parquet`, except `demanda_anomala`/`demanda_regular`, whose threshold
    uses a TRAIN-only robust scale (see `_train_mad`).
    """
    feats = features if features is not None else _features()
    bounds = chronological_split(feats["date"])
    split = split_masks(feats, bounds)
    train_mask = split["train"].to_numpy()

    day_type = feats[DAY_TYPE_COL].astype(str)
    bridge = feats[BRIDGE_COL] == 1
    month = feats["date"].dt.month

    m: dict[str, pd.Series] = {}
    m["laborable_ordinario"] = (day_type == "laborable") & ~bridge
    m["laborable_puente"] = (day_type == "laborable") & bridge
    m["sabado"] = day_type == "sabado"
    m["domingo"] = day_type == "domingo"
    m["festivo"] = day_type == "festivo"
    m["fin_de_semana"] = m["sabado"] | m["domingo"]
    m["entre_semana"] = m["laborable_ordinario"] | m["laborable_puente"] | m["festivo"]
    m["calendario_irregular"] = m["festivo"] | m["laborable_puente"]
    m["calendario_ordinario"] = ~m["calendario_irregular"]
    m["verano_jul_ago"] = month.isin([7, 8])
    m["resto_del_anio"] = ~m["verano_jul_ago"]

    md = feats["date"]
    m["periodo_navidad"] = ((md.dt.month == 12) & (md.dt.day >= 22)) | (
        (md.dt.month == 1) & (md.dt.day <= 6)
    )

    holiday_name = feats["holiday_name"].astype(str)
    jueves_santo = feats.loc[holiday_name.str.contains("Jueves Santo", na=False), "date"]
    easter = pd.Series(False, index=feats.index)
    for anchor in jueves_santo:
        easter |= (md >= anchor - pd.Timedelta(days=4)) & (
            md <= anchor + pd.Timedelta(days=4)
        )
    m["periodo_semana_santa"] = easter

    delta7 = feats["total"] - feats["total"].shift(ANOMALY_LAG)
    mad_train = _train_mad(delta7, train_mask)
    anomala = (delta7.abs() > ANOMALY_K * mad_train).fillna(False)
    m["demanda_anomala"] = anomala
    m["demanda_regular"] = delta7.notna() & ~anomala

    out = pd.DataFrame({"date": feats["date"].to_numpy()})
    for key in SUBGROUPS:
        out[key] = m[key].to_numpy(dtype=bool)
    return out


# --------------------------------------------------------------------------------------
# Evaluation matrix (contiguous, one column per model)
# --------------------------------------------------------------------------------------


def _eval_matrix(
    scope: str, predictions: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Wide predictions for `scope`: `(pred[date x model], y_true[date], models)`.

    `pred` is date-sorted and contiguous (val and test abut). Only models with full,
    NaN-free coverage over the scope index are kept, in MODEL_NAMES order.
    """
    if scope not in _SCOPE_SPLITS:
        raise ValueError(f"_eval_matrix: scope must be one of {SCOPES}, got {scope!r}")

    df = predictions if predictions is not None else load_all_predictions()
    df = df[df["split"].isin(_SCOPE_SPLITS[scope])]

    y_true = (
        df.drop_duplicates("date").set_index("date")["y_true"].sort_index().astype(float)
    )
    pred = (
        df.pivot_table(index="date", columns="model", values="y_pred", aggfunc="first")
        .reindex(y_true.index)
    )
    models = [m for m in MODEL_NAMES if m in pred.columns and pred[m].notna().all()]
    return pred[models], y_true, models


def _mask_frame(dates: pd.Index, masks: pd.DataFrame | None = None) -> pd.DataFrame:
    """Subgroup bool columns aligned to `dates` (a DatetimeIndex)."""
    masks = masks if masks is not None else subgroup_masks()
    return masks.set_index("date").reindex(dates).fillna(False)


# --------------------------------------------------------------------------------------
# 2. Systematic comparison
# --------------------------------------------------------------------------------------


def _safe_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    try:
        return all_metrics(y_true, y_pred)
    except ValueError:
        err = np.abs(y_true - y_pred)
        return {
            "n": float(len(y_true)),
            "MAE": float(err.mean()) if len(err) else float("nan"),
            "RMSE": float(np.sqrt((err**2).mean())) if len(err) else float("nan"),
            "MAPE": float("nan"),
            "R2": float("nan"),
        }


def subgroup_metrics(
    scope: str = "test", predictions: pd.DataFrame | None = None
) -> pd.DataFrame:
    """`[subgroup, scope, n, model, MAE, RMSE, MAPE, R2, rank_MAE, inferential]`.

    Every model with coverage on `scope`, every subgroup with at least one day. Metrics
    come from `metrics.all_metrics` unchanged (so nothing can diverge from chapter 5).
    `rank_MAE` is 1 = lowest MAE among models; it is left NaN for a subgroup below
    `SMALL_SAMPLE_THRESHOLD`, where a point MAE is not tabulated as a ranking. `inferential`
    is `n >= MIN_N_INFERENTIAL` on this scope.
    """
    pred, y_true, models = _eval_matrix(scope, predictions)
    masks = _mask_frame(pred.index)
    yt = y_true.to_numpy()

    rows = []
    for subgroup in SUBGROUPS:
        sel = masks[subgroup].to_numpy(dtype=bool)
        n = int(sel.sum())
        if n == 0:
            continue
        block = []
        for model in models:
            scores = _safe_metrics(yt[sel], pred[model].to_numpy()[sel])
            block.append(
                {
                    "subgroup": subgroup,
                    "scope": scope,
                    "n": n,
                    "model": model,
                    "MAE": scores["MAE"],
                    "RMSE": scores["RMSE"],
                    "MAPE": scores["MAPE"],
                    "R2": scores["R2"],
                    "inferential": n >= MIN_N_INFERENTIAL,
                }
            )
        frame = pd.DataFrame(block)
        if n >= SMALL_SAMPLE_THRESHOLD:
            frame["rank_MAE"] = frame["MAE"].rank(method="min").astype(int)
        else:
            frame["rank_MAE"] = np.nan
        rows.append(frame)

    return pd.concat(rows, ignore_index=True)[
        ["subgroup", "scope", "n", "model", "MAE", "RMSE", "MAPE", "R2", "rank_MAE",
         "inferential"]
    ]


def hybrid_win_table(
    scope: str = "test", predictions: pd.DataFrame | None = None
) -> pd.DataFrame:
    """THE DELIVERABLE — one row per subgroup.

    `[subgroup, n, inferential, best_model, best_MAE, best_hybrid, best_hybrid_MAE,
    delta_MAE, delta_pct, hybrid_ranks_first]`. `best_model` is the argmin MAE over
    `COMPETITOR_MODELS`; `best_hybrid` the argmin over `HYBRID_MODELS`. `delta_MAE > 0`
    means the best hybrid is worse than the best competitor. `hybrid_ranks_first` is
    `best_hybrid_MAE < best_MAE` — i.e. a hybrid has the lowest MAE of all models.
    """
    metrics = subgroup_metrics(scope, predictions)
    rows = []
    for subgroup, block in metrics.groupby("subgroup", sort=False):
        mae_by_model = block.set_index("model")["MAE"]
        competitors = mae_by_model[mae_by_model.index.isin(COMPETITOR_MODELS)]
        hybrids = mae_by_model[mae_by_model.index.isin(HYBRID_MODELS)]
        if competitors.empty or hybrids.empty:
            continue
        best_model = competitors.idxmin()
        best_hybrid = hybrids.idxmin()
        best_mae = float(competitors.min())
        best_hybrid_mae = float(hybrids.min())
        delta = best_hybrid_mae - best_mae
        rows.append(
            {
                "subgroup": subgroup,
                "n": int(block["n"].iloc[0]),
                "inferential": bool(block["inferential"].iloc[0]),
                "best_model": best_model,
                "best_MAE": best_mae,
                "best_hybrid": best_hybrid,
                "best_hybrid_MAE": best_hybrid_mae,
                "delta_MAE": delta,
                "delta_pct": 100.0 * delta / best_mae if best_mae else float("nan"),
                "hybrid_ranks_first": best_hybrid_mae < best_mae,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# 4. Period-level rolling view (descriptive only — O7)
# --------------------------------------------------------------------------------------


def rolling_mae(
    window: int = ROLLING_WINDOW, predictions: pd.DataFrame | None = None
) -> pd.DataFrame:
    """`[date, model, MAE, n_window]` over the continuous val+test span.

    Consecutive windows overlap by `window - 1` days, so no test is run on this series and
    the figure caption says so (O7). `date` is the last day of each window.
    """
    pred, y_true, models = _eval_matrix("val_test", predictions)
    yt = y_true.to_numpy()
    rows = []
    for model in models:
        err = np.abs(pred[model].to_numpy() - yt)
        roll = pd.Series(err, index=pred.index).rolling(window).mean()
        valid = roll.notna()
        rows.append(
            pd.DataFrame(
                {
                    "date": pred.index[valid],
                    "model": model,
                    "MAE": roll[valid].to_numpy(),
                    "n_window": window,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------------------
# 3. Multiple-comparison discipline and uncertainty
# --------------------------------------------------------------------------------------


def _moving_block_indices(
    n_days: int, block: int, n_boot: int, rng: np.random.Generator
) -> np.ndarray:
    """`(n_boot, n_days)` integer index into a contiguous day series, moving blocks."""
    n_blocks = int(np.ceil(n_days / block))
    starts = rng.integers(0, n_days - block + 1, size=(n_boot, n_blocks))
    offsets = np.arange(block)
    idx = (starts[:, :, None] + offsets[None, None, :]).reshape(n_boot, n_blocks * block)
    return idx[:, :n_days]


def bootstrap_delta(
    scope: str = "test",
    subgroups: list[str] | None = None,
    n_boot: int = N_BOOTSTRAP,
    block: int = BLOCK_LENGTH,
    seed: int = GLOBAL_SEED,
    predictions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Moving-block paired bootstrap of the subgroup ΔMAE, with BH correction.

    `[subgroup, n, delta_MAE, ci_low, ci_high, p_raw, p_adj_bh, significant,
    frac_resamples_usable]`. Resampling is on the FULL contiguous evaluation series (a
    categorical subgroup is scattered across it); subgroup membership travels with the day,
    and `best_model` / `best_hybrid` are re-selected inside each replicate (O3). A replicate
    contributing fewer than `MIN_N_INFERENTIAL` subgroup days is discarded; the CI is
    suppressed below `USABLE_RESAMPLE_MIN`. BH runs over the inferential family only.
    """
    pred, y_true, models = _eval_matrix(scope, predictions)
    masks = _mask_frame(pred.index)
    yt = y_true.to_numpy()

    ae = np.abs(pred.to_numpy() - yt[:, None])  # (n_days, n_models)
    comp_cols = [i for i, m in enumerate(models) if m in COMPETITOR_MODELS]
    hyb_cols = [i for i, m in enumerate(models) if m in HYBRID_MODELS]
    if not comp_cols or not hyb_cols:
        raise ValueError("bootstrap_delta: scope lacks competitor or hybrid coverage")

    win = hybrid_win_table(scope, predictions).set_index("subgroup")
    if subgroups is None:
        subgroups = [
            s for s in SUBGROUPS
            if s not in EXCLUDED_SUBGROUPS
            and s in win.index
            and bool(win.loc[s, "inferential"])
        ]

    n_days = len(pred.index)
    rng = np.random.default_rng(seed)
    boot_idx = _moving_block_indices(n_days, block, n_boot, rng)

    rows = []
    for subgroup in subgroups:
        sel = masks[subgroup].to_numpy(dtype=bool)
        obs_delta = float(win.loc[subgroup, "delta_MAE"]) if subgroup in win.index else float("nan")

        sel_boot = sel[boot_idx]  # (n_boot, n_days)
        usable = sel_boot.sum(axis=1) >= MIN_N_INFERENTIAL
        deltas = np.full(n_boot, np.nan)
        for b in np.flatnonzero(usable):
            rows_b = boot_idx[b][sel_boot[b]]
            block_ae = ae[rows_b]
            best_comp = block_ae[:, comp_cols].mean(axis=0).min()
            best_hyb = block_ae[:, hyb_cols].mean(axis=0).min()
            deltas[b] = best_hyb - best_comp

        frac_usable = float(usable.mean())
        drawn = deltas[~np.isnan(deltas)]
        if drawn.size:
            p_raw = float(
                2.0 * min((drawn <= 0).mean(), (drawn >= 0).mean())
            )
            p_raw = min(p_raw, 1.0)
            if frac_usable >= USABLE_RESAMPLE_MIN:
                ci_low, ci_high = (float(x) for x in np.percentile(drawn, [2.5, 97.5]))
            else:
                ci_low = ci_high = float("nan")
        else:
            p_raw = float("nan")
            ci_low = ci_high = float("nan")

        rows.append(
            {
                "subgroup": subgroup,
                "n": int(win.loc[subgroup, "n"]) if subgroup in win.index else 0,
                "delta_MAE": obs_delta,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_raw": p_raw,
                "frac_resamples_usable": frac_usable,
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        table["p_adj_bh"] = []
        table["significant"] = []
        return table

    finite = table["p_raw"].notna()
    table["p_adj_bh"] = np.nan
    table["significant"] = False
    if finite.any():
        reject, p_adj, _, _ = multipletests(
            table.loc[finite, "p_raw"], alpha=FDR_Q, method="fdr_bh"
        )
        table.loc[finite, "p_adj_bh"] = p_adj
        table.loc[finite, "significant"] = reject
    return table[
        ["subgroup", "n", "delta_MAE", "ci_low", "ci_high", "p_raw", "p_adj_bh",
         "significant", "frac_resamples_usable"]
    ]


# --------------------------------------------------------------------------------------
# 3.3 / 5 — the pre-registered reading
# --------------------------------------------------------------------------------------


def evaluate_subgroup_criteria(
    predictions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """`[subgroup, n_test, n_val, verdict]` — the pre-registered reading (§5.11.2).

    The verdict scope is `VERDICT_SCOPE` for EVERY subgroup; `REPLICATION_SCOPE` must agree
    for a win. No per-subgroup verdict scope exists — see the module docstring on
    `demanda_anomala`. A subgroup wins (`VERDICT_WIN`) only if a hybrid ranks first on the
    verdict scope AND on the replication scope, with `n >= MIN_N_INFERENTIAL` on both and a
    BH-significant Δ on the verdict scope.
    """
    win_verdict = hybrid_win_table(VERDICT_SCOPE, predictions).set_index("subgroup")
    win_repl = hybrid_win_table(REPLICATION_SCOPE, predictions).set_index("subgroup")
    boot = bootstrap_delta(VERDICT_SCOPE, predictions=predictions).set_index("subgroup")

    rows = []
    for subgroup in SUBGROUPS:
        n_test = PREREGISTERED_N[subgroup]["test"]
        n_val = PREREGISTERED_N[subgroup]["val"]

        if subgroup in EXCLUDED_SUBGROUPS:
            verdict = VERDICT_EXCLUDED
        elif n_test < MIN_N_INFERENTIAL:
            verdict = VERDICT_NON_INFER
        else:
            first_verdict = subgroup in win_verdict.index and bool(
                win_verdict.loc[subgroup, "hybrid_ranks_first"]
            )
            first_repl = subgroup in win_repl.index and bool(
                win_repl.loc[subgroup, "hybrid_ranks_first"]
            )
            replicated = first_repl and n_val >= MIN_N_INFERENTIAL
            bh_significant = subgroup in boot.index and bool(
                boot.loc[subgroup, "significant"]
            )
            if first_verdict and replicated and bh_significant:
                verdict = VERDICT_WIN
            elif first_verdict:
                verdict = VERDICT_RANK_ONLY
            else:
                verdict = VERDICT_NO_WIN

        rows.append(
            {"subgroup": subgroup, "n_test": n_test, "n_val": n_val, "verdict": verdict}
        )
    return pd.DataFrame(rows)


def expected_first_by_chance(scope: str = VERDICT_SCOPE) -> dict[str, float]:
    """Reference arithmetic for a "hybrid ranks first by chance" count. NOT reported.

    Computes `family_size * len(HYBRID_MODELS) / len(MODEL_NAMES)` (9 * 2/11 on the test
    verdict scope). It was once quoted in §5.11.3 as the null expectation, and was removed
    from the memoria at the supervisor's review (item 21): the figure presupposes that the
    eleven models are exchangeable and equiprobable under the null, which does not hold
    for a family that mixes naive baselines with tuned models. The bootstrap deltas and the
    BH-corrected comparisons carry the inference on their own. The helper is kept only
    because its arithmetic is pinned by a test; nothing in the report or the figures
    reads it.
    """
    family = [
        s for s in SUBGROUPS
        if s not in EXCLUDED_SUBGROUPS
        and PREREGISTERED_N[s][scope] >= MIN_N_INFERENTIAL
    ]
    n_models = len(MODEL_NAMES)
    n_hybrid = len(HYBRID_MODELS)
    p_null = n_hybrid / n_models
    return {
        "family_size": float(len(family)),
        "n_models": float(n_models),
        "n_hybrid_models": float(n_hybrid),
        "p_null": p_null,
        "expected_first": len(family) * p_null,
    }


def month_breakdown(
    scope: str = "val_test", predictions: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Descriptive-only per-month MAE for every covered model. NOT in the win/loss family.

    val and test cover disjoint months, so this is an annex view (§1.1) — it carries no
    verdict.
    """
    pred, y_true, models = _eval_matrix(scope, predictions)
    yt = y_true.to_numpy()
    month = pred.index.month
    rows = []
    for mo in sorted(set(month)):
        sel = month == mo
        for model in models:
            scores = _safe_metrics(yt[sel], pred[model].to_numpy()[sel])
            rows.append(
                {
                    "month": int(mo),
                    "n": int(sel.sum()),
                    "model": model,
                    "MAE": scores["MAE"],
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Aggregate
# --------------------------------------------------------------------------------------


def summarise(predictions: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """Every table this module produces, for the report and the tests."""
    preds = predictions if predictions is not None else load_all_predictions()
    return {
        "metrics_test": subgroup_metrics("test", preds),
        "metrics_val": subgroup_metrics("val", preds),
        "metrics_val_test": subgroup_metrics("val_test", preds),
        "win_test": hybrid_win_table("test", preds),
        "win_val": hybrid_win_table("val", preds),
        "win_val_test": hybrid_win_table("val_test", preds),
        "bootstrap_test": bootstrap_delta("test", predictions=preds),
        "criteria": evaluate_subgroup_criteria(preds),
        "rolling_mae": rolling_mae(predictions=preds),
        "month_breakdown": month_breakdown("val_test", preds),
    }


def main() -> None:
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 25)

    print("=" * 92)
    print("SUBGROUP AND PERIOD BREAKDOWN (5.11) -- diagnostic re-analysis, NO model trained")
    print("=" * 92)

    tables = summarise()

    metrics_all = pd.concat(
        [tables["metrics_test"], tables["metrics_val"], tables["metrics_val_test"]],
        ignore_index=True,
    )
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    metrics_all.to_parquet(SUBGROUP_BREAKDOWN_FILE, index=False)
    tables["rolling_mae"].to_parquet(SUBGROUP_ROLLING_MAE_FILE, index=False)
    print(f"\nsaved: {SUBGROUP_BREAKDOWN_FILE}")
    print(f"saved: {SUBGROUP_ROLLING_MAE_FILE}")

    for scope in SCOPES:
        print(f"\n--- hybrid win table — scope {scope} ---")
        print(
            hybrid_win_table(scope)
            .drop(columns=["delta_pct"])
            .to_string(index=False)
        )

    print("\n--- bootstrap delta MAE (moving block 7, 2000 resamples) -- verdict scope test ---")
    print(tables["bootstrap_test"].to_string(index=False))

    print("\n--- pre-registered verdicts ---")
    print(tables["criteria"].to_string(index=False))

    criteria = tables["criteria"]
    inferential_verdicts = {VERDICT_WIN, VERDICT_RANK_ONLY, VERDICT_NO_WIN}
    n_infer = int(criteria["verdict"].isin(inferential_verdicts).sum())
    n_win = int((criteria["verdict"] == VERDICT_WIN).sum())
    n_rank_only = int((criteria["verdict"] == VERDICT_RANK_ONLY).sum())
    print(
        f"\nOf the {n_infer} subgroups with adequate n, the hybrid wins with evidence in "
        f"{n_win}; {n_rank_only} rank first without evidence."
    )


if __name__ == "__main__":
    main()
