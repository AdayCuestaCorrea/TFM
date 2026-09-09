"""Informational-parity LSTM -- diagnostic control, answers Q1.

DIAGNOSTIC CONTROL, NOT A CANDIDATE MODEL. Nothing this module produces enters chapter 5's
master comparison, extends `dashboard_data.MODEL_NAMES`, or writes under `models/`. Its fits
are in-memory and discarded, exactly as `lstm_learning_curve.py`. Only the metrics parquet
is persisted. `test_phase8_adds_no_model_to_the_master_comparison` pins this mechanically.

QUESTION (Q1). Stage 1 is univariate while XGBoost sees 161 features from the start. Is
Stage 1 losing because it is an LSTM, or because it is univariate? This control gives the
LSTM branch informational parity: the target window feeds the recurrent branch and row t's
own `feat_`-prefixed exogenous vector is concatenated afterwards (the minimal change -- the
window spans [t-W, t) and excludes row t, so a plain multivariate (W,k) window would still
hide the calendar of the day being predicted).

ASYMMETRIC INTERPRETABILITY (objection O4). The parity LSTM sees a 28-step raw window PLUS
row t's features -- strictly MORE information than XGBoost's flat vector of 7 selected lags
and 4 rolling statistics. Consequently a LOSS is a strong conclusion (informational
asymmetry decisively ruled out) while a WIN is a weak one (partly extra information, not
only architecture). Stated before results so it is clear which outcomes are quotable.

A GOOD RESULT HERE IS AN ARGUMENT *AGAINST* THE HYBRID, NOT FOR A BETTER ONE (objection O8).
If the parity LSTM closes the gap, rebuilding the hybrid on top of it is forbidden: a
Stage 1 that already sees the exogenous features leaves Stage 2 nothing to correct that
Stage 1 could not see, which destroys the two-stage rationale.

SEED COUNT -- 5 seeds [42, 1, 2, 3, 4], decided by the pre-registered rule, NOT by results.
Section 2.3 / objection O6 fix the rule: "raise to 5 only if the timing probe shows
headroom, decided before the run". The timing probe showed ample headroom (slowest single
fit 9.7 s against a ~6 min ceiling), so the count was raised to 5. The probe also surfaced
provisional seed-42 val MAEs (calendario 395,274 / completo 369,835), both inside the
25-60% ambiguous band, and those numbers WERE visible before the seed count was chosen.
They did not and could not drive the choice: the rule is headroom-based and governs
regardless of where the provisional numbers fell. Stated plainly here so a reader need not
take it on trust. The univariate baseline is recomputed as the 5-seed median at
fraction 1.00 of the Phase 7 learning curve, so the comparison stays like-for-like.

VAL ONLY -- the test split is never read (consistent with `window_comparison.py` and
`lstm_learning_curve.py`). The reference point (`xgboost_alone` val MAE 244,843) lives on
val; the question is comparative-explanatory; and a test MAE for a new architecture would
function as a headline number. The asymmetry with the ablation (which does report a test
stability column) is deliberate: the ablation subsets are variants of a model already in
the master table with published test numbers; the parity LSTM is a new architecture where a
test number would create a candidate.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.dashboard_data import classify_feature
from src.evaluation.lstm_learning_curve import (
    DEGENERATE_CORR_MAX,
    DEGENERATE_STD_RATIO_MAX,
    LEARNING_CURVE_FILE,
    _val_degeneracy,
)
from src.features.build_features import FEATURES_DAILY_FILE, feature_columns
from src.models.lstm.final_model import train_final_model
from src.models.lstm.model import LSTMConfig
from src.utils.paths import PROCESSED_DIR
from src.utils.seed import GLOBAL_SEED

PARITY_FILE: Path = PROCESSED_DIR / "lstm_parity_control.parquet"

# Feature scopes, pre-registered before any result.
_CALENDARIO_BLOCKS = ("calendar", "fourier_weekly", "fourier_annual")
SCOPES: dict[str, tuple[str, ...]] = {
    "calendario": _CALENDARIO_BLOCKS,   # where Phase 7 located the residual structure (k=17)
    "completo": ("__all_feat__",),      # literal parity with xgboost_alone (k=161)
}

# 42 stays primary (GLOBAL_SEED); 1, 2, 3, 4 are the replicate seeds. Raised from 3 to 5
# by the pre-registered §2.3/O6 rule after the timing probe showed headroom (see the module
# docstring). Reported as median with a min-max band.
SEEDS: list[int] = [GLOBAL_SEED, 1, 2, 3, 4]

# --------------------------------------------------------------------------------------
# Reference MAEs -- published, not refitted.
#
# UNIVARIATE_VAL_MAE is the fraction-1.00 FIVE-SEED MEDIAN from the Phase 7 learning curve
# (seeds 42/1/2/3/4 -> 411,142 / 416,324 / 431,415 / 433,829 / 379,074; median 416,324),
# NOT the single best seed (411,142) quoted in CLAUDE.md Phase 4 and section 5.9. The parity
# verdict is read on a five-seed median, so the baseline must be a five-seed median too --
# same harness, same seeds, same val split, genuinely like-for-like. Comparing against a
# single-seed best would bias the gap-closure downward, toward the thesis's current
# conclusion, which is exactly where the analysis must be strictest. The five-seed median
# happens to equal the earlier three-seed median (seed 1 is the central value either way);
# _GAP and the thresholds below are unchanged. Section 5.10.2 reports BOTH figures.
# --------------------------------------------------------------------------------------
UNIVARIATE_VAL_MAE = 416_324               # Phase 7 learning curve, fraction 1.00 5-seed median
UNIVARIATE_VAL_MAE_BEST_SEED = 411_142     # Phase 4 / section 5.9, single best seed
XGBOOST_ALONE_VAL_MAE = 244_843            # Phase 5, published

PARITY_GAP_CLOSED_HIGH = 60.0   # median val MAE <= 313,435  -> asimetria informativa
PARITY_GAP_CLOSED_LOW = 25.0    # median val MAE >= 373,454  -> limitacion arquitectonica
UNDERTRAINED_EPOCH_MAX = 20     # best_epoch below this is flagged undertrained (O7)

VERDICT_CAPACITY = "fallo de capacidad/muestra"
VERDICT_ASYMMETRY = "asimetria informativa"
VERDICT_ARCHITECTURE = "limitacion arquitectonica"
VERDICT_AMBIGUOUS = "mixto/ambiguo"


def _assert_baseline_matches_artifact() -> None:
    """Fail loudly if the learning-curve artifact no longer yields the pinned median."""
    if not LEARNING_CURVE_FILE.exists():  # pragma: no cover - artifact present in-repo
        return
    curve = pd.read_parquet(LEARNING_CURVE_FILE)
    frac1 = curve[np.isclose(curve["fraction"], 1.0)]
    # Expect the 5-seed set [42, 1, 2, 3, 4]; the median must match the pinned constant.
    median = float(frac1["val_MAE"].median())
    if round(median) != UNIVARIATE_VAL_MAE:
        raise AssertionError(
            "lstm_parity_control.UNIVARIATE_VAL_MAE is out of sync with "
            f"{LEARNING_CURVE_FILE.name}: constant={UNIVARIATE_VAL_MAE}, "
            f"artifact fraction-1.00 median={median:.0f}. Regenerate the learning curve or "
            "update the constant deliberately."
        )


# NB: `_assert_baseline_matches_artifact()` is deliberately NOT called at module scope
# (CLAUDE.md forbids top-level I/O in `src/`). It runs from `run_parity_control()` and
# `main()`, so artifact drift still fails loudly at every real entry point.

_GAP = UNIVARIATE_VAL_MAE - XGBOOST_ALONE_VAL_MAE


def scope_columns(df: pd.DataFrame, scope: str) -> list[str]:
    """The `feat_`-prefixed columns for a scope."""
    if scope not in SCOPES:
        raise ValueError(f"scope_columns: unknown scope {scope!r}; expected {list(SCOPES)}")
    feat_cols = feature_columns(df)
    if scope == "completo":
        return feat_cols
    wanted = set(SCOPES[scope])
    return [c for c in feat_cols if classify_feature(c) in wanted]


def _gap_closed_pct(val_mae: float) -> float:
    return 100.0 * (UNIVARIATE_VAL_MAE - val_mae) / _GAP


def run_parity_control(
    scopes: dict[str, tuple[str, ...]] | None = None,
    seeds: list[int] | None = None,
    verbose: bool = True,
    config: LSTMConfig | None = None,
) -> pd.DataFrame:
    """Fit one parity LSTM per (scope, seed); score on val only.

    Returns one row per (scope, seed): [scope, n_exog, seed, train_rows, train_sequences,
    epochs_run, best_epoch, val_MAE, val_RMSE, val_MAPE, val_R2, pred_std_ratio,
    pred_actual_corr, degenerate, undertrained, gap_closed_pct].
    """
    _assert_baseline_matches_artifact()
    scopes = scopes or SCOPES
    seeds = seeds or SEEDS
    df = pd.read_parquet(FEATURES_DAILY_FILE)

    rows = []
    for scope in scopes:
        cols = scope_columns(df, scope)
        for seed in seeds:
            if verbose:
                print(f"  scope {scope} (k={len(cols)})  seed {seed} ...", flush=True)
            predictions, result, _, _ = train_final_model(
                seed=seed, verbose=False, exog_columns=cols, config=config
            )
            std_ratio, corr, degenerate = _val_degeneracy(predictions)
            val = result.metrics["val"]
            undertrained = bool(result.best_epoch < UNDERTRAINED_EPOCH_MAX)
            rows.append(
                {
                    "scope": scope,
                    "n_exog": len(cols),
                    "seed": seed,
                    "train_rows": result.n_train_rows,
                    "train_sequences": result.n_train_sequences,
                    "epochs_run": result.epochs_run,
                    "best_epoch": result.best_epoch,
                    "val_MAE": val["MAE"],
                    "val_RMSE": val["RMSE"],
                    "val_MAPE": val["MAPE"],
                    "val_R2": val["R2"],
                    "pred_std_ratio": std_ratio,
                    "pred_actual_corr": corr,
                    "degenerate": degenerate,
                    "undertrained": undertrained,
                    "gap_closed_pct": _gap_closed_pct(val["MAE"]),
                }
            )
            if verbose:
                flag = "  [DEGENERATE]" if degenerate else ""
                print(
                    f"    val MAE {val['MAE']:,.0f}  best_epoch {result.best_epoch}  "
                    f"gap closed {_gap_closed_pct(val['MAE']):.1f}%{flag}",
                    flush=True,
                )
    return pd.DataFrame(rows)


def summarise_by_scope(parity: pd.DataFrame) -> pd.DataFrame:
    """Median and min-max val MAE per scope, plus degenerate/undertrained counts."""
    grouped = parity.groupby("scope")
    return pd.DataFrame(
        {
            "scope": list(grouped.groups),
            "n_exog": grouped["n_exog"].first().to_numpy(),
            "val_MAE_median": grouped["val_MAE"].median().to_numpy(),
            "val_MAE_min": grouped["val_MAE"].min().to_numpy(),
            "val_MAE_max": grouped["val_MAE"].max().to_numpy(),
            "gap_closed_pct_median": grouped["gap_closed_pct"].median().to_numpy(),
            "best_epoch_median": grouped["best_epoch"].median().to_numpy(),
            "n_degenerate": grouped["degenerate"].sum().to_numpy(),
            "n_undertrained": grouped["undertrained"].sum().to_numpy(),
            "n_seeds": grouped["seed"].size().to_numpy(),
        }
    )


def evaluate_parity_criteria(parity: pd.DataFrame) -> pd.DataFrame:
    """Apply the pre-registered verdict logic mechanically, per scope.

    Order (fixed): degeneracy first, then the min-max band-straddle downgrade (O6), then the
    median gap-closure thresholds PARITY_GAP_CLOSED_HIGH / PARITY_GAP_CLOSED_LOW. All
    outcomes are reported with equal prominence; there is no preferred result.

    Returns [scope, n_seeds, n_degenerate, val_MAE_median, val_MAE_min, val_MAE_max,
    gap_closed_pct_median, verdict].
    """
    summary = summarise_by_scope(parity).set_index("scope")
    high_mae = UNIVARIATE_VAL_MAE - PARITY_GAP_CLOSED_HIGH / 100.0 * _GAP
    low_mae = UNIVARIATE_VAL_MAE - PARITY_GAP_CLOSED_LOW / 100.0 * _GAP

    out = []
    for scope, r in summary.iterrows():
        n_seeds = int(r["n_seeds"])
        n_degen = int(r["n_degenerate"])
        median_mae = float(r["val_MAE_median"])
        band_lo, band_hi = float(r["val_MAE_min"]), float(r["val_MAE_max"])

        if n_degen > n_seeds / 2:
            verdict = VERDICT_CAPACITY
        elif band_lo <= high_mae <= band_hi or band_lo <= low_mae <= band_hi:
            # Min-max band straddles a threshold -> unreadable -> downgrade (O6).
            verdict = VERDICT_AMBIGUOUS
        elif median_mae <= high_mae:
            verdict = VERDICT_ASYMMETRY
        elif median_mae >= low_mae:
            verdict = VERDICT_ARCHITECTURE
        else:
            verdict = VERDICT_AMBIGUOUS

        out.append(
            {
                "scope": scope,
                "n_seeds": n_seeds,
                "n_degenerate": n_degen,
                "val_MAE_median": median_mae,
                "val_MAE_min": band_lo,
                "val_MAE_max": band_hi,
                "gap_closed_pct_median": float(r["gap_closed_pct_median"]),
                "verdict": verdict,
            }
        )
    return pd.DataFrame(out)


def summarise() -> dict[str, pd.DataFrame]:
    parity = run_parity_control()
    return {
        "parity": parity,
        "by_scope": summarise_by_scope(parity),
        "criteria": evaluate_parity_criteria(parity),
    }


def main() -> None:
    _assert_baseline_matches_artifact()
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)

    print("=" * 88)
    print("INFORMATIONAL-PARITY LSTM -- diagnostic control, val-only, no model artifact")
    print("=" * 88)
    print(
        f"baseline (5-seed median univariate val MAE) : {UNIVARIATE_VAL_MAE:,}  "
        f"[best seed {UNIVARIATE_VAL_MAE_BEST_SEED:,}]"
    )
    print(f"reference (xgboost_alone val MAE)            : {XGBOOST_ALONE_VAL_MAE:,}")
    print(f"gap to close                                : {_GAP:,}")

    parity = run_parity_control()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    parity.to_parquet(PARITY_FILE, index=False)
    print(f"\nsaved: {PARITY_FILE}")

    print("\n--- per fit ---")
    print(parity.to_string(index=False))
    print("\n--- median (min-max) val MAE per scope ---")
    print(summarise_by_scope(parity).to_string(index=False))
    print("\n--- verdicts (degeneracy first, then band straddle, then median; O6) ---")
    print(evaluate_parity_criteria(parity).to_string(index=False))


if __name__ == "__main__":
    main()
