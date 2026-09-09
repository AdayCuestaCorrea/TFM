"""Feature-block ablation of `xgboost_alone` -- diagnostic control, answers Q2.

DIAGNOSTIC CONTROL, NOT A CANDIDATE MODEL. Every model trained here is a variant of the
`xgboost_alone` control retrained on a restricted feature subset; none enters chapter 5's
master comparison, extends `dashboard_data.MODEL_NAMES`, writes under `models/`, or competes
for best model. A good score is a diagnostic finding to discuss, never a new headline.
`test_phase8_adds_no_model_to_the_master_comparison` pins this mechanically.

QUESTION (Q2). Does direct XGBoost already integrate temporal *and* exogenous effects
simultaneously through the constructed features? If `solo_temporal` and `solo_exogeno` are
both materially worse than `completo`, the single-stage model demonstrably uses both
information types internally -- corroborating (not proving; Phase 5 already proved the
predictive half, objection O3) that the two-stage residual split adds complexity without
adding predictive information.

NOT A DECOMPOSITION (objection O2). The blocks are NOT information-disjoint:
`fourier_annual` and lagged weather both encode the annual cycle (Phase 7 watched the
weather correlations collapse once the annual terms were partialled out), and `lag_total`
at lags 7/14/21/28 encodes the weekly calendar implicitly. So `solo_temporal` still "knows"
the weekday and `solo_exogeno` still "knows" the season. This is a block-removal study; the
delta percentages must never be presented as summing to anything.

Same training split (889 post-warm-up rows), same 54-point grid and internal chronological
holdout, same evaluation protocol as Phase 5. The only thing that changes is which blocks
are visible. Re-tuning each subset (rather than freezing Phase 5's hyperparameters) is
deliberate -- it gives each subset its best shot (objection O1); the 5% materiality
threshold absorbs grid/seed noise, and the selected hyperparameters are reported per subset.
"""

from pathlib import Path

import joblib
import pandas as pd

from src.evaluation.dashboard_data import FEATURE_GROUPS, classify_feature
from src.evaluation.metrics import all_metrics
from src.features.build_features import FEATURES_DAILY_FILE, feature_columns
from src.models.hybrid_residual import xgboost_alone
from src.models.hybrid_residual.xgboost_alone import ALONE_MODEL_FILE, load_training_split
from src.utils.paths import PROCESSED_DIR
from src.utils.seed import GLOBAL_SEED
from src.utils.splits import apply_warmup_policy, chronological_split, split_masks

ABLATION_FILE: Path = PROCESSED_DIR / "feature_block_ablation.parquet"
FULL_COMPARISON_FILE: Path = PROCESSED_DIR / "full_comparison.parquet"

# A block contributes materially if withholding it raises val MAE by >= this percent
# relative to `completo`. 5% is the materiality floor because each subset re-runs its own
# grid search, so smaller differences are inside grid/seed noise (objection O1).
BLOCK_MATERIAL_PCT: float = 5.0

TARGET = "total"

# Pre-registered subsets, fixed before any result. Keys map to block names from
# `dashboard_data.FEATURE_GROUP_PATTERNS` -- no new taxonomy is invented.
_TEMPORAL = ("lag_total", "lag_operator", "rolling")
_EXOGENO = ("calendar", "fourier_weekly", "fourier_annual", "weather")
_CALENDARIO = ("calendar", "fourier_weekly", "fourier_annual")

SUBSETS: dict[str, tuple[str, ...]] = {
    "completo": tuple(FEATURE_GROUPS),
    "solo_temporal": _TEMPORAL,
    "solo_exogeno": _EXOGENO,
    "sin_meteo": tuple(b for b in FEATURE_GROUPS if b != "weather"),
    "solo_calendario": _CALENDARIO,
}

# Reported, not pass/fail (Phase 5b Experiment B precedent).
REPORTED_ONLY = {"sin_meteo", "solo_calendario"}

VERDICT_MATERIAL = "aporta (>= umbral)"
VERDICT_IMMATERIAL = "no aporta (< umbral)"
VERDICT_REPORTED = "solo informativo"


def subset_columns(df: pd.DataFrame, key: str) -> list[str]:
    """The `feat_`-prefixed columns of `df` whose block is in `SUBSETS[key]`."""
    if key not in SUBSETS:
        raise ValueError(f"subset_columns: unknown subset {key!r}; expected {list(SUBSETS)}")
    wanted = set(SUBSETS[key])
    return [c for c in feature_columns(df) if classify_feature(c) in wanted]


def _eval_frame(features_path: Path | str = FEATURES_DAILY_FILE) -> tuple[pd.DataFrame, dict]:
    """The full post-warm-up frame plus split masks, for scoring val/test."""
    df = pd.read_parquet(features_path).sort_values("date").reset_index(drop=True)
    bounds = chronological_split(df["date"])
    trimmed, _ = apply_warmup_policy(df, bounds)
    trimmed = trimmed.reset_index(drop=True)
    return trimmed, split_masks(trimmed, bounds)


def run_ablation(
    subsets: dict[str, tuple[str, ...]] | None = None,
    seed: int = GLOBAL_SEED,
    verbose: bool = True,
) -> pd.DataFrame:
    """Retrain `xgboost_alone` on each restricted subset; score val + test.

    Returns one row per (subset, split): [subset, n_features, blocks, max_depth,
    n_estimators, learning_rate, min_child_weight, holdout_mae, split, MAE, RMSE, MAPE, R2].
    `delta_val_MAE_pct` is added later by `evaluate_block_criteria`. The `completo` row is
    NOT retrained here -- use `completo_reference()`.
    """
    subsets = subsets or {k: v for k, v in SUBSETS.items() if k != "completo"}
    train_df = load_training_split()
    eval_df, masks = _eval_frame()

    rows = []
    for key, blocks in subsets.items():
        cols = subset_columns(train_df, key)
        if verbose:
            print(f"\n[{key}] {len(cols)} features, blocks={list(blocks)}")
        model, result, feat_cols = xgboost_alone.train(
            df=train_df, seed=seed, verbose=verbose, feature_subset=cols
        )
        for split in ("val", "test"):
            part = eval_df[masks[split]]
            preds = xgboost_alone.predict(model, part, feat_cols)
            m = all_metrics(part[TARGET], preds)
            rows.append(
                {
                    "subset": key,
                    "n_features": len(cols),
                    "blocks": ",".join(blocks),
                    "max_depth": result.params["max_depth"],
                    "n_estimators": result.params["n_estimators"],
                    "learning_rate": result.params["learning_rate"],
                    "min_child_weight": result.params["min_child_weight"],
                    "holdout_mae": result.holdout_mae,
                    "split": split,
                    "MAE": m["MAE"],
                    "RMSE": m["RMSE"],
                    "MAPE": m["MAPE"],
                    "R2": m["R2"],
                }
            )
    return pd.DataFrame(rows)


def completo_reference() -> pd.DataFrame:
    """The `completo` row -- read from the published Phase 5 artifacts, NOT retrained."""
    full = pd.read_parquet(FULL_COMPARISON_FILE)
    alone = full[full["Model"] == "xgboost_alone"]
    params = joblib.load(ALONE_MODEL_FILE)["params"]
    train_df = load_training_split()
    n_feat = len(feature_columns(train_df))

    rows = []
    for split in ("val", "test"):
        r = alone[alone["split"] == split].iloc[0]
        rows.append(
            {
                "subset": "completo",
                "n_features": n_feat,
                "blocks": ",".join(FEATURE_GROUPS),
                "max_depth": params["max_depth"],
                "n_estimators": params["n_estimators"],
                "learning_rate": params["learning_rate"],
                "min_child_weight": params["min_child_weight"],
                "holdout_mae": float("nan"),
                "split": split,
                "MAE": float(r["MAE"]),
                "RMSE": float(r["RMSE"]),
                "MAPE": float(r["MAPE"]),
                "R2": float(r["R2"]),
            }
        )
    return pd.DataFrame(rows)


def evaluate_block_criteria(ablation: pd.DataFrame) -> pd.DataFrame:
    """Apply BLOCK_MATERIAL_PCT mechanically. Verdict read on val; test is stability only.

    Returns [subset, n_features, val_MAE, delta_val_MAE_pct, test_MAE, verdict].
    """
    completo_val = float(
        ablation.loc[(ablation["subset"] == "completo") & (ablation["split"] == "val"), "MAE"].iloc[0]
    )
    out = []
    for key in ablation["subset"].unique():
        val_mae = float(
            ablation.loc[(ablation["subset"] == key) & (ablation["split"] == "val"), "MAE"].iloc[0]
        )
        test_mae = float(
            ablation.loc[(ablation["subset"] == key) & (ablation["split"] == "test"), "MAE"].iloc[0]
        )
        delta_pct = 100.0 * (val_mae - completo_val) / completo_val
        if key == "completo":
            verdict = "referencia"
        elif key in REPORTED_ONLY:
            verdict = VERDICT_REPORTED
        elif delta_pct >= BLOCK_MATERIAL_PCT:
            verdict = VERDICT_MATERIAL
        else:
            verdict = VERDICT_IMMATERIAL
        out.append(
            {
                "subset": key,
                "n_features": int(
                    ablation.loc[ablation["subset"] == key, "n_features"].iloc[0]
                ),
                "val_MAE": val_mae,
                "delta_val_MAE_pct": delta_pct,
                "test_MAE": test_mae,
                "verdict": verdict,
            }
        )
    return pd.DataFrame(out)


def _assemble() -> pd.DataFrame:
    ablation = pd.concat([completo_reference(), run_ablation()], ignore_index=True)
    criteria = evaluate_block_criteria(ablation).set_index("subset")["delta_val_MAE_pct"]
    ablation["delta_val_MAE_pct"] = ablation["subset"].map(criteria)
    return ablation


def summarise() -> dict[str, pd.DataFrame]:
    """Every table this module produces, for the report and the tests."""
    ablation = _assemble()
    return {"ablation": ablation, "criteria": evaluate_block_criteria(ablation)}


def main() -> None:
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)

    print("=" * 88)
    print("FEATURE-BLOCK ABLATION of xgboost_alone -- diagnostic control, NOT a candidate")
    print("=" * 88)

    ablation = _assemble()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ablation.to_parquet(ABLATION_FILE, index=False)
    print(f"\nsaved: {ABLATION_FILE}")

    print("\n--- per (subset, split) ---")
    print(ablation.to_string(index=False))
    print("\n--- block criteria (verdict on val; threshold "
          f"{BLOCK_MATERIAL_PCT}% -- NOT a decomposition, O2) ---")
    print(evaluate_block_criteria(ablation).to_string(index=False))


if __name__ == "__main__":
    main()
