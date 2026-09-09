"""Experiment A: Stage 2 residual model with upweighted irregular-calendar rows.

MOTIVATION. Phase 5 found the hybrid loses to SARIMAX and XGBoost-alone overall, but that
its one real achievement was on festivo days, where stage 2 cut stage 1's error from
1,585,388 to 666,572 (-58%). Those rows are also rare -- 27 festivo/bridge rows out of 552
in the residual training set -- so a squared-error objective spends almost all its capacity
on ordinary days. This experiment asks a single narrow question: does telling the model to
care more about the irregular days improve them?

WHAT CHANGES: exactly one thing, `sample_weight`.
WHAT DOES NOT CHANGE: the 552-row training set (same assembly, same fold exclusions), the
161-column feature matrix, the random seed, and the hyperparameters.

WEIGHT = 3.0 IS A PRIORI, NOT SEARCHED. It was fixed before running anything as a moderate
upweighting -- roughly compensating for irregular days being ~5% of rows without swamping
the objective. It was NOT chosen by trying several values and keeping the best; doing that
would convert a sensitivity analysis into a search for a flattering configuration, and the
resulting number would not be honest evidence about anything.

HYPERPARAMETERS ARE LOADED FROM THE PHASE 5 ARTIFACT, not re-tuned and not re-declared.
Re-running the grid search with weights active would change two things at once, and any
difference in the result could not be attributed to either. Loading the saved params
guarantees identity rather than merely intending it; `PHASE5_PARAMS` below is asserted
against the artifact so a silent drift in either fails loudly.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.models.hybrid_residual.tuning import build_model
from src.models.hybrid_residual.xgboost_residual import (
    RESIDUAL_MODEL_FILE,
    load_residual_training_set,
)
from src.utils.paths import MODELS_DIR, PROCESSED_DIR
from src.utils.seed import GLOBAL_SEED

WEIGHTED_MODEL_FILE: Path = MODELS_DIR / "xgboost_residual_weighted.joblib"
WEIGHTED_PREDICTIONS_FILE: Path = PROCESSED_DIR / "hybrid_weighted_predictions.parquet"

TARGET = "residual"

# The Phase 5 selection, restated for readability and asserted against the saved artifact.
PHASE5_PARAMS = {
    "max_depth": 5,
    "n_estimators": 300,
    "learning_rate": 0.01,
    "min_child_weight": 3,
}

# Rows treated as irregular-calendar days.
UPWEIGHT_COLS = [
    "feat_day_type_festivo",
    "feat_day_type_domingo_festivo",
    "feat_is_bridge_day",
]
UPWEIGHT_VALUE = 3.0
BASE_WEIGHT = 1.0


def load_phase5_params(path: Path | str = RESIDUAL_MODEL_FILE) -> dict:
    """Read the Phase 5 hyperparameters from the saved model, and verify them.

    Raises:
        ValueError: if the artifact is missing or its params differ from PHASE5_PARAMS,
            which would mean this experiment is no longer a controlled comparison.
    """
    path = Path(path)
    if not path.exists():
        raise ValueError(
            f"missing {path} -- run Phase 5 (xgboost_residual) before this experiment"
        )
    params = joblib.load(path)["params"]
    if params != PHASE5_PARAMS:
        raise ValueError(
            "Phase 5 hyperparameters have changed since this experiment was written: "
            f"artifact={params}, expected={PHASE5_PARAMS}. Experiment A must differ from "
            "Phase 5 in sample_weight ALONE; re-verify before proceeding."
        )
    return params


def build_sample_weights(df: pd.DataFrame) -> np.ndarray:
    """UPWEIGHT_VALUE on irregular-calendar rows, BASE_WEIGHT elsewhere."""
    missing = [c for c in UPWEIGHT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"weighted residual: missing column(s) {missing}")

    is_irregular = df[UPWEIGHT_COLS].sum(axis=1) > 0
    return np.where(is_irregular, UPWEIGHT_VALUE, BASE_WEIGHT).astype("float64")


def train(
    df: pd.DataFrame | None = None, seed: int = GLOBAL_SEED, verbose: bool = True
) -> tuple[object, list[str], np.ndarray]:
    """Train the weighted residual model on the unchanged Phase 5 residual set."""
    df = df if df is not None else load_residual_training_set()
    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    params = load_phase5_params()

    weights = build_sample_weights(df)
    X, y = df[feat_cols], df[TARGET]

    if verbose:
        n_up = int((weights == UPWEIGHT_VALUE).sum())
        print("=" * 78)
        print("EXPERIMENT A — Stage 2 residual model, irregular-calendar rows upweighted")
        print("=" * 78)
        print(f"  training rows     : {len(df)}  (unchanged from Phase 5)")
        print(f"  features          : {len(feat_cols)}  (unchanged)")
        print(f"  hyperparameters   : {params}  (loaded from Phase 5 artifact)")
        print(
            f"  upweighted rows   : {n_up} of {len(df)} "
            f"({n_up / len(df):.1%}) at weight {UPWEIGHT_VALUE}"
        )
        print(f"  weight choice     : a priori, NOT searched over multiple values")

    # build_model rather than fit_final: identical estimator construction, but the fit call
    # needs sample_weight, which is the single intended difference from Phase 5.
    model = build_model(params, seed)
    model.fit(X, y, sample_weight=weights)

    return model, feat_cols, weights


def combine_weighted(
    model, feat_cols: list[str], verbose: bool = True
) -> pd.DataFrame:
    """Apply the weighted residual model on top of the Phase 4 LSTM predictions.

    Uses the identical combination rule as Phase 5's combine.py: y = y_lstm + e_pred, with
    stage-1 values taken from the saved artifact rather than regenerated.
    """
    from src.features.build_features import FEATURES_DAILY_FILE
    from src.models.lstm.final_model import VAL_TEST_PREDICTIONS_FILE
    from src.utils.splits import chronological_split, split_masks

    lstm = pd.read_parquet(VAL_TEST_PREDICTIONS_FILE)
    feats = pd.read_parquet(FEATURES_DAILY_FILE)
    merged = lstm.merge(feats[["date", *feat_cols]], on="date", how="left")

    if merged[feat_cols].isna().any(axis=1).any():
        raise ValueError("combine_weighted: val/test dates missing feature rows")

    e_pred = np.asarray(model.predict(merged[feat_cols]), dtype="float64")
    y_lstm = merged["y_pred_lstm"].to_numpy(dtype="float64")

    out = pd.DataFrame(
        {
            "date": merged["date"],
            "y_true": merged["y_true"].to_numpy(dtype="float64"),
            "y_pred_lstm": y_lstm,
            "e_pred_xgb_weighted": e_pred,
            "y_pred_hybrid_weighted": y_lstm + e_pred,
        }
    )

    bounds = chronological_split(feats["date"].sort_values())
    masks = split_masks(out, bounds)
    out["split"] = np.select(
        [masks["train"], masks["val"], masks["test"]], ["train", "val", "test"], "unassigned"
    )
    return out.sort_values("date").reset_index(drop=True)


def main() -> None:
    from src.evaluation.metrics import all_metrics

    model, feat_cols, _ = train()
    out = combine_weighted(model, feat_cols)

    print("\n" + "-" * 78)
    print(f"{'split':<7} {'MAE':>12} {'RMSE':>12} {'MAPE':>8} {'R2':>9}")
    print("-" * 78)
    for split in ("val", "test"):
        subset = out[out["split"] == split]
        m = all_metrics(subset["y_true"], subset["y_pred_hybrid_weighted"])
        print(
            f"{split:<7} {m['MAE']:>12,.0f} {m['RMSE']:>12,.0f} "
            f"{m['MAPE']:>7.2f}% {m['R2']:>9.4f}"
        )
    print("-" * 78)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "features": feat_cols, "params": PHASE5_PARAMS,
         "upweight_value": UPWEIGHT_VALUE, "upweight_cols": UPWEIGHT_COLS},
        WEIGHTED_MODEL_FILE,
    )
    out.to_parquet(WEIGHTED_PREDICTIONS_FILE, index=False)
    print(f"\nsaved: {WEIGHTED_MODEL_FILE}")
    print(f"saved: {WEIGHTED_PREDICTIONS_FILE}")


if __name__ == "__main__":
    main()
