"""XGBoost applied directly to `total` -- the control, NOT part of the hybrid.

This model exists solely to satisfy CLAUDE.md Phase Acceptance Criterion (b): demonstrating
whether the residual-correction DESIGN adds anything over simply pointing a gradient-boosted
tree at the forecasting problem with the same features. Without it, a hybrid that beats
SARIMAX proves only that the features are good, not that the two-stage decomposition is.

CRITICAL DIFFERENCE FROM THE RESIDUAL MODEL, AND AN EASY MISTAKE TO MAKE:
this model trains on the FULL official Phase 3 training split (889 post-warm-up rows,
2023-01-29 -> 2025-07-05), NOT the 552-row residual subset. It has no dependency on the
LSTM's OOF folds, so the exclusions that shape the residual set -- the 200-row initial
history and the degenerate fold 1 -- simply do not apply to it. Training it on 552 rows
"to be fair" would be the opposite of fair: it would handicap the control by starving it of
a third of its data for a reason that has nothing to do with it. What must match between
the two models is the FEATURE MATRIX and the TUNING PROCEDURE, and both do.
`tests/test_hybrid_residual.py` pins the 889-row expectation explicitly.
"""

from pathlib import Path

import joblib
import pandas as pd

from src.features.build_features import FEATURES_DAILY_FILE, feature_columns
from src.models.hybrid_residual.tuning import (
    TuningResult,
    feature_importance,
    fit_final,
    grid_search,
)
from src.utils.paths import MODELS_DIR
from src.utils.seed import GLOBAL_SEED
from src.utils.splits import apply_warmup_policy, chronological_split, split_masks

ALONE_MODEL_FILE: Path = MODELS_DIR / "xgboost_alone.joblib"
TARGET = "total"


def load_training_split(features_path: Path | str = FEATURES_DAILY_FILE) -> pd.DataFrame:
    """The official Phase 3 training split, post warm-up. 889 rows."""
    df = pd.read_parquet(features_path).sort_values("date").reset_index(drop=True)
    bounds = chronological_split(df["date"])
    trimmed, _ = apply_warmup_policy(df, bounds)
    trimmed = trimmed.reset_index(drop=True)
    return trimmed[split_masks(trimmed, bounds)["train"]].reset_index(drop=True)


def train(
    df: pd.DataFrame | None = None,
    seed: int = GLOBAL_SEED,
    verbose: bool = True,
    feature_subset: list[str] | None = None,
) -> tuple[object, TuningResult, list[str]]:
    """Tune on the internal chronological holdout, then refit on the full train split.

    Args:
        feature_subset: If given, restrict training to exactly these columns instead of the
            full `feature_columns(df)` set. Used by the Phase 8 feature-block ablation
            (`src/evaluation/feature_block_ablation.py`), a DIAGNOSTIC control that writes
            nothing under `models/`. `None` (the default) is byte-identical to the original
            behaviour and is pinned by
            `test_feature_subset_none_reproduces_the_phase5_feature_set`.
    """
    df = df if df is not None else load_training_split()
    feat_cols = feature_columns(df) if feature_subset is None else list(feature_subset)

    X, y = df[feat_cols], df[TARGET]

    if verbose:
        print("=" * 78)
        print("CONTROL — XGBoost trained directly on `total` (not on residuals)")
        print("=" * 78)
        print(f"  training rows        : {len(df)}  (full Phase 3 train split)")
        print(f"  features             : {len(feat_cols)}")
        print(
            f"  date coverage        : {df['date'].min():%Y-%m-%d} -> "
            f"{df['date'].max():%Y-%m-%d}"
        )

    result = grid_search(X, y, df["date"], seed=seed, verbose=verbose)
    model = fit_final(X, y, result.params, seed)
    return model, result, feat_cols


def predict(model, features_df: pd.DataFrame, feat_cols: list[str]) -> pd.Series:
    """Predict `total` for the given rows."""
    return pd.Series(model.predict(features_df[feat_cols]), index=features_df.index)


def main() -> None:
    model, result, feat_cols = train()

    print("\n  top features by gain:")
    print(feature_importance(model, feat_cols).to_string(index=False))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": feat_cols, "params": result.params}, ALONE_MODEL_FILE)
    print(f"\nsaved: {ALONE_MODEL_FILE}")


if __name__ == "__main__":
    main()
