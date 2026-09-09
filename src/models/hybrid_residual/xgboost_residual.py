"""Stage 2: XGBoost trained on LSTM out-of-fold residuals.

Target is `residual = y_true - y_pred_oof`, features are the full feat_ matrix (calendar,
Fourier, lagged weather, lagged operator/total), all already guarded against same-day
leakage in Phases 2-3.

The premise this model tests: stage 1 is univariate and cannot see the calendar, so its
errors should be systematically larger and structured around holidays and bridge days --
which the Phase 4 step-0 diagnostic confirmed for the linear baseline (bridge days carried
2.70x the error of ordinary working days). If that structure is learnable from exogenous
features, stage 2 recovers it and the hybrid beats stage 1 by a wide margin.

Hyperparameters are selected by the shared procedure in tuning.py -- identical grid and
identical chronological-holdout methodology to the XGBoost-alone baseline, so the
comparison between them measures architecture rather than tuning effort.
"""

from pathlib import Path

import joblib
import pandas as pd

from src.models.hybrid_residual.assemble_residual_dataset import (
    RESIDUAL_TRAINING_FILE,
    assemble,
)
from src.models.hybrid_residual.tuning import (
    TuningResult,
    feature_importance,
    fit_final,
    grid_search,
)
from src.utils.paths import MODELS_DIR
from src.utils.seed import GLOBAL_SEED

RESIDUAL_MODEL_FILE: Path = MODELS_DIR / "xgboost_residual.joblib"
TARGET = "residual"


def load_residual_training_set(path: Path | str = RESIDUAL_TRAINING_FILE) -> pd.DataFrame:
    """Load the assembled residual set, rebuilding it if absent."""
    path = Path(path)
    if not path.exists():
        return assemble()
    return pd.read_parquet(path).sort_values("date").reset_index(drop=True)


def train(
    df: pd.DataFrame | None = None, seed: int = GLOBAL_SEED, verbose: bool = True
) -> tuple[object, TuningResult, list[str]]:
    """Tune on the internal holdout, then refit on the full residual training set."""
    df = df if df is not None else load_residual_training_set()
    feat_cols = [c for c in df.columns if c.startswith("feat_")]

    X, y = df[feat_cols], df[TARGET]

    if verbose:
        print("=" * 78)
        print("STAGE 2 — XGBoost on LSTM OOF residuals")
        print("=" * 78)
        print(f"  training rows        : {len(df)}")
        print(f"  features             : {len(feat_cols)}")
        print(
            f"  date coverage        : {df['date'].min():%Y-%m-%d} -> "
            f"{df['date'].max():%Y-%m-%d}"
        )

    result = grid_search(X, y, df["date"], seed=seed, verbose=verbose)
    # Refit on ALL residual rows: the holdout existed only to choose hyperparameters.
    model = fit_final(X, y, result.params, seed)
    return model, result, feat_cols


def main() -> None:
    model, result, feat_cols = train()

    print("\n  top features by gain:")
    print(feature_importance(model, feat_cols).to_string(index=False))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": feat_cols, "params": result.params}, RESIDUAL_MODEL_FILE)
    print(f"\nsaved: {RESIDUAL_MODEL_FILE}")


if __name__ == "__main__":
    main()
