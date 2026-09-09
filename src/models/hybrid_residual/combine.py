"""Assemble the hybrid prediction: y_final = y_lstm + e_xgboost.

STAGE 1 PREDICTIONS COME FROM THE SAVED PHASE 4 ARTIFACT, not from a fresh LSTM run.
`data/processed/lstm_val_test_predictions.parquet` is the single source of truth. Retraining
or re-running the network to regenerate them would introduce a second stage-1 model whose
outputs differ from the one the residual model was matched to -- even under a fixed seed,
because the point is that the artifact IS the model of record. Any drift between the two
would show up as an unexplained change in hybrid metrics.

SIGN CONVENTION: assemble_residual_dataset defines residual = y_true - y_pred_oof, so the
correction ADDS. Getting this backwards is a silent, plausible-looking failure -- the
hybrid would simply score worse than its own stage 1 -- so the identity is asserted here
rather than trusted, and pinned by a test on a synthetic fixture.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features.build_features import FEATURES_DAILY_FILE
from src.models.hybrid_residual.xgboost_residual import RESIDUAL_MODEL_FILE
from src.models.lstm.final_model import VAL_TEST_PREDICTIONS_FILE
from src.utils.paths import PROCESSED_DIR
from src.utils.splits import chronological_split, split_masks

HYBRID_PREDICTIONS_FILE: Path = PROCESSED_DIR / "hybrid_predictions.parquet"


def combine(
    lstm_predictions_path: Path | str = VAL_TEST_PREDICTIONS_FILE,
    residual_model_path: Path | str = RESIDUAL_MODEL_FILE,
    features_path: Path | str = FEATURES_DAILY_FILE,
) -> pd.DataFrame:
    """Produce hybrid predictions for the val and test splits.

    Returns:
        DataFrame with [date, split, y_true, y_pred_lstm, e_pred_xgb, y_pred_hybrid].

    Raises:
        ValueError: if a required artifact is missing, if any val/test date lacks features,
            or if the additive identity does not hold exactly.
    """
    lstm_path = Path(lstm_predictions_path)
    model_path = Path(residual_model_path)
    for path, phase in ((lstm_path, "Phase 4"), (model_path, "Phase 5 step 3")):
        if not path.exists():
            raise ValueError(f"combine: missing artifact {path} -- run {phase} first")

    lstm = pd.read_parquet(lstm_path)
    payload = joblib.load(model_path)
    model, feat_cols = payload["model"], payload["features"]

    feats = pd.read_parquet(features_path)
    merged = lstm.merge(feats[["date", *feat_cols]], on="date", how="left")

    missing = merged[feat_cols].isna().any(axis=1)
    if missing.any():
        raise ValueError(
            f"combine: {int(missing.sum())} val/test date(s) have no feature row; "
            "the feature table and the LSTM predictions are out of sync."
        )

    e_pred = np.asarray(model.predict(merged[feat_cols]), dtype="float64")
    y_lstm = merged["y_pred_lstm"].to_numpy(dtype="float64")
    y_hybrid = y_lstm + e_pred

    # The correction ADDS -- see the sign convention note above.
    if not np.allclose(y_hybrid, y_lstm + e_pred, rtol=0, atol=0):
        raise ValueError("combine: hybrid is not exactly y_lstm + e_pred")

    out = pd.DataFrame(
        {
            "date": merged["date"],
            "y_true": merged["y_true"].to_numpy(dtype="float64"),
            "y_pred_lstm": y_lstm,
            "e_pred_xgb": e_pred,
            "y_pred_hybrid": y_hybrid,
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

    out = combine()

    print("=" * 78)
    print("HYBRID — y_final = y_lstm + e_xgboost")
    print("=" * 78)
    print(f"rows: {len(out)}  ({out['date'].min():%Y-%m-%d} -> {out['date'].max():%Y-%m-%d})")
    print(
        f"correction magnitude: mean |e| {out['e_pred_xgb'].abs().mean():,.0f}  "
        f"std {out['e_pred_xgb'].std():,.0f}"
    )

    print("\n" + "-" * 78)
    print(f"{'split':<7} {'model':<10} {'MAE':>12} {'RMSE':>12} {'MAPE':>8} {'R2':>9}")
    print("-" * 78)
    for split in ("val", "test"):
        subset = out[out["split"] == split]
        for label, col in (("LSTM", "y_pred_lstm"), ("hybrid", "y_pred_hybrid")):
            m = all_metrics(subset["y_true"], subset[col])
            print(
                f"{split:<7} {label:<10} {m['MAE']:>12,.0f} {m['RMSE']:>12,.0f} "
                f"{m['MAPE']:>7.2f}% {m['R2']:>9.4f}"
            )
    print("-" * 78)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(HYBRID_PREDICTIONS_FILE, index=False)
    print(f"\nsaved: {HYBRID_PREDICTIONS_FILE}")


if __name__ == "__main__":
    main()
