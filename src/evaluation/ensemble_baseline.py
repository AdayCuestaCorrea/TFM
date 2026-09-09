"""Experiment B: simple ensembles of SARIMAX and XGBoost-alone.

NOT a pass/fail experiment. Phase 5 established that the two best models are SARIMAX and
XGBoost-alone, and that they fail differently -- SARIMAX is strongest on ordinary working
days (test MAE 148,297 vs 169,488) while XGBoost-alone is far better on weekends and
festivo days (domingo 100,200 vs 169,129; festivo 251,553 vs 271,410). Errors that
decompose that cleanly are the textbook case where averaging helps, so the discussion
chapter needs the number whether or not it flatters the hybrid.

NO RETRAINING. Both components come from artifacts already fitted in Phases 3 and 5;
combining saved predictions cannot introduce leakage that was not already present, and
keeps this experiment a pure post-hoc combination.

TWO VARIANTS:
  - ensemble_equal:       0.5 * sarimax + 0.5 * xgboost_alone
  - ensemble_inverse_mae: weights proportional to 1 / val_MAE of each component

CAVEAT ON THE INVERSE-MAE VARIANT, which matters for how its numbers may be cited: its
weights are derived from validation MAE, so its VAL score is mildly optimistic -- the
weights were chosen using the very rows being scored. Its TEST score is clean, because the
weights were fixed before test data was consulted. Report the val figure with that caveat
attached, or lean on the test figure.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.metrics import all_metrics, mae
from src.utils.paths import PROCESSED_DIR

ENSEMBLE_PREDICTIONS_FILE: Path = PROCESSED_DIR / "ensemble_predictions.parquet"

COMPONENTS = ["sarimax", "xgboost_alone"]


def equal_weights() -> dict[str, float]:
    """0.5 / 0.5 -- fixed a priori, no data consulted."""
    return {name: 0.5 for name in COMPONENTS}


def inverse_mae_weights(val_df: pd.DataFrame) -> dict[str, float]:
    """Weights proportional to 1 / val_MAE, normalised to sum to 1."""
    inverse = {}
    for name in COMPONENTS:
        pair = val_df[["y_true", name]].dropna()
        inverse[name] = 1.0 / mae(pair["y_true"], pair[name])
    total = sum(inverse.values())
    return {name: w / total for name, w in inverse.items()}


def build_ensembles(df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Add both ensemble columns to the assembled prediction frame.

    Returns:
        (frame with ensemble columns, {variant: weights used}).
    """
    if df is None:
        from src.evaluation.full_comparison import assemble_predictions

        df = assemble_predictions()

    missing = [c for c in COMPONENTS if c not in df.columns]
    if missing:
        raise ValueError(f"ensemble: missing component prediction(s) {missing}")

    out = df.copy()

    w_equal = equal_weights()
    out["ensemble_equal"] = sum(w_equal[name] * out[name] for name in COMPONENTS)

    w_inverse = inverse_mae_weights(out[out["split"] == "val"])
    out["ensemble_inverse_mae"] = sum(w_inverse[name] * out[name] for name in COMPONENTS)

    return out, {"ensemble_equal": w_equal, "ensemble_inverse_mae": w_inverse}


def main() -> None:
    out, weights = build_ensembles()

    print("=" * 78)
    print("EXPERIMENT B — ensembles of SARIMAX and XGBoost-alone (no retraining)")
    print("=" * 78)
    for variant, w in weights.items():
        pretty = ", ".join(f"{name} {value:.4f}" for name, value in w.items())
        print(f"  {variant:<22}: {pretty}")
    print(
        "\n  NOTE: ensemble_inverse_mae's weights come from validation MAE, so its VAL\n"
        "  score is mildly optimistic. Its TEST score is clean."
    )

    print("\n" + "-" * 78)
    print(f"{'split':<7} {'model':<22} {'MAE':>12} {'RMSE':>12} {'MAPE':>8} {'R2':>9}")
    print("-" * 78)
    for split in ("val", "test"):
        subset = out[out["split"] == split]
        for name in ("sarimax", "xgboost_alone", "ensemble_equal", "ensemble_inverse_mae"):
            pair = subset[["y_true", name]].dropna()
            m = all_metrics(pair["y_true"], pair[name])
            print(
                f"{split:<7} {name:<22} {m['MAE']:>12,.0f} {m['RMSE']:>12,.0f} "
                f"{m['MAPE']:>7.2f}% {m['R2']:>9.4f}"
            )
        print("-" * 78)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    keep = ["date", "split", "y_true", *COMPONENTS, "ensemble_equal", "ensemble_inverse_mae"]
    out[keep].to_parquet(ENSEMBLE_PREDICTIONS_FILE, index=False)
    print(f"\nsaved: {ENSEMBLE_PREDICTIONS_FILE}")


if __name__ == "__main__":
    main()
