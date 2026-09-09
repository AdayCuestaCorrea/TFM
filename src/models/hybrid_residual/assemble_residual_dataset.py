"""Assemble the Stage 2 residual training set from Phase 4's OOF predictions.

TWO EXCLUSIONS, BOTH DELIBERATE AND BOTH DOCUMENTED IN THE OUTPUT:

1. `has_oof == False` -- the first 200 training rows (2023-01-29 -> 2023-08-16). These
   never received an OOF prediction because predicting them would have required a model
   trained on less than the minimum initial window. There is no residual to compute.

2. `fold == 1` -- the degenerate fold (2023-08-17 -> 2023-12-31, 137 rows). Its OOF
   predictions have std 0.199x the actuals and correlate 0.300: with 147 fitting sequences
   the stage-1 model never escaped predicting a near-constant. Its residuals therefore
   encode STAGE-1 FAILURE, not the calendar structure stage 2 exists to learn. Training on
   them would teach XGBoost to correct a phantom that the final (fully trained) stage-1
   model does not exhibit. Justified by fold_coverage_check.py, which confirms every
   holiday/bridge category in fold 1 also occurs in folds 2-5, so this drops redundant
   coverage rather than unique information.

RESIDUAL SIGN CONVENTION: residual = y_true - y_pred_oof, so the hybrid combines as
y_final = y_lstm + e_pred. Getting this backwards would subtract the correction and make
the hybrid worse than its stage 1; combine.py asserts the identity explicitly.

KNOWN LIMITATION, worth stating rather than discovering later: the residuals here come
from fold models trained on 337-748 rows, while the stage-1 model used at inference is
trained on all 889. The final model is therefore slightly better than the models that
generated these residuals, so the residual distribution stage 2 learns is a little wider
than the one it will face. This is inherent to OOF stacking and is the accepted cost of
the alternative being far worse -- in-sample residuals would be optimistically small and
structurally unlike anything seen at inference (CLAUDE.md Critical Design Decision).
"""

from pathlib import Path

import pandas as pd

from src.features.build_features import FEATURES_DAILY_FILE, feature_columns
from src.features.weather_features import WEATHER_COLS
from src.models.lstm.oof import OOF_PREDICTIONS_FILE
from src.utils.paths import PROCESSED_DIR

EXCLUDED_FOLD = 1
RESIDUAL_TRAINING_FILE: Path = PROCESSED_DIR / "residual_training_set.parquet"

FORBIDDEN_SAME_DAY = ["total", "metro", "emt", "carretera", "cercanias"]


def assemble(
    oof_path: Path | str = OOF_PREDICTIONS_FILE,
    features_path: Path | str = FEATURES_DAILY_FILE,
    excluded_fold: int | None = EXCLUDED_FOLD,
) -> pd.DataFrame:
    """Build the residual training set.

    Returns:
        DataFrame with [date, residual, y_true, y_pred_oof, fold] + every feat_ column.

    Raises:
        ValueError: if the assembled set contains NaNs or any same-day target/operator
            column, or if it is unexpectedly empty.
    """
    oof = pd.read_parquet(oof_path)
    feats = pd.read_parquet(features_path)

    kept = oof[oof["has_oof"]].copy()
    if excluded_fold is not None:
        kept = kept[kept["fold"] != excluded_fold]

    kept["residual"] = kept["y_true"] - kept["y_pred_oof"]

    feat_cols = feature_columns(feats)
    merged = kept[["date", "residual", "y_true", "y_pred_oof", "fold"]].merge(
        feats[["date", *feat_cols]], on="date", how="left"
    )

    if merged.empty:
        raise ValueError("assemble_residual_dataset: produced an empty training set")

    # The Phase 2/3 guards should already make these impossible; assert rather than assume,
    # because a join is exactly where an unexpected column can reappear.
    leaked = [c for c in FORBIDDEN_SAME_DAY if c in feat_cols]
    if leaked:
        raise ValueError(
            f"LEAKAGE GUARD: same-day column(s) {leaked} present in the residual feature "
            "matrix. metro+emt+carretera+cercanias == total exactly."
        )
    leaked_wx = [c for c in WEATHER_COLS if c in feat_cols]
    if leaked_wx:
        raise ValueError(
            f"SAME-DAY WEATHER GUARD: unlagged weather column(s) {leaked_wx} present."
        )

    nulls = merged.isna().sum()
    offenders = nulls[nulls > 0]
    if not offenders.empty:
        raise ValueError(
            f"assemble_residual_dataset: NaNs present {offenders.to_dict()}. The warm-up "
            "trim and OOF exclusions should have guaranteed none."
        )

    return merged.sort_values("date").reset_index(drop=True)


def main() -> None:
    df = assemble()
    feat_cols = [c for c in df.columns if c.startswith("feat_")]

    print("=" * 78)
    print("RESIDUAL TRAINING SET")
    print("=" * 78)
    print(f"rows           : {len(df)}")
    print(f"features       : {len(feat_cols)}")
    print(f"date coverage  : {df['date'].min():%Y-%m-%d} -> {df['date'].max():%Y-%m-%d}")
    print(f"folds retained : {sorted(df['fold'].unique().tolist())} (fold 1 excluded)")
    print(
        f"residual stats : mean {df['residual'].mean():,.0f}  "
        f"std {df['residual'].std():,.0f}  "
        f"min {df['residual'].min():,.0f}  max {df['residual'].max():,.0f}"
    )
    print(f"nulls          : {int(df.isna().sum().sum())}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RESIDUAL_TRAINING_FILE, index=False)
    print(f"\nsaved: {RESIDUAL_TRAINING_FILE}")


if __name__ == "__main__":
    main()
