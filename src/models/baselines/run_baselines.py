"""Phase 3 entry point: split, apply the warm-up policy, run all baselines, report.

Every model here consumes the same trimmed frame and the same split boundaries from
src.utils.splits, so the comparison table is apples-to-apples by construction.
"""

import json
from pathlib import Path

import pandas as pd

from src.evaluation.metrics import comparison_table, format_table
from src.features.build_features import FEATURES_DAILY_FILE
from src.models.baselines.moving_average import (
    predict_moving_average,
    predict_moving_average_28,
)
from src.models.baselines.persistence import predict_persistence, predict_seasonal_naive
from src.models.baselines.sarimax import (
    EXOG_COLS,
    SarimaxOrder,
    fit_and_forecast,
    grid_search_order,
)
from src.utils.paths import PROCESSED_DIR
from src.utils.splits import apply_warmup_policy, chronological_split, split_masks

TARGET_COL = "total"
BASELINE_PREDICTIONS_FILE: Path = PROCESSED_DIR / "baseline_predictions.parquet"
SARIMAX_ORDER_FILE: Path = PROCESSED_DIR / "sarimax_selected_order.json"

DESCRIPTIONS = {
    "persistence": "yhat_t = y_{t-1}",
    "seasonal_naive": "yhat_t = y_{t-7} (same weekday last week)",
    "moving_average_7": "yhat_t = mean(y_{t-7}..y_{t-1})",
    "moving_average_28": "yhat_t = mean(y_{t-28}..y_{t-1})",
    "sarimax": "SARIMAX + calendar exog, walk-forward 1-step",
}


def run(
    features_path: Path | str = FEATURES_DAILY_FILE,
    skip_grid_search: bool = False,
    cached_order: SarimaxOrder | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], SarimaxOrder]:
    """Run every baseline and build per-split comparison tables."""
    df = pd.read_parquet(features_path).sort_values("date").reset_index(drop=True)

    bounds = chronological_split(df["date"])
    print("=" * 78)
    print("CHRONOLOGICAL SPLIT (70/15/15, single source of truth: src/utils/splits.py)")
    print("=" * 78)
    print(bounds.describe())

    trimmed, dropped = apply_warmup_policy(df, bounds)
    print(
        f"\nwarm-up policy    : dropped {dropped} row(s) "
        f"({df['date'].iloc[0]:%Y-%m-%d} -> {df['date'].iloc[dropped - 1]:%Y-%m-%d}), "
        "all inside the training block"
    )
    print(f"rows after trim   : {len(trimmed)} (was {len(df)})")

    masks = split_masks(trimmed, bounds)
    for name, mask in masks.items():
        print(f"  {name:<6}: {int(mask.sum()):>4} rows")

    # ---- baselines -------------------------------------------------------------------
    preds: dict[str, pd.Series] = {
        "persistence": predict_persistence(trimmed),
        "seasonal_naive": predict_seasonal_naive(trimmed),
        "moving_average_7": predict_moving_average(trimmed),
        "moving_average_28": predict_moving_average_28(trimmed),
    }

    print("\n" + "=" * 78)
    print("SARIMAX — grid search over (p,d,q)(P,D,Q,7), AIC on TRAINING SET ONLY")
    print("=" * 78)
    if cached_order is not None:
        spec = cached_order
        print(f"using cached order: {spec}")
    elif skip_grid_search:
        spec = SarimaxOrder((1, 1, 1), (1, 1, 1, 7), float("nan"))
        print(f"grid search skipped; using default {spec.order}x{spec.seasonal_order}")
    else:
        spec, grid_table = grid_search_order(
            trimmed.loc[masks["train"], TARGET_COL],
            trimmed.loc[masks["train"], EXOG_COLS],
        )
        print(f"\nselected: {spec}")
        print("\ntop 5 by AIC:")
        print(grid_table.head(5).to_string(index=False))

    print(f"\nexog ({len(EXOG_COLS)} calendar columns, no lag features):")
    for col in EXOG_COLS:
        print(f"  - {col}")

    preds["sarimax"] = fit_and_forecast(trimmed, masks["train"], spec)

    # ---- scoring ---------------------------------------------------------------------
    predictions = pd.DataFrame({"date": trimmed["date"], TARGET_COL: trimmed[TARGET_COL]})
    for name, series in preds.items():
        predictions[name] = series.to_numpy()

    tables: dict[str, pd.DataFrame] = {}
    for split_name, mask in masks.items():
        subset = predictions[mask.to_numpy()]
        results = {
            name: (subset[TARGET_COL], subset[name]) for name in preds if name in subset
        }
        tables[split_name] = comparison_table(results, DESCRIPTIONS)

    return predictions, tables, spec


def main() -> None:
    predictions, tables, spec = run()

    for split_name in ("train", "val", "test"):
        print("\n" + "=" * 78)
        print(f"BASELINE COMPARISON — {split_name.upper()} SPLIT")
        print("=" * 78)
        print(format_table(tables[split_name]))

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(BASELINE_PREDICTIONS_FILE, index=False)
    SARIMAX_ORDER_FILE.write_text(
        json.dumps(
            {
                "order": list(spec.order),
                "seasonal_order": list(spec.seasonal_order),
                "aic": spec.aic,
                "exog_cols": EXOG_COLS,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\npredictions saved : {BASELINE_PREDICTIONS_FILE}")
    print(f"selected order    : {SARIMAX_ORDER_FILE}")


if __name__ == "__main__":
    main()
