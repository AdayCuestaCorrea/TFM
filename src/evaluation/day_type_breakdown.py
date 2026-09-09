"""Error breakdown by day type and bridge day, shared across every model.

Written once and applied to all four models rather than duplicated per model: the whole
point is a like-for-like comparison, and two near-copies of a grouping function are exactly
how a subtle difference in denominators creeps in and invalidates it.

This is where the thesis's central claim is actually visible. Aggregate MAE can improve for
uninteresting reasons -- a model that is slightly better on the 800-odd ordinary working
days will move the headline number while doing nothing about the failure mode that
motivated the architecture. The Phase 4 step-0 diagnostic established that the linear
baseline's error concentrates on festivo and bridge days (2.70x ordinary working days). The
claim under test is that the residual stage specifically reduces THAT error, so it has to be
read group by group.
"""

import pandas as pd

from src.evaluation.metrics import mae, mape

DAY_TYPE_COL = "day_type"
BRIDGE_COL = "feat_is_bridge_day"


def breakdown(
    y_true: pd.Series, y_pred: pd.Series, features_df: pd.DataFrame
) -> pd.DataFrame:
    """MAE and MAPE grouped by day type, plus the laborable bridge-day split.

    Args:
        y_true: Actual values, index-aligned to `features_df`.
        y_pred: Predictions, index-aligned to `features_df`.
        features_df: Rows carrying `day_type` and `feat_is_bridge_day`.

    Returns:
        DataFrame [group, n, MAE, MAPE], one row per day-type category plus
        'laborable, bridge day' and 'laborable, ordinary'.
    """
    missing = [c for c in (DAY_TYPE_COL, BRIDGE_COL) if c not in features_df.columns]
    if missing:
        raise ValueError(f"day_type_breakdown: missing column(s) {missing}")

    frame = pd.DataFrame(
        {
            "y_true": pd.Series(y_true).to_numpy(dtype="float64"),
            "y_pred": pd.Series(y_pred).to_numpy(dtype="float64"),
            "day_type": features_df[DAY_TYPE_COL].astype(str).to_numpy(),
            "bridge": features_df[BRIDGE_COL].to_numpy(),
        }
    )

    rows = []
    for group, subset in frame.groupby("day_type"):
        rows.append(
            {
                "group": group,
                "n": len(subset),
                "MAE": mae(subset["y_true"], subset["y_pred"]),
                "MAPE": mape(subset["y_true"], subset["y_pred"]),
            }
        )

    # Bridge days are by definition laborable, so isolating them within laborable separates
    # the puente effect from the weekday/weekend level shift.
    laborable = frame[frame["day_type"] == "laborable"]
    for label, subset in (
        ("laborable, bridge day", laborable[laborable["bridge"] == 1]),
        ("laborable, ordinary", laborable[laborable["bridge"] == 0]),
    ):
        if len(subset) == 0:
            continue
        rows.append(
            {
                "group": label,
                "n": len(subset),
                "MAE": mae(subset["y_true"], subset["y_pred"]),
                "MAPE": mape(subset["y_true"], subset["y_pred"]),
            }
        )

    return pd.DataFrame(rows)


def compare_models(
    predictions: dict[str, tuple[pd.Series, pd.Series]], features_df: pd.DataFrame
) -> pd.DataFrame:
    """Side-by-side breakdown for several models on the same rows.

    Args:
        predictions: {model name: (y_true, y_pred)}, all aligned to `features_df`.
        features_df: The matching feature rows.

    Returns:
        Wide table indexed by group with a MAE and MAPE column per model.
    """
    frames = []
    for name, (y_true, y_pred) in predictions.items():
        table = breakdown(y_true, y_pred, features_df).set_index("group")
        table = table.rename(columns={"MAE": f"{name}_MAE", "MAPE": f"{name}_MAPE"})
        frames.append(table.drop(columns="n") if frames else table)

    return pd.concat(frames, axis=1).reset_index()


def format_comparison(table: pd.DataFrame) -> str:
    """Fixed-width rendering with thousands separators and percent signs."""
    display = table.copy()
    for col in display.columns:
        if col.endswith("_MAE"):
            display[col] = display[col].map(lambda v: f"{v:,.0f}")
        elif col.endswith("_MAPE"):
            display[col] = display[col].map(lambda v: f"{v:.2f}%")
    return display.to_string(index=False)


def main() -> None:
    """Test-split breakdown for SARIMAX, LSTM-alone, XGBoost-alone and the hybrid."""
    from src.evaluation.full_comparison import assemble_predictions
    from src.features.build_features import FEATURES_DAILY_FILE

    df = assemble_predictions()
    test = df[df["split"] == "test"].copy()

    feats = pd.read_parquet(FEATURES_DAILY_FILE)[["date", DAY_TYPE_COL, BRIDGE_COL]]
    test = test.merge(feats, on="date", how="left")

    models = ["sarimax", "lstm_alone", "xgboost_alone", "hybrid"]
    test = test.dropna(subset=[*models, "y_true"]).reset_index(drop=True)

    print("=" * 110)
    print("DAY-TYPE ERROR BREAKDOWN — TEST SPLIT")
    print("=" * 110)
    print(
        f"rows: {len(test)}  "
        f"({test['date'].min():%Y-%m-%d} -> {test['date'].max():%Y-%m-%d})"
    )
    print()

    table = compare_models(
        {name: (test["y_true"], test[name]) for name in models}, test
    )
    print(format_comparison(table))

    print()
    print("-" * 110)
    print("Reading: the architecture's claim is that stage 2 reduces error specifically on")
    print("festivo and bridge days, not merely in aggregate. Compare those rows, not the")
    print("laborable-ordinary row, which dominates the headline MAE by sheer count.")
    print("-" * 110)


if __name__ == "__main__":
    main()
