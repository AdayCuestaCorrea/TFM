"""Exploratory diagnostic: where does the linear baseline systematically fail?

Motivation. The residual hybrid's premise is that a linear/seasonal model captures the
regular weekly-annual structure while a nonlinear learner mops up what is left --
principally the irregular calendar effects (holidays, bridge days) whose demand impact is
not a fixed additive offset. That premise is testable BEFORE building the hybrid: if
SARIMAX's error is no larger on festivo/bridge days than on ordinary working days, then
the nonlinear correction has little systematic structure to find and the hybrid's expected
value-add is weaker than assumed.

This module answers that question and nothing else. It fits nothing and is not part of any
modeling path.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.features.build_features import FEATURES_DAILY_FILE
from src.models.baselines.run_baselines import BASELINE_PREDICTIONS_FILE
from src.utils.splits import chronological_split, split_masks

MODEL_COL = "sarimax"
TARGET_COL = "total"


def load_residuals(
    predictions_path: Path | str = BASELINE_PREDICTIONS_FILE,
    features_path: Path | str = FEATURES_DAILY_FILE,
    model_col: str = MODEL_COL,
) -> pd.DataFrame:
    """Join baseline predictions to the calendar features and compute residuals."""
    preds = pd.read_parquet(predictions_path)
    feats = pd.read_parquet(features_path)

    calendar_cols = [
        "date",
        "day_type",
        "feat_is_bridge_day",
        "feat_is_weekend",
        "feat_holiday_type_festivo_nacional",
        "feat_holiday_type_festivo_de_la_comunidad_de_madrid",
        "feat_holiday_type_festivo_local_de_la_ciudad_de_madrid",
    ]
    df = preds.merge(feats[calendar_cols], on="date", how="left")

    df["residual"] = df[TARGET_COL] - df[model_col]
    df["abs_residual"] = df["residual"].abs()
    # Percentage error makes groups with different demand levels comparable: a 300k miss
    # on a 6M weekday and on a 2.5M Sunday are not the same failure.
    df["abs_pct_error"] = (df["abs_residual"] / df[TARGET_COL]) * 100.0
    return df


def _summarise(df: pd.DataFrame, group: pd.Series, label: str) -> pd.DataFrame:
    """Mean/median absolute error and bias per group."""
    out = (
        df.groupby(group, observed=False)
        .agg(
            n=("abs_residual", "size"),
            mean_abs_error=("abs_residual", "mean"),
            median_abs_error=("abs_residual", "median"),
            mean_abs_pct=("abs_pct_error", "mean"),
            mean_signed_bias=("residual", "mean"),
        )
        .reset_index()
        .rename(columns={group.name: label})
    )
    return out.sort_values("mean_abs_error", ascending=False)


def run(evaluate_on: str = "all") -> dict[str, pd.DataFrame]:
    """Compute residual summaries grouped by calendar characteristics.

    Args:
        evaluate_on: 'all', 'train', 'val', or 'test'. Defaults to the whole period, since
            this is exploratory and a larger sample gives a steadier picture of holidays,
            which are rare (49 festivo days across the full range).

    Returns:
        {summary name: table}.
    """
    df = load_residuals()

    if evaluate_on != "all":
        bounds = chronological_split(
            pd.read_parquet(FEATURES_DAILY_FILE)["date"].sort_values()
        )
        df = df[split_masks(df, bounds)[evaluate_on].to_numpy()]

    tables: dict[str, pd.DataFrame] = {}

    tables["by_day_type"] = _summarise(df, df["day_type"].astype(str), "day_type")

    bridge = df["feat_is_bridge_day"].map({0: "not bridge day", 1: "bridge day (puente)"})
    bridge.name = "bridge"
    tables["by_bridge_day"] = _summarise(df, bridge, "bridge_day")

    holiday_cols = [c for c in df.columns if c.startswith("feat_holiday_type_")]
    holiday_label = pd.Series("none (not a holiday)", index=df.index, name="holiday")
    for col in holiday_cols:
        pretty = col.replace("feat_holiday_type_", "").replace("_", " ")
        holiday_label = holiday_label.mask(df[col] == 1, pretty)
    tables["by_holiday_type"] = _summarise(df, holiday_label, "holiday_type")

    # Laborable-only bridge comparison: isolates the puente effect from the weekend effect,
    # since every bridge day is by construction a laborable day.
    laborable = df[df["day_type"].astype(str) == "laborable"]
    lab_bridge = laborable["feat_is_bridge_day"].map(
        {0: "laborable, ordinary", 1: "laborable, bridge day"}
    )
    lab_bridge.name = "laborable_bridge"
    tables["laborable_only"] = _summarise(laborable, lab_bridge, "laborable_subset")

    return tables


def main() -> None:
    df = load_residuals()
    print("=" * 78)
    print("DIAGNOSTIC — SARIMAX residual magnitude by calendar characteristic")
    print("=" * 78)
    print(f"model     : {MODEL_COL} (Phase 3 baseline)")
    print(f"period    : {df['date'].min():%Y-%m-%d} -> {df['date'].max():%Y-%m-%d}")
    print(f"rows      : {len(df)}")
    print(f"overall MAE: {df['abs_residual'].mean():,.0f}  ({df['abs_pct_error'].mean():.2f}%)")
    print("\nNOTE: exploratory only. Computed over the whole period (train+val+test)")
    print("because holidays are rare; this is NOT an out-of-sample performance estimate.")

    for name, table in run().items():
        print("\n" + "-" * 78)
        print(name)
        print("-" * 78)
        display = table.copy()
        for col in ("mean_abs_error", "median_abs_error", "mean_signed_bias"):
            display[col] = display[col].map(lambda v: f"{v:,.0f}")
        display["mean_abs_pct"] = display["mean_abs_pct"].map(lambda v: f"{v:.2f}%")
        print(display.to_string(index=False))

    # Headline comparison for the hybrid's premise.
    lab = run()["laborable_only"]
    if len(lab) == 2:
        ordinary = lab[lab["laborable_subset"] == "laborable, ordinary"].iloc[0]
        bridgey = lab[lab["laborable_subset"] == "laborable, bridge day"].iloc[0]
        ratio = bridgey["mean_abs_error"] / ordinary["mean_abs_error"]
        print("\n" + "=" * 78)
        print(
            f"Bridge days carry {ratio:.2f}x the mean absolute error of ordinary "
            "laborable days."
        )
        print("=" * 78)


if __name__ == "__main__":
    main()
