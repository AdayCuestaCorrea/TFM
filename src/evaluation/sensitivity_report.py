"""Phase 5b consolidated report: master table and day-type breakdown, both extended.

Extends Phase 5's tables with `hybrid_weighted` (Experiment A) and the two ensemble
variants (Experiment B), scored by the same metrics.py on the same split boundaries so the
new rows are directly comparable to everything above them.

Also evaluates Experiment A against its PRE-REGISTERED success criterion, which was fixed
before any of these numbers existed:
    A succeeds only if it reduces test MAE on festivo + bridge-day rows AND leaves overall
    test MAE no more than 5% worse than Phase 5's hybrid.
The check is coded here rather than eyeballed, so the verdict cannot drift to fit the
result.
"""

from pathlib import Path

import pandas as pd

from src.evaluation.day_type_breakdown import (
    BRIDGE_COL,
    DAY_TYPE_COL,
    compare_models,
    format_comparison,
)
from src.evaluation.ensemble_baseline import build_ensembles
from src.evaluation.metrics import comparison_table, format_table, mae
from src.features.build_features import FEATURES_DAILY_FILE
from src.models.hybrid_residual.xgboost_residual_weighted import WEIGHTED_PREDICTIONS_FILE
from src.utils.paths import PROCESSED_DIR

SENSITIVITY_FILE: Path = PROCESSED_DIR / "sensitivity_comparison.parquet"

# Fixed before running Experiment A.
PHASE5_HYBRID_TEST_MAE = 234_223.0
MAX_OVERALL_DEGRADATION = 0.05

DESCRIPTIONS = {
    "persistence": "yhat_t = y_{t-1}",
    "seasonal_naive": "yhat_t = y_{t-7}",
    "moving_average_7": "trailing 7-day mean",
    "moving_average_28": "trailing 28-day mean",
    "sarimax": "SARIMAX(2,1,2)x(0,1,2,7) + calendar exog",
    "lstm_alone": "Stage 1 only: univariate LSTM(32), W=28",
    "xgboost_alone": "XGBoost on total, full feature matrix",
    "hybrid": "Stage 1 LSTM + Stage 2 XGBoost residual",
    "hybrid_weighted": "[5b-A] hybrid, irregular days upweighted 3x",
    "ensemble_equal": "[5b-B] 0.50 sarimax + 0.50 xgboost_alone",
    "ensemble_inverse_mae": "[5b-B] inverse-val-MAE weighted ensemble",
}

ORDER = [
    "ensemble_inverse_mae",
    "ensemble_equal",
    "xgboost_alone",
    "sarimax",
    "hybrid_weighted",
    "hybrid",
    "lstm_alone",
    "seasonal_naive",
    "moving_average_7",
    "moving_average_28",
    "persistence",
]

BREAKDOWN_MODELS = [
    "sarimax",
    "xgboost_alone",
    "hybrid",
    "hybrid_weighted",
    "ensemble_equal",
]

IRREGULAR_GROUPS = ["festivo", "laborable, bridge day"]


def assemble_all() -> pd.DataFrame:
    """Every model's predictions, including the Phase 5b variants, on one frame."""
    out, _ = build_ensembles()

    weighted = pd.read_parquet(WEIGHTED_PREDICTIONS_FILE)[
        ["date", "y_pred_hybrid_weighted"]
    ].rename(columns={"y_pred_hybrid_weighted": "hybrid_weighted"})

    return out.merge(weighted, on="date", how="left")


def build_master_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    models = [m for m in ORDER if m in df.columns]
    tables: dict[str, pd.DataFrame] = {}
    for split in ("val", "test"):
        subset = df[df["split"] == split]
        results = {}
        for name in models:
            pair = subset[["y_true", name]].dropna()
            if not pair.empty:
                results[name] = (pair["y_true"], pair[name])
        if results:
            tables[split] = comparison_table(results, DESCRIPTIONS)
    return tables


def build_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    feats = pd.read_parquet(FEATURES_DAILY_FILE)[["date", DAY_TYPE_COL, BRIDGE_COL]]
    test = df[df["split"] == "test"].merge(feats, on="date", how="left")
    models = [m for m in BREAKDOWN_MODELS if m in test.columns]
    test = test.dropna(subset=[*models, "y_true"]).reset_index(drop=True)
    return compare_models({m: (test["y_true"], test[m]) for m in models}, test)


def evaluate_criterion_a(breakdown: pd.DataFrame, master: pd.DataFrame) -> dict:
    """Apply Experiment A's pre-registered success criterion mechanically."""
    rows = breakdown.set_index("group")
    irregular = [g for g in IRREGULAR_GROUPS if g in rows.index]

    # Sample-size-weighted MAE across the irregular groups.
    def weighted(model: str) -> float:
        total_n = sum(rows.loc[g, "n"] for g in irregular)
        return sum(rows.loc[g, "n"] * rows.loc[g, f"{model}_MAE"] for g in irregular) / total_n

    base = weighted("hybrid")
    variant = weighted("hybrid_weighted")

    test_mae = master.set_index("Model").loc["hybrid_weighted", "MAE"]
    degradation = (test_mae - PHASE5_HYBRID_TEST_MAE) / PHASE5_HYBRID_TEST_MAE

    improved = variant < base
    within_budget = degradation <= MAX_OVERALL_DEGRADATION

    return {
        "irregular_mae_hybrid": base,
        "irregular_mae_weighted": variant,
        "irregular_improved": improved,
        "overall_test_mae": test_mae,
        "overall_change_pct": degradation * 100.0,
        "within_budget": within_budget,
        "criterion_met": improved and within_budget,
    }


def main() -> None:
    df = assemble_all()
    tables = build_master_tables(df)
    breakdown = build_breakdown(df)

    print("=" * 100)
    print("PHASE 5b — PRE-REGISTERED CRITERIA (fixed before any result was seen)")
    print("=" * 100)
    print(
        "A succeeds ONLY IF: test MAE on festivo + bridge-day rows falls, AND overall test\n"
        f"   MAE is no more than {MAX_OVERALL_DEGRADATION:.0%} worse than Phase 5's hybrid "
        f"({PHASE5_HYBRID_TEST_MAE:,.0f})."
    )
    print("B is not pass/fail: it is a required comparison point either way.")
    print("No third variant, no sweep, no architecture change is run based on these results.")

    for split in ("val", "test"):
        print("\n" + "=" * 100)
        print(f"MASTER COMPARISON (extended) — {split.upper()} SPLIT")
        print("=" * 100)
        print(format_table(tables[split]))

    print("\n" + "=" * 100)
    print("DAY-TYPE BREAKDOWN (extended) — TEST SPLIT")
    print("=" * 100)
    print(format_comparison(breakdown))

    verdict = evaluate_criterion_a(breakdown, tables["test"])
    print("\n" + "=" * 100)
    print("EXPERIMENT A — VERDICT AGAINST THE PRE-REGISTERED CRITERION")
    print("=" * 100)
    print(
        f"irregular-day MAE (n-weighted, festivo + bridge): "
        f"hybrid {verdict['irregular_mae_hybrid']:,.0f} -> "
        f"weighted {verdict['irregular_mae_weighted']:,.0f}   "
        f"[{'IMPROVED' if verdict['irregular_improved'] else 'NOT IMPROVED'}]"
    )
    print(
        f"overall test MAE: {verdict['overall_test_mae']:,.0f} vs "
        f"{PHASE5_HYBRID_TEST_MAE:,.0f} = {verdict['overall_change_pct']:+.2f}%   "
        f"[{'WITHIN' if verdict['within_budget'] else 'OUTSIDE'} the "
        f"{MAX_OVERALL_DEGRADATION:.0%} budget]"
    )
    print(
        f"\nCRITERION {'MET' if verdict['criterion_met'] else 'NOT MET'}"
    )
    print("=" * 100)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(
        [t.assign(split=split) for split, t in tables.items()], ignore_index=True
    )
    combined.to_parquet(SENSITIVITY_FILE, index=False)
    print(f"\nsaved: {SENSITIVITY_FILE}")


if __name__ == "__main__":
    main()
