"""Decision support: is excluding fold 1's residuals a redundancy or an information loss?

Phase 4 found fold 1 degenerate -- its OOF predictions have std 0.199x the actuals and
correlate 0.300, because 147 fitting sequences were not enough for the recurrent model to
escape a near-constant. Its residuals therefore encode stage-1 failure rather than the
calendar structure stage 2 is meant to learn, which argues for dropping them.

But "these rows are low quality" does not by itself justify discarding them. If fold 1's
OOF block is the ONLY place some holiday category appears, dropping it means XGBoost never
sees that category during residual training and must extrapolate blind at test time. That
is a real loss, and a different decision from dropping redundant rows.

So this module answers a narrow, checkable question: does every holiday category and every
bridge day occurring in fold 1's block also occur somewhere in folds 2-5? If yes, exclusion
costs coverage of nothing and is safe. If no, the caller must stop rather than proceed.

This is a REPORT, not a gate that silently blocks the pipeline: it prints the table and
returns a verdict, and `is_exclusion_safe` is what downstream code branches on.
"""

from pathlib import Path

import pandas as pd

from src.features.build_features import FEATURES_DAILY_FILE
from src.models.lstm.oof import OOF_PREDICTIONS_FILE

DEGENERATE_FOLD = 1

HOLIDAY_COLS = [
    "feat_holiday_type_none",
    "feat_holiday_type_festivo_nacional",
    "feat_holiday_type_festivo_de_la_comunidad_de_madrid",
    "feat_holiday_type_festivo_local_de_la_ciudad_de_madrid",
]
DAY_TYPE_COLS = [
    "feat_day_type_laborable",
    "feat_day_type_sabado",
    "feat_day_type_domingo",
    "feat_day_type_festivo",
    "feat_day_type_domingo_festivo",
]
BRIDGE_COL = "feat_is_bridge_day"


def build_coverage_table(
    oof_path: Path | str = OOF_PREDICTIONS_FILE,
    features_path: Path | str = FEATURES_DAILY_FILE,
    excluded_fold: int = DEGENERATE_FOLD,
) -> tuple[pd.DataFrame, bool]:
    """Compare category occurrence in the excluded fold against the retained folds.

    Returns:
        (coverage table, is_exclusion_safe). Safe means every category present in the
        excluded fold has at least one occurrence in the retained folds.
    """
    oof = pd.read_parquet(oof_path)
    feats = pd.read_parquet(features_path)

    categorical_cols = [*DAY_TYPE_COLS, *HOLIDAY_COLS, BRIDGE_COL]
    df = oof[oof["has_oof"]].merge(
        feats[["date", *categorical_cols]], on="date", how="left"
    )

    excluded = df[df["fold"] == excluded_fold]
    retained = df[df["fold"] != excluded_fold]

    rows = []
    for col in categorical_cols:
        in_excluded = int(excluded[col].sum())
        in_retained = int(retained[col].sum())
        rows.append(
            {
                "category": col.replace("feat_", ""),
                f"count_in_fold_{excluded_fold}": in_excluded,
                "count_in_folds_2_5": in_retained,
                # The only combination that constitutes information loss: present in the
                # fold being dropped, absent everywhere else.
                "zero_coverage_elsewhere": in_excluded > 0 and in_retained == 0,
            }
        )

    table = pd.DataFrame(rows)
    is_safe = not bool(table["zero_coverage_elsewhere"].any())
    return table, is_safe


def main() -> None:
    table, is_safe = build_coverage_table()

    oof = pd.read_parquet(OOF_PREDICTIONS_FILE)
    excluded = oof[(oof["has_oof"]) & (oof["fold"] == DEGENERATE_FOLD)]
    retained = oof[(oof["has_oof"]) & (oof["fold"] != DEGENERATE_FOLD)]

    print("=" * 78)
    print("FOLD 1 EXCLUSION — CATEGORY COVERAGE CHECK")
    print("=" * 78)
    print(
        f"fold {DEGENERATE_FOLD} block : {excluded['date'].min():%Y-%m-%d} -> "
        f"{excluded['date'].max():%Y-%m-%d}  ({len(excluded)} rows)"
    )
    print(
        f"folds 2-5 block: {retained['date'].min():%Y-%m-%d} -> "
        f"{retained['date'].max():%Y-%m-%d}  ({len(retained)} rows)"
    )
    print()
    print(table.to_string(index=False))
    print()

    if is_safe:
        print("VERDICT: exclusion is SAFE.")
        print(
            "Every category occurring in fold 1 also occurs in folds 2-5, so dropping "
            "fold 1 removes redundant coverage, not unique information."
        )
    else:
        lost = table[table["zero_coverage_elsewhere"]]["category"].tolist()
        print("VERDICT: DO NOT EXCLUDE — zero coverage elsewhere.")
        print(
            f"Categor{'y' if len(lost) == 1 else 'ies'} {lost} appear ONLY in fold 1. "
            "Dropping it would leave the residual model with no training example of "
            "them, forcing blind extrapolation at test time."
        )
    print("=" * 78)


if __name__ == "__main__":
    main()
