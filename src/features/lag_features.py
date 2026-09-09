"""Lag features for the target and the four operator series.

LEAKAGE POSTURE — the central point of this module:

Same-day `metro`, `emt`, `carretera`, `cercanias` must NEVER enter the feature set.
Phase 1 verified that `metro + emt + carretera + cercanias == total` on all 1310 rows, so
those four columns are not merely correlated with the target -- they *determine* it
exactly. A model given them same-day would score near-perfectly while learning nothing but
an identity, and would collapse at inference time when the operator split for day t is not
yet known. `add_lag_features` therefore ends with an explicit guard that raises if any of
the four ever appears in its own output.

*Lagged* operator values are legitimate: yesterday's operator split is genuinely known
information at prediction time for today, and it carries real signal (a modal shift shows
up in the split before it shows up in the total).

Ordering note (kept for consistency with the CLAUDE.md leakage rules even though it is
vacuous here): these features use `.shift(n)` directly on the raw series with no rolling
or aggregation involved, so the shift-before-rolling ordering trap that governs
rolling_features.py cannot arise. `shift(n)` for n >= 1 moves strictly past values forward,
never future values backward, so the value at row t is always drawn from row t-n.
"""

import pandas as pd

from src.features.naming import FEATURE_PREFIX

LAGS: list[int] = [1, 2, 3, 7, 14, 21, 28]

TARGET_COL = "total"
OPERATOR_COLS: list[str] = ["metro", "emt", "carretera", "cercanias"]
LAG_SOURCE_COLS: list[str] = [TARGET_COL] + OPERATOR_COLS

# Columns that must never survive into a feature frame, at any lag of 0.
FORBIDDEN_SAME_DAY_COLS: list[str] = [TARGET_COL] + OPERATOR_COLS


def lag_feature_names() -> list[str]:
    """The exact column names this module produces, in order."""
    return [
        f"{FEATURE_PREFIX}{col}_lag_{lag}" for col in LAG_SOURCE_COLS for lag in LAGS
    ]


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build lagged features for the target and operator series.

    Args:
        df: Unified daily frame, sorted ascending by 'date' with a gap-free daily index.
            Must contain 'date' plus every column in LAG_SOURCE_COLS.

    Returns:
        A features-only DataFrame sharing `df`'s index, containing one column per
        (source column, lag) pair. The first `max(LAGS)` rows contain NaNs by
        construction; they are left in place deliberately (see build_features).

    Raises:
        ValueError: if required columns are missing, if the frame is not chronologically
            sorted (which would make `.shift()` meaningless), or if the leakage guard
            detects a raw same-day column in the output.
    """
    missing = [c for c in ["date", *LAG_SOURCE_COLS] if c not in df.columns]
    if missing:
        raise ValueError(f"lag_features: missing required column(s) {missing}")

    # shift(n) is positional, not date-aware: it only equals "n days ago" if the rows are
    # sorted and the daily index is gap-free. Phase 1 guarantees both, but a silent
    # re-order upstream would corrupt every lag here, so verify rather than assume.
    if not df["date"].is_monotonic_increasing:
        raise ValueError(
            "lag_features: frame is not sorted ascending by date; .shift() would draw "
            "values from the wrong rows."
        )
    gaps = pd.date_range(df["date"].min(), df["date"].max(), freq="D").difference(
        pd.DatetimeIndex(df["date"])
    )
    if len(gaps) > 0:
        raise ValueError(
            f"lag_features: date index has {len(gaps)} gap(s), so positional .shift() "
            f"does not equal a calendar-day lag. First gaps: "
            f"{[d.strftime('%Y-%m-%d') for d in gaps[:5]]}"
        )

    features = pd.DataFrame(
        {
            f"{FEATURE_PREFIX}{col}_lag_{lag}": df[col].shift(lag)
            for col in LAG_SOURCE_COLS
            for lag in LAGS
        },
        index=df.index,
    )

    _assert_no_same_day_leakage(features)
    return features


def _assert_no_same_day_leakage(features: pd.DataFrame) -> None:
    """Fail loudly if any raw same-day target/operator column reached the feature frame."""
    leaked = [c for c in FORBIDDEN_SAME_DAY_COLS if c in features.columns]
    if leaked:
        raise ValueError(
            f"LEAKAGE GUARD: raw same-day column(s) {leaked} present in the lag feature "
            "output. metro+emt+carretera+cercanias == total exactly, so a same-day "
            "operator column is perfect leakage of the target."
        )
