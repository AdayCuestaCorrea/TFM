"""Rolling mean and standard deviation of the target.

ORDERING IS THE WHOLE POINT OF THIS MODULE.

    MANDATED:  total.shift(1).rolling(w).mean()
    BANNED:    total.rolling(w).mean()               <- the real leak
    EQUIVALENT HERE, STILL BANNED BY CONVENTION:
               total.rolling(w).mean().shift(1)

A precise statement of the three cases, because a vague one invites the wrong fix later:

1. The UNSHIFTED window is the actual leak, and it is not subtle. `rolling(w).mean()` at
   row t averages y_{t-w+1} .. y_t, so the feature contains the very value being predicted.
   Verified numerically: on [1,2,4,8,...] with w=3, the unshifted mean at index 3 is 4.667
   (it has eaten y_3=8) where the correct value is 2.333 = mean(y_0,y_1,y_2).

2. For a TRAILING window, shift-then-roll and roll-then-shift are mathematically
   IDENTICAL -- verified empirically for min_periods=w, min_periods=1, and for a source
   series containing NaNs. Both yield mean(y_{t-w} .. y_{t-1}) at row t with identical NaN
   placement. Do not let a future session "discover" that roll-then-shift is a live bug in
   this codebase and go hunting for a defect that is not there.

3. The equivalence in (2) is a property of trailing windows, NOT a general law. With
   `center=True` the two orderings genuinely diverge: roll-then-shift emits a value at the
   final index built from future observations, where shift-then-roll correctly emits NaN.

The shift-first form is mandated anyway, because its correctness does not depend on
noticing which of (2) or (3) applies to the aggregation at hand. `shift(1)` yields a series
whose row t holds y_{t-1}, so the window at row t physically cannot reach y_t -- true for
any window, any aggregation, centered or not. That is a habit worth keeping even where the
alternative happens to be safe.

test_features.py pins case (1) against the correct output, and pins case (3) as the
demonstration that ordering can matter.
"""

import pandas as pd

from src.features.naming import FEATURE_PREFIX

ROLLING_WINDOWS: list[int] = [7, 28]
TARGET_COL = "total"

# ddof=1 (pandas default): the window is treated as a sample, not the population. Stated
# explicitly because a silent ddof change would shift every std feature slightly.
STD_DDOF = 1


def rolling_feature_names() -> list[str]:
    """The exact column names this module produces, in order."""
    names: list[str] = []
    for window in ROLLING_WINDOWS:
        names.append(f"{FEATURE_PREFIX}{TARGET_COL}_roll_mean_{window}")
        names.append(f"{FEATURE_PREFIX}{TARGET_COL}_roll_std_{window}")
    return names


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build shift-first rolling mean/std features for the target.

    Args:
        df: Unified daily frame, sorted ascending by 'date' with a gap-free daily index.
            Must contain 'date' and TARGET_COL.

    Returns:
        A features-only DataFrame sharing `df`'s index with a mean and std column per
        window. The first `max(ROLLING_WINDOWS)` rows are NaN by construction
        (min_periods equals the window, so partial windows are not emitted -- a partial
        mean over 3 days is not a 28-day mean and must not masquerade as one).

    Raises:
        ValueError: if required columns are missing or the frame is not chronologically
            sorted with a gap-free daily index.
    """
    missing = [c for c in ["date", TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"rolling_features: missing required column(s) {missing}")

    if not df["date"].is_monotonic_increasing:
        raise ValueError(
            "rolling_features: frame is not sorted ascending by date; the rolling window "
            "would span the wrong rows."
        )
    gaps = pd.date_range(df["date"].min(), df["date"].max(), freq="D").difference(
        pd.DatetimeIndex(df["date"])
    )
    if len(gaps) > 0:
        raise ValueError(
            f"rolling_features: date index has {len(gaps)} gap(s), so a positional window "
            f"does not equal a calendar-day window. First gaps: "
            f"{[d.strftime('%Y-%m-%d') for d in gaps[:5]]}"
        )

    # shift(1) FIRST -- see module docstring. Never reorder these two calls.
    shifted = df[TARGET_COL].shift(1)

    features: dict[str, pd.Series] = {}
    for window in ROLLING_WINDOWS:
        roll = shifted.rolling(window=window, min_periods=window)
        features[f"{FEATURE_PREFIX}{TARGET_COL}_roll_mean_{window}"] = roll.mean()
        features[f"{FEATURE_PREFIX}{TARGET_COL}_roll_std_{window}"] = roll.std(ddof=STD_DDOF)

    return pd.DataFrame(features, index=df.index)
