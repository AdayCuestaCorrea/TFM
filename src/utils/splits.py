"""Chronological train/validation/test split and the warm-up NaN policy.

SINGLE SOURCE OF TRUTH. Every model in Phases 3-5 must obtain its boundaries from
`chronological_split()` and never recompute them locally. If the LSTM and the baselines
disagree by even one day about where the test set starts, their metrics are not comparable
and the whole comparison table is meaningless.

Splits are strictly chronological, per CLAUDE.md Leakage Rule 2: train is the earliest
block, then validation, then test. No shuffling, no random assignment.
"""

from dataclasses import dataclass, field

import pandas as pd

TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
# Test takes the remainder, so the three always sum to exactly the dataset length.

# First row index at which every engineered feature is non-NaN, set by the longest
# lag/rolling window in Phase 2 (28-day lag and 28-day rolling mean).
WARMUP_ROWS = 28


@dataclass(frozen=True)
class SplitBoundaries:
    """Inclusive calendar boundaries of each chronological split."""

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_train: int = field(default=0)
    n_val: int = field(default=0)
    n_test: int = field(default=0)

    def as_dict(self) -> dict[str, str]:
        return {
            "train_start": f"{self.train_start:%Y-%m-%d}",
            "train_end": f"{self.train_end:%Y-%m-%d}",
            "val_start": f"{self.val_start:%Y-%m-%d}",
            "val_end": f"{self.val_end:%Y-%m-%d}",
            "test_start": f"{self.test_start:%Y-%m-%d}",
            "test_end": f"{self.test_end:%Y-%m-%d}",
        }

    def describe(self) -> str:
        total = self.n_train + self.n_val + self.n_test
        return (
            f"train {self.train_start:%Y-%m-%d} -> {self.train_end:%Y-%m-%d}  "
            f"({self.n_train:>4} d, {self.n_train / total:.1%})\n"
            f"val   {self.val_start:%Y-%m-%d} -> {self.val_end:%Y-%m-%d}  "
            f"({self.n_val:>4} d, {self.n_val / total:.1%})\n"
            f"test  {self.test_start:%Y-%m-%d} -> {self.test_end:%Y-%m-%d}  "
            f"({self.n_test:>4} d, {self.n_test / total:.1%})"
        )


def chronological_split(
    dates: pd.Series | pd.DatetimeIndex,
    train_fraction: float = TRAIN_FRACTION,
    val_fraction: float = VAL_FRACTION,
) -> SplitBoundaries:
    """Compute inclusive calendar boundaries for a 70/15/15 chronological split.

    Rounding is resolved on CUMULATIVE fractions rather than per-split ones. With
    n=1310, 0.15*n = 196.5: rounding each split independently would either lose or
    duplicate a day at the boundary. Taking floor of the cumulative position instead
    (floor(0.70n), floor(0.85n)) guarantees the three blocks tile the range exactly with
    no gap and no overlap, and gives the leftover day to the test set.

    Args:
        dates: The complete daily date index, ascending and gap-free.
        train_fraction: Share of days assigned to training.
        val_fraction: Share of days assigned to validation.

    Returns:
        SplitBoundaries with inclusive start/end dates and row counts per split.

    Raises:
        ValueError: if `dates` is empty, unsorted, gapped, or contains duplicates, or if
            the fractions leave any split empty.
    """
    index = pd.DatetimeIndex(pd.Series(dates).reset_index(drop=True))

    if len(index) == 0:
        raise ValueError("chronological_split: empty date index")
    if not index.is_monotonic_increasing:
        raise ValueError("chronological_split: dates must be sorted ascending")
    if index.has_duplicates:
        raise ValueError("chronological_split: dates contain duplicates")
    expected = pd.date_range(index.min(), index.max(), freq="D")
    if len(expected) != len(index):
        raise ValueError(
            f"chronological_split: date index has {len(expected) - len(index)} calendar "
            "gap(s); split fractions would not correspond to equal spans of time."
        )
    if not 0 < train_fraction < 1 or not 0 < val_fraction < 1:
        raise ValueError("chronological_split: fractions must lie in (0, 1)")
    if train_fraction + val_fraction >= 1:
        raise ValueError(
            "chronological_split: train + val fractions leave no room for a test set"
        )

    n = len(index)
    # EPS neutralises binary floating-point representation error before flooring.
    # Without it, 0.70 * 1310 evaluates to 916.9999999999999 and floors to 916, silently
    # delivering a split one day short of the documented floor(0.70n) = 917.
    EPS = 1e-9
    train_end_pos = int(train_fraction * n + EPS)  # exclusive upper bound
    val_end_pos = int((train_fraction + val_fraction) * n + EPS)  # exclusive upper bound

    if train_end_pos < 1 or val_end_pos <= train_end_pos or val_end_pos >= n:
        raise ValueError(
            f"chronological_split: fractions produce an empty split for n={n} "
            f"(train={train_end_pos}, val={val_end_pos - train_end_pos}, "
            f"test={n - val_end_pos})"
        )

    return SplitBoundaries(
        train_start=index[0],
        train_end=index[train_end_pos - 1],
        val_start=index[train_end_pos],
        val_end=index[val_end_pos - 1],
        test_start=index[val_end_pos],
        test_end=index[-1],
        n_train=train_end_pos,
        n_val=val_end_pos - train_end_pos,
        n_test=n - val_end_pos,
    )


def split_masks(
    df: pd.DataFrame, bounds: SplitBoundaries, date_col: str = "date"
) -> dict[str, pd.Series]:
    """Boolean masks selecting each split from `df`."""
    dates = df[date_col]
    return {
        "train": (dates >= bounds.train_start) & (dates <= bounds.train_end),
        "val": (dates >= bounds.val_start) & (dates <= bounds.val_end),
        "test": (dates >= bounds.test_start) & (dates <= bounds.test_end),
    }


def apply_warmup_policy(
    df: pd.DataFrame,
    bounds: SplitBoundaries,
    warmup_rows: int = WARMUP_ROWS,
    date_col: str = "date",
) -> tuple[pd.DataFrame, int]:
    """Drop warm-up rows, but ONLY from the training block.

    The first `warmup_rows` rows carry NaNs from the longest lag/rolling window. They are
    removed so models are not fed incomplete feature vectors -- but removal is confined to
    the training block by design. If the warm-up ever reached into validation or test, the
    quiet outcome would be a silently shortened evaluation period and metrics that are not
    comparable across models or across re-runs. That must be an error, not a shrug, so it
    raises instead.

    Args:
        df: Feature table sorted ascending by date.
        bounds: Boundaries from `chronological_split`.
        warmup_rows: Number of leading rows to drop.
        date_col: Name of the date column.

    Returns:
        (frame with warm-up rows removed, number of rows dropped).

    Raises:
        ValueError: if the frame is unsorted, or if the warm-up window would extend into
            the validation or test split.
    """
    if not df[date_col].is_monotonic_increasing:
        raise ValueError("apply_warmup_policy: frame must be sorted ascending by date")
    if warmup_rows < 0:
        raise ValueError("apply_warmup_policy: warmup_rows must be non-negative")
    if warmup_rows == 0:
        return df.copy(), 0

    if warmup_rows > len(df):
        raise ValueError(
            f"apply_warmup_policy: warm-up ({warmup_rows}) exceeds frame length ({len(df)})"
        )

    # The last row that would be dropped. Assert it sits strictly inside training rather
    # than assuming it -- with the current date range it does, but a shorter refresh or a
    # longer lag window would change that silently.
    last_warmup_date = df[date_col].iloc[warmup_rows - 1]
    if last_warmup_date >= bounds.val_start:
        raise ValueError(
            f"apply_warmup_policy: warm-up window ends at "
            f"{last_warmup_date:%Y-%m-%d}, which falls inside the validation/test period "
            f"(val starts {bounds.val_start:%Y-%m-%d}). Dropping it would silently "
            "truncate the evaluation set. Shorten the lag windows or extend the history."
        )

    trimmed = df.iloc[warmup_rows:].reset_index(drop=True)
    return trimmed, warmup_rows
