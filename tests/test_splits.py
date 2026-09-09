"""Tests for the chronological split and the warm-up NaN policy."""

import pandas as pd
import pytest

from src.utils.splits import (
    WARMUP_ROWS,
    apply_warmup_policy,
    chronological_split,
    split_masks,
)

EXPECTED_ROWS = 1310


@pytest.fixture(scope="session")
def dates() -> pd.Series:
    return pd.Series(pd.date_range("2023-01-01", "2026-08-02", freq="D"))


@pytest.fixture(scope="session")
def bounds(dates):
    return chronological_split(dates)


def _frame(dates: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"date": dates, "total": range(len(dates))})


# --------------------------------------------------------------------------------------
# Boundary correctness and stability
# --------------------------------------------------------------------------------------


def test_split_covers_every_row_exactly_once(bounds, dates):
    assert bounds.n_train + bounds.n_val + bounds.n_test == len(dates) == EXPECTED_ROWS


def test_split_is_chronological_and_non_overlapping(bounds):
    assert bounds.train_start < bounds.train_end < bounds.val_start
    assert bounds.val_start < bounds.val_end < bounds.test_start
    assert bounds.test_start < bounds.test_end
    # Adjacent splits must be exactly one day apart: no gap, no overlap.
    assert bounds.val_start - bounds.train_end == pd.Timedelta(days=1)
    assert bounds.test_start - bounds.val_end == pd.Timedelta(days=1)


def test_split_boundaries_are_exact_and_stable(bounds):
    """Pinned values. If these change, every metric in Phases 3-5 becomes incomparable."""
    assert bounds.as_dict() == {
        "train_start": "2023-01-01",
        "train_end": "2025-07-05",
        "val_start": "2025-07-06",
        "val_end": "2026-01-17",
        "test_start": "2026-01-18",
        "test_end": "2026-08-02",
    }
    assert (bounds.n_train, bounds.n_val, bounds.n_test) == (917, 196, 197)


def test_split_fractions_are_approximately_70_15_15(bounds):
    total = bounds.n_train + bounds.n_val + bounds.n_test
    assert bounds.n_train / total == pytest.approx(0.70, abs=0.005)
    assert bounds.n_val / total == pytest.approx(0.15, abs=0.005)
    assert bounds.n_test / total == pytest.approx(0.15, abs=0.005)


def test_floating_point_floor_does_not_lose_a_day():
    """0.70 * 1310 == 916.9999999999999 in binary float; floor must still give 917."""
    b = chronological_split(pd.Series(pd.date_range("2023-01-01", periods=1310, freq="D")))
    assert b.n_train == 917


def test_split_is_deterministic(dates):
    assert chronological_split(dates).as_dict() == chronological_split(dates).as_dict()


def test_masks_partition_the_frame(dates, bounds):
    df = _frame(dates)
    masks = split_masks(df, bounds)
    stacked = masks["train"].astype(int) + masks["val"].astype(int) + masks["test"].astype(int)
    assert (stacked == 1).all()  # every row in exactly one split


def test_split_rejects_gapped_index():
    gapped = pd.Series(pd.date_range("2023-01-01", periods=100, freq="D")).drop(index=50)
    with pytest.raises(ValueError, match="gap"):
        chronological_split(gapped)


def test_split_rejects_unsorted_index(dates):
    with pytest.raises(ValueError, match="sorted"):
        chronological_split(dates[::-1])


def test_split_rejects_duplicates(dates):
    with pytest.raises(ValueError, match="duplicate"):
        chronological_split(pd.concat([dates, dates.iloc[:1]]).sort_values())


def test_split_rejects_fractions_leaving_no_test(dates):
    with pytest.raises(ValueError, match="no room"):
        chronological_split(dates, train_fraction=0.9, val_fraction=0.1)


# --------------------------------------------------------------------------------------
# Warm-up policy
# --------------------------------------------------------------------------------------


def test_warmup_drops_exactly_the_leading_rows(dates, bounds):
    df = _frame(dates)
    trimmed, dropped = apply_warmup_policy(df, bounds)
    assert dropped == WARMUP_ROWS
    assert len(trimmed) == len(df) - WARMUP_ROWS
    assert trimmed["date"].iloc[0] == pd.Timestamp("2023-01-29")


def test_warmup_rows_all_come_from_the_training_block(dates, bounds):
    df = _frame(dates)
    removed = df["date"].iloc[:WARMUP_ROWS]
    assert (removed < bounds.val_start).all()
    assert (removed >= bounds.train_start).all()


def test_warmup_never_truncates_val_or_test(dates, bounds):
    df = _frame(dates)
    trimmed, _ = apply_warmup_policy(df, bounds)
    masks_before = split_masks(df, bounds)
    masks_after = split_masks(trimmed, bounds)
    assert masks_after["val"].sum() == masks_before["val"].sum() == bounds.n_val
    assert masks_after["test"].sum() == masks_before["test"].sum() == bounds.n_test
    # Only training shrinks, and by exactly the warm-up count.
    assert masks_after["train"].sum() == bounds.n_train - WARMUP_ROWS


def test_warmup_raises_if_it_would_reach_the_validation_split():
    """The guard must fire rather than silently shortening the evaluation window."""
    short = pd.Series(pd.date_range("2023-01-01", periods=40, freq="D"))
    bounds = chronological_split(short)  # train ends around row 28
    with pytest.raises(ValueError, match="inside the validation/test period"):
        apply_warmup_policy(_frame(short), bounds, warmup_rows=35)


def test_warmup_zero_is_a_noop(dates, bounds):
    df = _frame(dates)
    trimmed, dropped = apply_warmup_policy(df, bounds, warmup_rows=0)
    assert dropped == 0
    assert len(trimmed) == len(df)


def test_warmup_rejects_unsorted_frame(dates, bounds):
    df = _frame(dates).iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="sorted"):
        apply_warmup_policy(df, bounds)


def test_trimmed_feature_table_has_no_nans_in_training(bounds):
    """End-to-end: after the warm-up trim, no engineered feature is NaN in training."""
    from src.features.build_features import FEATURES_DAILY_FILE, feature_columns

    df = pd.read_parquet(FEATURES_DAILY_FILE).sort_values("date").reset_index(drop=True)
    trimmed, _ = apply_warmup_policy(df, bounds)
    train = trimmed[split_masks(trimmed, bounds)["train"]]
    assert train[feature_columns(trimmed)].isna().sum().sum() == 0
