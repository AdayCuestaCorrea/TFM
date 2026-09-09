"""Phase 2 feature-engineering tests.

The priority targets, per CLAUDE.md, are the leakage rules: shift-before-rolling ordering
and the same-day operator guard. Those are tested against synthetic series where a wrong
implementation produces a *different number*, not merely a different shape -- a test that
only checks columns exist would pass on leaky code.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import (
    FEATURES_DAILY_FILE,
    RAW_REFERENCE_COLS,
    TARGET_COL,
    build,
    build_features,
    feature_columns,
)
from src.features.calendar_features import (
    DAY_TYPE_CATEGORIES,
    HOLIDAY_TYPE_CATEGORIES,
    add_calendar_features,
    day_type_column_names,
    holiday_type_column_names,
)
from src.features.fourier_features import add_fourier_features, day_index
from src.features.lag_features import (
    FORBIDDEN_SAME_DAY_COLS,
    LAGS,
    OPERATOR_COLS,
    add_lag_features,
    lag_feature_names,
)
from src.features.naming import FEATURE_PREFIX
from src.features.rolling_features import ROLLING_WINDOWS, add_rolling_features
from src.utils.paths import UNIFIED_DAILY_FILE

EXPECTED_ROWS = 1310


# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="session")
def unified() -> pd.DataFrame:
    return pd.read_parquet(UNIFIED_DAILY_FILE)


@pytest.fixture(scope="session")
def features(unified) -> pd.DataFrame:
    return build_features(unified)


def _synth(n: int = 40, start: str = "2024-01-01") -> pd.DataFrame:
    """Small deterministic frame with a non-constant target."""
    dates = pd.date_range(start, periods=n, freq="D")
    total = pd.Series(np.arange(n, dtype="int64") ** 2 + 100)
    return pd.DataFrame(
        {
            "date": dates,
            "total": total,
            "metro": total // 2,
            "emt": total // 4,
            "carretera": total // 8,
            "cercanias": total - (total // 2) - (total // 4) - (total // 8),
            "day_of_week_es": ["lunes"] * n,
            "day_type": pd.Categorical(["laborable"] * n, categories=DAY_TYPE_CATEGORIES),
            "holiday_type": ["none"] * n,
            "holiday_name": ["none"] * n,
        }
    )


# --------------------------------------------------------------------------------------
# Rolling: shift-before-rolling ordering (CLAUDE.md leakage rule 1)
# --------------------------------------------------------------------------------------


def test_unshifted_rolling_is_the_real_leak_and_differs_numerically():
    """The banned form must produce a DIFFERENT number, not just a different shape.

    This is the discriminating case: an unshifted window contains y_t itself.
    (Trailing shift-then-roll vs roll-then-shift are mathematically identical, so they
    cannot serve as the discriminator -- see test_trailing_orderings_are_equivalent.)
    """
    s = pd.Series([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0])
    window = 3
    correct = s.shift(1).rolling(window, min_periods=window).mean()
    leaky = s.rolling(window, min_periods=window).mean()

    assert not correct.equals(leaky)
    assert correct.iloc[3] == pytest.approx(s.iloc[0:3].mean())  # 2.333, excludes y3
    assert leaky.iloc[3] == pytest.approx(s.iloc[1:4].mean())  # 4.667, HAS EATEN y3
    assert correct.iloc[3] != pytest.approx(leaky.iloc[3])


def test_trailing_orderings_are_equivalent():
    """Documents case (2) of the module docstring so nobody 'fixes' a non-bug.

    For a trailing window the two orderings coincide exactly, including NaN placement.
    """
    s = pd.Series([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0])
    for min_periods in (3, 1):
        shift_first = s.shift(1).rolling(3, min_periods=min_periods).mean()
        roll_first = s.rolling(3, min_periods=min_periods).mean().shift(1)
        assert shift_first.equals(roll_first)


def test_centered_orderings_diverge():
    """Documents case (3): the equivalence is a trailing-window property, not a law.

    With center=True, roll-then-shift emits a value at the last index built from future
    observations, where shift-then-roll correctly emits NaN. This is why shift-first is
    mandated as a habit rather than justified case by case.
    """
    s = pd.Series([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0])
    shift_first = s.shift(1).rolling(3, min_periods=3, center=True).mean()
    roll_first = s.rolling(3, min_periods=3, center=True).mean().shift(1)
    assert not shift_first.equals(roll_first)
    assert pd.isna(shift_first.iloc[-1])
    assert not pd.isna(roll_first.iloc[-1])


def test_rolling_features_use_shift_before_rolling():
    """The module must produce the shift-FIRST value, computed independently here."""
    df = _synth(n=40)
    out = add_rolling_features(df)

    window = 7
    col = f"{FEATURE_PREFIX}total_roll_mean_{window}"
    expected = df["total"].shift(1).rolling(window, min_periods=window).mean()
    leaky = df["total"].rolling(window, min_periods=window).mean()

    pd.testing.assert_series_equal(out[col], expected, check_names=False)
    # And prove that assertion was discriminating: the leaky form differs on this data.
    assert not out[col].equals(leaky)


def test_rolling_window_never_sees_same_day_value():
    """Direct leakage check: recompute row t's mean from y_{t-w}..y_{t-1} explicitly."""
    df = _synth(n=40)
    out = add_rolling_features(df)
    window = 7
    t = 20
    manual = df["total"].iloc[t - window : t].mean()  # excludes y_t by construction
    assert out[f"{FEATURE_PREFIX}total_roll_mean_{window}"].iloc[t] == pytest.approx(manual)


def test_rolling_first_valid_index_matches_window():
    df = _synth(n=60)
    out = add_rolling_features(df)
    for window in ROLLING_WINDOWS:
        col = f"{FEATURE_PREFIX}total_roll_mean_{window}"
        # shift(1) costs one row, then the window needs `window` observations.
        assert out[col].first_valid_index() == window


def test_rolling_rejects_gapped_index():
    df = _synth(n=40).drop(index=10).reset_index(drop=True)
    with pytest.raises(ValueError, match="gap"):
        add_rolling_features(df)


# --------------------------------------------------------------------------------------
# Lags: same-day operator leakage guard
# --------------------------------------------------------------------------------------


def test_lag_features_exclude_raw_same_day_operator_columns():
    out = add_lag_features(_synth())
    for col in OPERATOR_COLS:
        assert col not in out.columns, f"raw same-day '{col}' leaked into features"
    assert TARGET_COL not in out.columns


def test_lag_features_column_names_and_count():
    out = add_lag_features(_synth())
    assert list(out.columns) == lag_feature_names()
    assert len(out.columns) == 5 * len(LAGS)  # total + 4 operators


def test_lag_values_are_drawn_from_the_past():
    df = _synth(n=40)
    out = add_lag_features(df)
    for lag in LAGS:
        col = f"{FEATURE_PREFIX}total_lag_{lag}"
        t = 35
        assert out[col].iloc[t] == df["total"].iloc[t - lag]
        assert out[col].iloc[:lag].isna().all()


def test_lag_guard_raises_when_same_day_column_injected(monkeypatch):
    """The guard must actually fire, not just be unreachable in the happy path."""
    from src.features import lag_features as lf

    def leaky(df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame({"metro": df["metro"]}, index=df.index)
        lf._assert_no_same_day_leakage(features)
        return features

    with pytest.raises(ValueError, match="LEAKAGE GUARD"):
        leaky(_synth())


def test_lag_rejects_unsorted_frame():
    df = _synth(n=20).sort_values("date", ascending=False).reset_index(drop=True)
    with pytest.raises(ValueError, match="not sorted"):
        add_lag_features(df)


# --------------------------------------------------------------------------------------
# Fourier: deterministic and periodic
# --------------------------------------------------------------------------------------


def test_fourier_weekly_is_periodic_with_period_7():
    df = _synth(n=40)
    out = add_fourier_features(df)
    sin_col = f"{FEATURE_PREFIX}fourier_weekly_k1_sin"
    cos_col = f"{FEATURE_PREFIX}fourier_weekly_k1_cos"
    for t in range(0, 30):
        assert out[sin_col].iloc[t] == pytest.approx(out[sin_col].iloc[t + 7], abs=1e-12)
        assert out[cos_col].iloc[t] == pytest.approx(out[cos_col].iloc[t + 7], abs=1e-12)


def test_fourier_weekly_is_not_constant():
    """Guards against a degenerate implementation that would trivially pass periodicity."""
    out = add_fourier_features(_synth(n=40))
    assert out[f"{FEATURE_PREFIX}fourier_weekly_k1_sin"].nunique() > 1


def test_fourier_is_deterministic_and_split_invariant():
    """Same dates -> same values, whether computed on the full range or on a slice.

    This is the property that makes computing them pre-split leakage-free.
    """
    df = _synth(n=60)
    full = add_fourier_features(df)
    tail = add_fourier_features(df.iloc[30:].reset_index(drop=True))
    for col in full.columns:
        np.testing.assert_allclose(
            full[col].iloc[30:].to_numpy(), tail[col].to_numpy(), atol=1e-12
        )


def test_fourier_annual_harmonics_present_and_distinct():
    out = add_fourier_features(_synth(n=400))
    k1 = f"{FEATURE_PREFIX}fourier_annual_k1_sin"
    k2 = f"{FEATURE_PREFIX}fourier_annual_k2_sin"
    assert k1 in out.columns and k2 in out.columns
    assert not np.allclose(out[k1], out[k2])


def test_fourier_has_no_nans_anywhere():
    """Deterministic calendar functions need no history, so row 0 is already valid."""
    out = add_fourier_features(_synth(n=40))
    assert out.isna().sum().sum() == 0


def test_fourier_epoch_is_fixed_not_data_dependent():
    """Shifting the window must not change the encoding of a given date."""
    early = add_fourier_features(_synth(n=10, start="2024-01-01"))
    later = add_fourier_features(_synth(n=10, start="2024-01-05"))
    col = f"{FEATURE_PREFIX}fourier_weekly_k1_sin"
    # 2024-01-05 is row 4 of `early` and row 0 of `later`.
    assert early[col].iloc[4] == pytest.approx(later[col].iloc[0], abs=1e-12)


def test_day_index_counts_whole_days():
    idx = day_index(pd.Series(pd.to_datetime(["2023-01-01", "2023-01-02", "2024-01-01"])))
    assert idx.tolist() == [0, 1, 365]


# --------------------------------------------------------------------------------------
# Calendar one-hots, weekend and bridge flags
# --------------------------------------------------------------------------------------


def test_day_type_one_hot_sums_to_one_per_row(features):
    cols = day_type_column_names()
    assert set(cols).issubset(features.columns)
    assert (features[cols].sum(axis=1) == 1).all()


def test_holiday_type_one_hot_sums_to_one_per_row(features):
    cols = holiday_type_column_names()
    assert set(cols).issubset(features.columns)
    assert (features[cols].sum(axis=1) == 1).all()


def test_holiday_none_is_an_explicit_column(features):
    """'none' is a real one-hot column, not an implicit dropped reference level."""
    col = f"{FEATURE_PREFIX}holiday_type_none"
    assert col in features.columns
    assert features[col].sum() > 0


def test_domingo_festivo_column_exists_but_is_all_zero(features):
    col = f"{FEATURE_PREFIX}day_type_domingo_festivo"
    assert col in features.columns
    assert features[col].sum() == 0


def test_is_bridge_day_never_set_on_holiday_or_weekend(features):
    """A puente is by definition a laborable day adjacent to a holiday, not a holiday."""
    bridge = features[f"{FEATURE_PREFIX}is_bridge_day"] == 1
    for bad in ["sabado", "domingo", "festivo", "domingo festivo"]:
        overlap = bridge & (features["day_type"].astype(str) == bad)
        assert overlap.sum() == 0, f"is_bridge_day set on {overlap.sum()} '{bad}' row(s)"
    assert (features.loc[bridge, "day_type"].astype(str) == "laborable").all()


def test_is_bridge_day_detects_a_known_puente():
    """Synthetic: Tue is laborable and adjacent to a Mon festivo -> bridge."""
    n = 5
    df = _synth(n=n)
    day_type = ["laborable"] * n
    day_type[1] = "festivo"  # index 1 is the holiday
    df["day_type"] = pd.Categorical(day_type, categories=DAY_TYPE_CATEGORIES)
    df["holiday_type"] = ["none", "Festivo nacional", "none", "none", "none"]

    out = add_calendar_features(df)
    bridge = out[f"{FEATURE_PREFIX}is_bridge_day"]
    assert bridge.iloc[0] == 1  # day before the holiday
    assert bridge.iloc[1] == 0  # the holiday itself is not a bridge day
    assert bridge.iloc[2] == 1  # day after the holiday
    assert bridge.iloc[3] == 0


def test_is_weekend_matches_day_type(features):
    weekend = features[f"{FEATURE_PREFIX}is_weekend"] == 1
    observed = set(features.loc[weekend, "day_type"].astype(str).unique())
    assert observed <= {"sabado", "domingo", "domingo festivo"}
    assert (features.loc[~weekend, "day_type"].astype(str) != "sabado").all()


def test_calendar_rejects_unmapped_category():
    df = _synth(n=5)
    df["holiday_type"] = ["none", "Festivo autonomico inventado", "none", "none", "none"]
    with pytest.raises(ValueError, match="unmapped value"):
        add_calendar_features(df)


def test_day_of_week_not_one_hot_encoded(features):
    """Documented decision: redundant with weekly Fourier + is_weekend."""
    assert not any(c.startswith(f"{FEATURE_PREFIX}day_of_week") for c in features.columns)
    # ...but preserved raw for EDA.
    assert "day_of_week_es" in features.columns


# --------------------------------------------------------------------------------------
# Orchestrator contract
# --------------------------------------------------------------------------------------


def test_build_features_preserves_row_count(features, unified):
    assert len(features) == len(unified) == EXPECTED_ROWS


def test_build_features_does_not_drop_nan_rows(features):
    """Warm-up NaNs must survive: dropping is a Phase 3 decision."""
    feats = feature_columns(features)
    assert features[feats].isna().any(axis=1).sum() > 0
    assert features["date"].iloc[0] == pd.Timestamp("2023-01-01")


def test_feature_block_excludes_raw_same_day_columns(features):
    feats = feature_columns(features)
    for col in [TARGET_COL, *OPERATOR_COLS]:
        assert col not in feats


def test_raw_reference_columns_preserved(features):
    for col in [TARGET_COL, *RAW_REFERENCE_COLS]:
        assert col in features.columns


def test_warm_up_nans_end_after_longest_window(features):
    """First fully-usable row is at index 28 (28-day lag and 28-day rolling window)."""
    feats = feature_columns(features)
    first_clean = features.index[features[feats].notna().all(axis=1)][0]
    assert first_clean == 28
    assert features.loc[first_clean, "date"] == pd.Timestamp("2023-01-29")


def test_leakage_free_features_have_no_warm_up_nans(features):
    """Calendar and Fourier features must be valid from row 0."""
    safe = [
        c
        for c in feature_columns(features)
        if "lag" not in c and "roll" not in c
    ]
    assert features[safe].isna().sum().sum() == 0


def test_build_round_trips_through_parquet(tmp_path):
    out_path = tmp_path / "features.parquet"
    built = build(output_path=out_path, save=True)
    reloaded = pd.read_parquet(out_path)
    assert len(reloaded) == len(built) == EXPECTED_ROWS
    assert list(reloaded.columns) == list(built.columns)


def test_saved_artifact_matches_expected_location():
    assert FEATURES_DAILY_FILE.name == "features_daily.parquet"
    assert FEATURES_DAILY_FILE.parent.name == "processed"


# ---------------------------------------------------------------------------------------
# Concat-time leakage guards in build_features.
#
# These two guards sat UNREACHABLE across several phases: they scanned the feature block for
# the BARE column name (`metro`, `temperature_2m_mean`), which `feature_columns()` filters out
# by construction, so `leaked` could never be non-empty. Nothing failed as a result -- the
# structural property held throughout -- but the checks promised a guarantee they could not
# deliver, which is worse than no check at all for the next reader.
#
# They now scan for the PREFIXED name, which is what a builder regression would actually
# produce. These tests trip them, so "unreachable" cannot come back silently.
# ---------------------------------------------------------------------------------------


def test_concat_guard_catches_a_prefixed_same_day_operator_column(monkeypatch, unified):
    """A builder emitting `feat_metro` must be stopped at assembly.

    Simulated by making the lag builder return a same-day operator column, which is what a
    regression such as adding lag 0 to LAGS would do.
    """
    import src.features.build_features as bf

    original = bf.add_lag_features
    monkeypatch.setattr(
        bf,
        "add_lag_features",
        lambda df: original(df).assign(feat_metro=df["metro"].to_numpy()),
    )

    with pytest.raises(ValueError, match="LEAKAGE GUARD"):
        bf.build_features(unified)


def test_concat_guard_catches_a_prefixed_same_day_weather_column(monkeypatch, unified):
    """A builder emitting `feat_temperature_2m_mean` (unlagged) must be stopped at assembly."""
    import src.features.build_features as bf

    original = bf.add_weather_features
    monkeypatch.setattr(
        bf,
        "add_weather_features",
        lambda df: original(df).assign(
            feat_temperature_2m_mean=df["temperature_2m_mean"].to_numpy()
        ),
    )

    with pytest.raises(ValueError, match="SAME-DAY WEATHER GUARD"):
        bf.build_features(unified)


def test_bare_same_day_columns_are_expected_outside_the_feature_block(features):
    """The guards must NOT be rewritten to scan bare names against the whole frame.

    `total` and the four operator columns are deliberately present in the output as the
    target and as RAW_REFERENCE_COLS. A guard scanning bare names against `out.columns`
    would raise on every valid frame; one scanning them against the feature block is
    vacuous. Pinning the column contract here so neither mistake gets made later.
    """
    feats = set(feature_columns(features))
    assert not [c for c in FORBIDDEN_SAME_DAY_COLS if c in feats]
    assert all(c in features.columns for c in FORBIDDEN_SAME_DAY_COLS)
