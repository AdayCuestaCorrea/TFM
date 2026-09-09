"""Phase 1 ingestion tests.

Two distinct classes of test live here:
  - Contract tests against the REAL files: canonical columns, dtypes, integrity of the
    unified dataset. These are what catch a bad data refresh.
  - Negative tests against small IN-MEMORY fixtures: they prove the validation actually
    raises. Injecting a synthetic gap into the real files is neither possible nor
    desirable, so the pure `unify_frames` entry point is exercised directly.
"""

import pandas as pd
import pytest

from src.ingestion.load_calendar import DAY_TYPE_CATEGORIES, load_calendar
from src.ingestion.load_crtm import CANONICAL_COLUMNS as CRTM_COLUMNS
from src.ingestion.load_crtm import EXPECTED_RAW_COLUMNS, load_crtm
from src.ingestion.load_weather import load_weather
from src.ingestion.unify import NO_HOLIDAY, unify_frames

CALENDAR_COLUMNS = ["date", "day_of_week_es", "day_type", "holiday_type", "holiday_name"]


# --------------------------------------------------------------------------------------
# Fixtures over the real sources (loaded once per session: the Excel read is not cheap)
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="session")
def crtm() -> pd.DataFrame:
    return load_crtm()


@pytest.fixture(scope="session")
def weather() -> pd.DataFrame:
    return load_weather()


@pytest.fixture(scope="session")
def calendar() -> pd.DataFrame:
    return load_calendar()


@pytest.fixture(scope="session")
def unified(crtm, weather, calendar) -> pd.DataFrame:
    return unify_frames(crtm, weather, calendar)


# --------------------------------------------------------------------------------------
# Synthetic in-memory fixtures for the negative tests
# --------------------------------------------------------------------------------------

_SYNTH_DATES = pd.date_range("2024-01-01", periods=5, freq="D")


def _synth_crtm(dates: pd.DatetimeIndex = _SYNTH_DATES) -> pd.DataFrame:
    n = len(dates)
    return pd.DataFrame(
        {
            "date": dates,
            "metro": range(100, 100 + n),
            "emt": range(200, 200 + n),
            "carretera": range(300, 300 + n),
            "cercanias": range(400, 400 + n),
            "total": range(1000, 1000 + n),
        }
    ).astype({c: "int64" for c in ["metro", "emt", "carretera", "cercanias", "total"]})


def _synth_weather(dates: pd.DatetimeIndex = _SYNTH_DATES) -> pd.DataFrame:
    return pd.DataFrame({"date": dates, "temperature_2m_mean": [8.5] * len(dates)})


def _synth_calendar(dates: pd.DatetimeIndex = _SYNTH_DATES) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": dates,
            "day_of_week_es": ["lunes"] * len(dates),
            "day_type": pd.Categorical(
                ["laborable"] * len(dates), categories=DAY_TYPE_CATEGORIES
            ),
            "holiday_type": [None] * len(dates),
            "holiday_name": [None] * len(dates),
        }
    )


# --------------------------------------------------------------------------------------
# Canonical schema / dtype contracts
# --------------------------------------------------------------------------------------


def test_crtm_canonical_columns(crtm):
    assert list(crtm.columns) == CRTM_COLUMNS


def test_crtm_dtypes(crtm):
    assert crtm["date"].dtype == "datetime64[ns]"
    for col in ["metro", "emt", "carretera", "cercanias", "total"]:
        assert crtm[col].dtype == "int64", f"{col} is {crtm[col].dtype}, expected int64"


def test_crtm_date_has_no_time_component(crtm):
    assert (crtm["date"] == crtm["date"].dt.normalize()).all()


def test_crtm_sorted_ascending(crtm):
    assert crtm["date"].is_monotonic_increasing


def test_crtm_operator_columns_sum_to_total(crtm):
    # Not a schema check but a cheap sanity check on the mapping: if metro/emt/carretera/
    # cercanias were mis-mapped, their sum would stop reconciling with 'total'.
    operators = crtm[["metro", "emt", "carretera", "cercanias"]].sum(axis=1)
    assert (operators == crtm["total"]).all()


def test_crtm_rounds_excel_formula_noise(crtm):
    """2026-07-07 arrives as carretera=847904.00000001 (float64 formula round-off).

    It must be rounded to the integer passenger count, not truncated to ...903.
    """
    row = crtm.loc[crtm["date"] == pd.Timestamp("2026-07-07")]
    assert len(row) == 1
    assert row["carretera"].iloc[0] == 847904
    assert row["total"].iloc[0] == 4748815


def test_crtm_genuinely_fractional_value_still_raises(tmp_path):
    """The rounding tolerance must not swallow a real fractional value."""
    frame = pd.DataFrame(
        [
            [None, pd.Timestamp("2023-01-01"), 1.0, 2.0, 3.5, 4.0, 10.5, None, None],
        ]
    )
    frame.columns = EXPECTED_RAW_COLUMNS
    path = tmp_path / "crtm.xlsx"
    # Written with header=1 semantics: one spacer row above the header row.
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame([[None] * len(EXPECTED_RAW_COLUMNS)]).to_excel(
            writer, sheet_name="diaria", index=False, header=False, startrow=0
        )
        frame.to_excel(writer, sheet_name="diaria", index=False, startrow=1)

    with pytest.raises(ValueError, match="genuinely fractional"):
        load_crtm(path)


def test_weather_has_canonical_date_column(weather):
    assert "date" in weather.columns
    assert "time" not in weather.columns
    assert weather["date"].dtype == "datetime64[ns]"


def test_weather_unit_suffixes_stripped(weather):
    # No column may retain a parenthesised unit annotation, and names must be snake_case.
    for col in weather.columns:
        assert "(" not in col and ")" not in col, f"unit suffix survived in {col!r}"
        assert col == col.lower(), f"{col!r} is not lowercase"
        assert " " not in col, f"{col!r} contains a space"
    assert "temperature_2m_mean" in weather.columns
    assert "shortwave_radiation_sum" in weather.columns
    assert "weather_code" in weather.columns


def test_calendar_canonical_columns(calendar):
    assert list(calendar.columns) == CALENDAR_COLUMNS


def test_calendar_dtypes(calendar):
    assert calendar["date"].dtype == "datetime64[ns]"
    assert isinstance(calendar["day_type"].dtype, pd.CategoricalDtype)


def test_calendar_day_of_week_is_unaccented(calendar):
    # The source uses ASCII 'miercoles'/'sabado'. Guard the exact form, since any future
    # join-by-name logic must match it.
    observed = set(calendar["day_of_week_es"].unique())
    assert observed == {
        "lunes",
        "martes",
        "miercoles",
        "jueves",
        "viernes",
        "sabado",
        "domingo",
    }


# --------------------------------------------------------------------------------------
# The 'domingo festivo' category requirement
# --------------------------------------------------------------------------------------


def test_day_type_includes_domingo_festivo_category(calendar):
    """'domingo festivo' must be a declared category even with zero observations."""
    assert "domingo festivo" in list(calendar["day_type"].cat.categories)
    # Precondition for the above mattering: it genuinely has no rows today.
    assert (calendar["day_type"] == "domingo festivo").sum() == 0


def test_day_type_has_no_nulls_from_failed_mapping(calendar):
    # A value outside the declared categories would surface here as NaN.
    assert calendar["day_type"].isna().sum() == 0


def test_day_type_categories_are_exactly_as_declared(calendar):
    assert list(calendar["day_type"].cat.categories) == DAY_TYPE_CATEGORIES


def test_domingo_festivo_survives_unification():
    """A future row carrying 'domingo festivo' must reach the unified frame intact."""
    cal = _synth_calendar()
    cal.loc[0, "day_type"] = "domingo festivo"

    unified = unify_frames(_synth_crtm(), _synth_weather(), cal)

    assert unified.loc[0, "day_type"] == "domingo festivo"
    assert "domingo festivo" in list(unified["day_type"].cat.categories)


def test_unmapped_day_type_raises(tmp_path):
    """An unexpected category must raise, not be silently coerced to NaN."""
    csv = tmp_path / "cal.csv"
    csv.write_text(
        "Dia;Dia_semana;laborable / festivo / domingo festivo;Tipo de Festivo;Festividad\n"
        "01/01/2023;domingo;puente;;\n",
        encoding="utf-8-sig",
    )
    with pytest.raises(ValueError, match="unmapped value"):
        load_calendar(csv)


# --------------------------------------------------------------------------------------
# Unified dataset integrity
# --------------------------------------------------------------------------------------


def test_unified_has_zero_nulls(unified):
    null_counts = unified.isna().sum()
    assert null_counts.sum() == 0, f"nulls found: {null_counts[null_counts > 0].to_dict()}"


def test_unified_has_zero_duplicate_dates(unified):
    assert unified["date"].duplicated().sum() == 0


def test_unified_has_zero_date_gaps(unified):
    expected = pd.date_range(unified["date"].min(), unified["date"].max(), freq="D")
    assert len(unified) == len(expected)
    assert pd.DatetimeIndex(unified["date"]).equals(expected)


def test_unified_preserves_operator_columns(unified):
    # CLAUDE.md: the four operator columns must survive ingestion for secondary EDA,
    # even though 'total' is the sole modeling target.
    for col in ["metro", "emt", "carretera", "cercanias", "total"]:
        assert col in unified.columns


def test_unified_row_count_matches_sources(unified, crtm, weather, calendar):
    assert len(unified) == len(crtm) == len(weather) == len(calendar)


def test_unified_holiday_sentinel_replaces_nan(unified):
    assert (unified["holiday_type"] == NO_HOLIDAY).sum() > 0
    assert unified["holiday_name"].isna().sum() == 0


# --------------------------------------------------------------------------------------
# Negative tests: injected gaps must raise
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("source", ["crtm", "weather", "calendar"])
def test_gap_in_any_source_raises(source):
    """Dropping a date from one source must raise, naming the missing date."""
    frames = {
        "crtm": _synth_crtm(),
        "weather": _synth_weather(),
        "calendar": _synth_calendar(),
    }
    # Drop the middle date so the gap is interior, not a range-boundary truncation.
    frames[source] = frames[source].drop(index=2).reset_index(drop=True)

    with pytest.raises(ValueError, match="Inner join dropped rows"):
        unify_frames(frames["crtm"], frames["weather"], frames["calendar"])


def test_gap_error_names_the_missing_dates():
    frames = (_synth_crtm(), _synth_weather(), _synth_calendar().drop(index=2))
    with pytest.raises(ValueError) as exc:
        unify_frames(*frames)
    assert "2024-01-03" in str(exc.value)


def test_common_gap_in_all_sources_raises():
    """A hole shared by all three sources survives the join and must still raise."""
    gapped = _SYNTH_DATES.delete(2)
    with pytest.raises(ValueError, match="missing calendar day"):
        unify_frames(_synth_crtm(gapped), _synth_weather(gapped), _synth_calendar(gapped))


def test_duplicate_date_in_source_raises():
    dupe = pd.concat([_synth_crtm(), _synth_crtm().head(1)], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate date"):
        unify_frames(dupe, _synth_weather(), _synth_calendar())


def test_clean_synthetic_sources_unify_without_error():
    """Control for the negative tests: the fixtures themselves are valid."""
    unified = unify_frames(_synth_crtm(), _synth_weather(), _synth_calendar())
    assert len(unified) == len(_SYNTH_DATES)
    assert unified.isna().sum().sum() == 0
