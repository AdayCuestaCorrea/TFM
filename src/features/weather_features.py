"""Lagged weather features — promoted into the model matrix, same-day values excluded.

WHY LAGGED-ONLY. THIS IS A DELIBERATE CONSERVATIVE CHOICE, NOT AN OVERSIGHT.

The weather columns in this dataset are *observed* values, not forecasts. Feeding
same-day weather to a model that predicts same-day demand implicitly assumes a perfect
weather forecast is available at prediction time. It is not. A model built that way
reports performance it could never reproduce in deployment, and the inflation is not
uniform -- it is largest exactly on the anomalous days (storms, heatwaves) where the
weather signal carries the most information and where a real forecast is least reliable.

So the reference model uses lagged weather only: values genuinely known when the forecast
is issued. The same-day variant is not discarded as a question, it is deferred: it belongs
in the evaluation phase as an explicitly labelled **sensitivity analysis** ("how much of
the achievable gain depends on forecast quality?"), never as the headline model. Report it
as an upper bound under a perfect-forecast idealization, and say so.

`_assert_no_same_day_weather` enforces this mechanically, mirroring the same-day operator
guard in lag_features.py: no `feat_`-prefixed column here may carry a bare weather column
name.

WHY LAGS [1, 2, 3, 7] AND NOT THE FULL [1, 2, 3, 7, 14, 21, 28] USED FOR `total`.

The two series have different memory structures. Demand autocorrelation at lag 28 is
strong because it is *calendar-driven* -- 28 days is exactly four weeks, so lag 28 lands on
the same weekday and largely the same weekly regime. Atmospheric autocorrelation has no
such mechanism: synoptic weather patterns decay over roughly 3-7 days, and a temperature
28 days ago carries essentially only the seasonal signal, which the Fourier annual terms
already represent far more cleanly. Adding lags 14/21/28 for 21 weather variables would
add 63 near-redundant columns against 1310 observations, inflating dimensionality for
little information. `roll_mean_7` is retained per variable to capture the slower
"what kind of week has it been" effect that individual daily lags miss.
"""

import pandas as pd

from src.features.naming import FEATURE_PREFIX

# Declared explicitly rather than inferred, so a renamed or dropped upstream column fails
# loudly here instead of silently shrinking the feature matrix.
WEATHER_COLS: list[str] = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "apparent_temperature_mean",
    "apparent_temperature_max",
    "apparent_temperature_min",
    "precipitation_sum",
    "relative_humidity_2m_mean",
    "relative_humidity_2m_max",
    "relative_humidity_2m_min",
    "wind_speed_10m_max",
    "wind_speed_10m_mean",
    "wind_speed_10m_min",
    "pressure_msl_min",
    "pressure_msl_max",
    "pressure_msl_mean",
    "shortwave_radiation_sum",
    "precipitation_hours",
    "rain_sum",
    "snowfall_sum",
    "weather_code",
]

WEATHER_LAGS: list[int] = [1, 2, 3, 7]
WEATHER_ROLL_WINDOW = 7


def weather_feature_names() -> list[str]:
    """The exact column names this module produces, in order."""
    names: list[str] = []
    for col in WEATHER_COLS:
        names += [f"{FEATURE_PREFIX}{col}_lag_{lag}" for lag in WEATHER_LAGS]
        names.append(f"{FEATURE_PREFIX}{col}_roll_mean_{WEATHER_ROLL_WINDOW}")
    return names


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build lagged and rolling weather features.

    Args:
        df: Frame sorted ascending by 'date' with a gap-free daily index, containing
            every column in WEATHER_COLS.

    Returns:
        A features-only DataFrame sharing `df`'s index: for each weather variable, one
        column per lag in WEATHER_LAGS plus a 7-day trailing mean. All `feat_`-prefixed.

    Raises:
        ValueError: if weather columns are missing, the index is unsorted or gapped, or
            the same-day guard detects an unlagged weather value in the output.
    """
    missing = [c for c in ["date", *WEATHER_COLS] if c not in df.columns]
    if missing:
        raise ValueError(f"weather_features: missing required column(s) {missing}")

    if not df["date"].is_monotonic_increasing:
        raise ValueError(
            "weather_features: frame is not sorted ascending by date; .shift() would draw "
            "values from the wrong rows."
        )
    gaps = pd.date_range(df["date"].min(), df["date"].max(), freq="D").difference(
        pd.DatetimeIndex(df["date"])
    )
    if len(gaps) > 0:
        raise ValueError(
            f"weather_features: date index has {len(gaps)} gap(s), so positional .shift() "
            f"does not equal a calendar-day lag. First gaps: "
            f"{[d.strftime('%Y-%m-%d') for d in gaps[:5]]}"
        )

    features: dict[str, pd.Series] = {}
    for col in WEATHER_COLS:
        series = df[col].astype("float64")
        for lag in WEATHER_LAGS:
            features[f"{FEATURE_PREFIX}{col}_lag_{lag}"] = series.shift(lag)
        # shift(1) FIRST, consistent with rolling_features.py: the window at row t must
        # span t-7..t-1 and cannot be allowed to touch today's observation.
        features[f"{FEATURE_PREFIX}{col}_roll_mean_{WEATHER_ROLL_WINDOW}"] = (
            series.shift(1)
            .rolling(WEATHER_ROLL_WINDOW, min_periods=WEATHER_ROLL_WINDOW)
            .mean()
        )

    out = pd.DataFrame(features, index=df.index)
    _assert_no_same_day_weather(out)
    return out


def _assert_no_same_day_weather(features: pd.DataFrame) -> None:
    """Fail loudly if any unlagged weather value reached the feature frame.

    Checks both the bare column name and the `feat_`-prefixed bare name: either would mean
    a same-day observation entered the model matrix under the perfect-forecast idealization
    this module exists to avoid.
    """
    leaked = [
        c
        for c in features.columns
        if c in WEATHER_COLS or c in {f"{FEATURE_PREFIX}{w}" for w in WEATHER_COLS}
    ]
    if leaked:
        raise ValueError(
            f"SAME-DAY WEATHER GUARD: unlagged weather column(s) {leaked} present in the "
            "feature output. Same-day weather is observed data, not a forecast; using it "
            "assumes perfect foreknowledge and inflates apparent performance."
        )
