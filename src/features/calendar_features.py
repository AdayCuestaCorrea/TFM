"""Calendar-derived features: day-type / holiday-type one-hots, weekend and bridge flags.

All features here are functions of the calendar table only -- never of `total` -- so they
carry no leakage risk and are safe to compute on the full date range before splitting, for
the same reason set out in fourier_features.py. That includes `is_bridge_day`, which reads
the *neighbouring calendar rows* (`day_type` at t-1 and t+1). Reading t+1 is a forward look
at the calendar, not at the target: Spanish public holidays are published a year ahead, so
tomorrow's holiday status is genuinely known when predicting today. It would be leakage
only if the neighbour lookup touched observed demand, which it does not.

DECISION -- `day_of_week_es` is deliberately NOT one-hot encoded.
This is a choice, not an oversight. Seven day-of-week dummies would be near-redundant with
what is already present: the weekly Fourier pair encodes the same cycle in two smooth
columns, and `is_weekend` captures the sharp weekday/weekend level shift that matters most
for transport demand. Adding seven collinear dummies on top would inflate dimensionality
against 1310 observations for little gain, and would make the XGBoost residual stage split
on redundant columns. `day_of_week_es` is preserved as a raw reference column for EDA and
grouping, and can be revisited in Phase 3 if per-weekday residual structure shows up.

Encoding is EXPLICIT rather than reference-coded: every category, including the `'none'`
holiday sentinel and the zero-observation `'domingo festivo'`, gets its own column. There
is no implicit base level. Per CLAUDE.md this project favours explicit failure over
implicit defaults -- a dropped reference level makes an all-zero row ambiguous between
"base category" and "encoding bug", and the one-hot-sums-to-1 test in test_features.py
would not be able to tell them apart.
"""

import pandas as pd

from src.features.naming import FEATURE_PREFIX, slugify

# Mirrors DAY_TYPE_CATEGORIES declared in src/ingestion/load_calendar.py. 'domingo festivo'
# has zero observed rows but still gets an (all-zero) column, so the feature matrix width
# does not change if the category ever appears in refreshed data.
DAY_TYPE_CATEGORIES: list[str] = [
    "laborable",
    "sabado",
    "domingo",
    "festivo",
    "domingo festivo",
]

# 'none' is the Phase 1 sentinel for "not a holiday" and is encoded as a real column.
HOLIDAY_TYPE_CATEGORIES: list[str] = [
    "none",
    "Festivo nacional",
    "Festivo de la Comunidad de Madrid",
    "Festivo local de la ciudad de Madrid",
]

# Day types that count as a holiday for the purpose of bridge-day adjacency.
HOLIDAY_DAY_TYPES: list[str] = ["festivo", "domingo festivo"]
# Day types that count as weekend.
WEEKEND_DAY_TYPES: list[str] = ["sabado", "domingo", "domingo festivo"]

DAY_TYPE_COL = "day_type"
HOLIDAY_TYPE_COL = "holiday_type"


def day_type_column_names() -> list[str]:
    return [f"{FEATURE_PREFIX}day_type_{slugify(c)}" for c in DAY_TYPE_CATEGORIES]


def holiday_type_column_names() -> list[str]:
    return [f"{FEATURE_PREFIX}holiday_type_{slugify(c)}" for c in HOLIDAY_TYPE_CATEGORIES]


def calendar_feature_names() -> list[str]:
    """The exact column names this module produces, in order."""
    return [
        *day_type_column_names(),
        *holiday_type_column_names(),
        f"{FEATURE_PREFIX}is_weekend",
        f"{FEATURE_PREFIX}is_bridge_day",
    ]


def _one_hot(series: pd.Series, categories: list[str], prefix: str) -> pd.DataFrame:
    """Explicit one-hot over a fixed category list, raising on any unmapped value."""
    observed = set(series.dropna().astype(str).unique())
    unexpected = sorted(observed - set(categories))
    if unexpected:
        raise ValueError(
            f"calendar_features: column '{prefix}' contains unmapped value(s) "
            f"{unexpected}. Expected one of {categories}. Extend the category list "
            "deliberately rather than emitting an all-zero row."
        )

    as_str = series.astype(str)
    return pd.DataFrame(
        {
            f"{FEATURE_PREFIX}{prefix}_{slugify(cat)}": (as_str == cat).astype("int8")
            for cat in categories
        },
        index=series.index,
    )


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build calendar one-hots and the weekend / bridge-day flags.

    Args:
        df: Unified daily frame sorted ascending by 'date', containing 'day_type' and
            'holiday_type'.

    Returns:
        A features-only DataFrame sharing `df`'s index: one column per day_type category,
        one per holiday_type category, plus `feat_is_weekend` and `feat_is_bridge_day`.
        All int8, no NaNs at any row.

    Raises:
        ValueError: if required columns are missing, the frame is unsorted, or a category
            value falls outside the declared lists.
    """
    missing = [c for c in ["date", DAY_TYPE_COL, HOLIDAY_TYPE_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"calendar_features: missing required column(s) {missing}")

    if not df["date"].is_monotonic_increasing:
        raise ValueError(
            "calendar_features: frame is not sorted ascending by date; bridge-day "
            "adjacency reads neighbouring rows and would pair the wrong days."
        )

    day_type_oh = _one_hot(df[DAY_TYPE_COL], DAY_TYPE_CATEGORIES, "day_type")
    holiday_oh = _one_hot(df[HOLIDAY_TYPE_COL], HOLIDAY_TYPE_CATEGORIES, "holiday_type")

    day_type_str = df[DAY_TYPE_COL].astype(str)

    is_weekend = day_type_str.isin(WEEKEND_DAY_TYPES).astype("int8")

    # A bridge day ("puente") is a WORKING day wedged against a holiday. The `laborable`
    # conjunct is what keeps holidays and weekends themselves out of the flag.
    # fill_value=False rather than .fillna(): the series boundaries have no known
    # neighbour, and treating "unknown" as "not a holiday" is the conservative choice
    # (it can only ever clear the flag, never set it spuriously). Passing fill_value
    # directly also avoids the object-dtype downcast that .fillna() now warns about.
    is_holiday = day_type_str.isin(HOLIDAY_DAY_TYPES)
    prev_is_holiday = is_holiday.shift(1, fill_value=False).astype(bool)
    next_is_holiday = is_holiday.shift(-1, fill_value=False).astype(bool)
    is_bridge_day = (
        (day_type_str == "laborable") & (prev_is_holiday | next_is_holiday)
    ).astype("int8")

    features = pd.concat([day_type_oh, holiday_oh], axis=1)
    features[f"{FEATURE_PREFIX}is_weekend"] = is_weekend
    features[f"{FEATURE_PREFIX}is_bridge_day"] = is_bridge_day

    return features
