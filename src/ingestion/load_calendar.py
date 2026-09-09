"""Load the Comunidad de Madrid working calendar into the canonical schema.

The file is semicolon-separated and UTF-8 with BOM, so encoding='utf-8-sig' is required:
with plain 'utf-8' the BOM is glued onto the first column name ('﻿Dia') and the
rename silently no-ops.
"""

from pathlib import Path

import pandas as pd

from src.utils.paths import CALENDAR_FILE

DATE_FORMAT = "%d/%m/%Y"  # explicit: never rely on dayfirst inference for dd/mm/yyyy

COLUMN_MAP = {
    "Dia": "date",
    "Dia_semana": "day_of_week_es",
    "laborable / festivo / domingo festivo": "day_type",
    "Tipo de Festivo": "holiday_type",
    "Festividad": "holiday_name",
}

CANONICAL_COLUMNS = ["date", "day_of_week_es", "day_type", "holiday_type", "holiday_name"]

# 'domingo festivo' is declared in the source COLUMN NAME but has zero observed rows.
# It is kept as an explicit category so that if it ever appears in a data refresh it is
# encoded as a first-class level instead of silently becoming NaN. Any value outside this
# list raises rather than being dropped -- see the unmapped-value check in load_calendar.
DAY_TYPE_CATEGORIES = ["laborable", "sabado", "domingo", "festivo", "domingo festivo"]

# 'day_of_week_es' values are unaccented ASCII in the source: 'miercoles' and 'sabado',
# NOT 'miércoles'/'sábado'. They are deliberately left as-is. Any future join-by-name or
# mapping logic must match this exact unaccented form.


def load_calendar(path: Path | str = CALENDAR_FILE) -> pd.DataFrame:
    """Read the working calendar and return the canonical five-column frame.

    Args:
        path: Path to 300082-1-calendario_laboral-csv.csv.

    Returns:
        DataFrame with columns [date, day_of_week_es, day_type, holiday_type,
        holiday_name], sorted ascending by date. 'date' is datetime64[ns] and 'day_type'
        is a pandas Categorical over DAY_TYPE_CATEGORIES. 'holiday_type'/'holiday_name'
        are NaN on non-holidays, which is meaningful absence rather than missing data.

    Raises:
        ValueError: if expected columns are missing, or if 'day_type' contains any value
            outside DAY_TYPE_CATEGORIES.
    """
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")

    missing = [c for c in COLUMN_MAP if c not in df.columns]
    if missing:
        raise ValueError(
            f"Calendar schema drift: missing expected column(s) {missing}. "
            f"Found: {list(df.columns)}"
        )

    df = df.rename(columns=COLUMN_MAP)[CANONICAL_COLUMNS].copy()

    df["date"] = pd.to_datetime(df["date"], format=DATE_FORMAT).dt.normalize()

    # Detect unmapped values BEFORE the categorical cast: casting first would turn them
    # into NaN and destroy the evidence needed for the error message.
    observed = set(df["day_type"].dropna().unique())
    unexpected = sorted(observed - set(DAY_TYPE_CATEGORIES))
    if unexpected:
        raise ValueError(
            f"Calendar 'day_type' contains unmapped value(s): {unexpected}. "
            f"Expected one of {DAY_TYPE_CATEGORIES}. Extend DAY_TYPE_CATEGORIES "
            "deliberately rather than letting the value be dropped."
        )

    df["day_type"] = pd.Categorical(df["day_type"], categories=DAY_TYPE_CATEGORIES)

    return df.sort_values("date").reset_index(drop=True)
