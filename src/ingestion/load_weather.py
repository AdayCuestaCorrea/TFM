"""Load Open-Meteo daily weather into the canonical schema.

File layout (verified against the bytes, not assumed):
    line 1  metadata header  (latitude,longitude,elevation,utc_offset_seconds,timezone,...)
    line 2  metadata values  (40.386642,-3.6760864,666.0,7200,Europe/Berlin,GMT+2)
    line 3  blank separator
    line 4  the real column header
    line 5+ data
so skiprows=3 drops lines 1-3 and lands the header on line 4.

TIMEZONE NOTE: the source declares its timezone as 'Europe/Berlin' (GMT+2, utc_offset
7200s) rather than 'Europe/Madrid'. This is NOT a real offset discrepancy: peninsular
Spain and Germany share the CET/CEST timezone and switch DST on the same dates, so the
daily aggregation boundaries are identical. It is purely a labeling quirk of how the
extract was requested. Dates align correctly with the CRTM and calendar sources and NO
timezone conversion is required or applied here.
"""

import re
from pathlib import Path

import pandas as pd

from src.utils.paths import WEATHER_FILE

SKIPROWS = 3  # metadata header, metadata values, blank separator

# Trailing unit annotation, e.g. ' (°C)', ' (MJ/m²)', ' (wmo code)'. Matched generically
# rather than against a fixed list of units, so a new variable with a new unit still
# normalises correctly instead of keeping a parenthesised suffix in its column name.
_UNIT_SUFFIX = re.compile(r"\s*\([^)]*\)\s*$")


def _to_snake_case(name: str) -> str:
    """Strip any trailing unit annotation and normalise to snake_case."""
    cleaned = _UNIT_SUFFIX.sub("", name).strip()
    cleaned = re.sub(r"[\s\-/]+", "_", cleaned)
    return cleaned.lower()


def load_weather(path: Path | str = WEATHER_FILE) -> pd.DataFrame:
    """Read the Open-Meteo daily extract and return it with canonical column names.

    Args:
        path: Path to open-meteo-40.39N3.68W666m.csv.

    Returns:
        DataFrame whose first column is 'date' (datetime64[ns]) followed by the weather
        variables in source order, unit suffixes stripped and names snake_cased
        (e.g. 'temperature_2m_mean (°C)' -> 'temperature_2m_mean'), sorted by date.

    Raises:
        ValueError: if the 'time' column is absent, which would mean the header landed on
            the wrong physical line.
    """
    df = pd.read_csv(path, skiprows=SKIPROWS, encoding="utf-8")

    if "time" not in df.columns:
        raise ValueError(
            "Weather schema drift: no 'time' column after skiprows=%d; the header did not "
            "land on the expected physical line. Found: %s"
            % (SKIPROWS, list(df.columns)[:5])
        )

    df = df.rename(columns={c: _to_snake_case(c) for c in df.columns})
    df = df.rename(columns={"time": "date"})

    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    return df.sort_values("date").reset_index(drop=True)
