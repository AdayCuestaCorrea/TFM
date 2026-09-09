"""Load CRTM daily demand (target + operator-level series) into the canonical schema.

The raw workbook is a human-facing report, not a data export, so its layout is
defensive-parsing territory: the header sits on Excel row 2 (header=1, NOT header=2 --
header=2 silently consumes 2023-01-01 as the header), the date column carries no header
at all, and three junk columns surround the data.
"""

from pathlib import Path

import pandas as pd

from src.utils.paths import CRTM_FILE

SHEET_NAME = "diaria"
HEADER_ROW = 1  # 0-indexed -> Excel row 2

# Physical labels as they appear once the sheet is read with header=1. Asserting on this
# exact list (not just its length) makes any upstream schema drift fail loudly here rather
# than surface as a silent NaN column three phases downstream.
EXPECTED_RAW_COLUMNS = [
    "Unnamed: 0",  # stray 'x' marker in column A
    "Unnamed: 1",  # the date column: present but unnamed in the source
    "Metro de Madrid",
    "EMT",
    "Conc. por carretera",
    "Renfe Cercanías",
    "Total",
    "Unnamed: 7",  # fully empty spacer column
    "Nota: los datos de demanda reflejados son provisionales y como tal han de considerarse.",
]

JUNK_COLUMNS = ["Unnamed: 0", "Unnamed: 7", EXPECTED_RAW_COLUMNS[-1]]

COLUMN_MAP = {
    "Unnamed: 1": "date",
    "Metro de Madrid": "metro",
    "EMT": "emt",
    "Conc. por carretera": "carretera",  # interurban bus concessionaires
    "Renfe Cercanías": "cercanias",
    "Total": "total",
}

CANONICAL_COLUMNS = ["date", "metro", "emt", "carretera", "cercanias", "total"]
NUMERIC_COLUMNS = ["metro", "emt", "carretera", "cercanias", "total"]

# Some cells are Excel formula results and arrive with float64 round-off: 2026-07-07 holds
# carretera=847904.00000001 (and total=4748815.00000001, since total is their sum), a
# deviation of ~1e-8. These are integer passenger counts, not fractional values, so they
# are rounded rather than rejected. The tolerance sits far below one passenger yet well
# above float64 noise at this magnitude (~1e-9), so a genuinely fractional value -- which
# would signal that the column no longer means "number of trips" -- still raises.
INTEGER_TOLERANCE = 1e-6


def load_crtm(path: Path | str = CRTM_FILE) -> pd.DataFrame:
    """Read the CRTM daily demand sheet and return the canonical six-column frame.

    Args:
        path: Path to CRTM_Evolucion_demanda_diaria.xlsx.

    Returns:
        DataFrame with columns [date, metro, emt, carretera, cercanias, total],
        sorted ascending by date. Date is datetime64[ns]; the five demand columns
        are int64.

    Raises:
        ValueError: if the raw column layout drifts from the expected one, or if any
            demand value carries a fractional part (which would make the int64 cast lossy).
    """
    raw = pd.read_excel(path, sheet_name=SHEET_NAME, header=HEADER_ROW)

    if len(raw.columns) != len(EXPECTED_RAW_COLUMNS):
        raise ValueError(
            f"CRTM schema drift: expected {len(EXPECTED_RAW_COLUMNS)} raw columns, "
            f"found {len(raw.columns)}: {list(raw.columns)}"
        )
    if list(raw.columns) != EXPECTED_RAW_COLUMNS:
        raise ValueError(
            "CRTM schema drift: raw column labels differ from expected.\n"
            f"  expected: {EXPECTED_RAW_COLUMNS}\n"
            f"  found:    {list(raw.columns)}"
        )

    # Drop by explicit label rather than by position, so a future inserted column cannot
    # shift the slice and quietly discard real data.
    df = raw.drop(columns=JUNK_COLUMNS).rename(columns=COLUMN_MAP)

    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    # carretera/total arrive as float64 partly because openpyxl widens the column and
    # partly from formula round-off. Verify the values really are integral (within
    # INTEGER_TOLERANCE) before casting, so a genuinely non-integer future value raises
    # instead of being silently truncated toward zero.
    for col in NUMERIC_COLUMNS:
        values = pd.to_numeric(df[col], errors="raise").astype("float64")
        deviation = (values - values.round()).abs()
        fractional = values[values.notna() & (deviation > INTEGER_TOLERANCE)]
        if not fractional.empty:
            offending = df.loc[fractional.index, "date"].dt.strftime("%Y-%m-%d").tolist()
            raise ValueError(
                f"CRTM column '{col}' holds {len(fractional)} genuinely fractional "
                f"value(s); refusing lossy int64 cast. "
                f"Examples: {list(zip(offending[:5], fractional.head().tolist()))}"
            )
        df[col] = values.round().astype("int64")

    df = df[CANONICAL_COLUMNS].sort_values("date").reset_index(drop=True)
    return df
