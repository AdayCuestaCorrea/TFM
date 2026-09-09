"""Merge the three canonical sources into a single unified daily dataset.

The merge is deliberately paranoid: an inner join on 'date' is the only safe join here
(a left join would fabricate all-NaN exogenous rows), but an inner join also *hides*
coverage gaps by construction. Every assertion below exists to make a gap that the join
would have swallowed raise instead, naming the offending dates.
"""

from pathlib import Path

import pandas as pd

from src.ingestion.load_calendar import load_calendar
from src.ingestion.load_crtm import load_crtm
from src.ingestion.load_weather import load_weather
from src.utils.paths import UNIFIED_DAILY_FILE

# 'holiday_type' and 'holiday_name' are NaN on non-holidays. That is meaningful absence,
# not missing data, so it is encoded as an explicit sentinel rather than left as NaN:
# it keeps the "zero nulls" invariant below honest (a null in this dataset always means a
# real defect) and gives Phase 2 a clean categorical level to one-hot encode.
NO_HOLIDAY = "none"
HOLIDAY_COLUMNS = ["holiday_type", "holiday_name"]


def _missing_dates(reference: pd.Series, other: pd.Series) -> list[str]:
    """Dates present in `reference` but absent from `other`, as ISO strings."""
    gap = pd.DatetimeIndex(reference).difference(pd.DatetimeIndex(other))
    return [d.strftime("%Y-%m-%d") for d in gap]


def unify_frames(
    crtm: pd.DataFrame, weather: pd.DataFrame, calendar: pd.DataFrame
) -> pd.DataFrame:
    """Merge three canonical frames on 'date' and validate temporal integrity.

    Args:
        crtm: Canonical CRTM demand frame (see load_crtm).
        weather: Canonical weather frame (see load_weather).
        calendar: Canonical calendar frame (see load_calendar).

    Returns:
        The unified daily DataFrame, sorted by date, with a continuous daily index and
        no nulls.

    Raises:
        ValueError: if the inner join drops rows from any source, if the unified date
            range has calendar gaps, or if duplicates or nulls survive to the end.
    """
    sources = {"crtm": crtm, "weather": weather, "calendar": calendar}

    for name, frame in sources.items():
        dupes = frame["date"][frame["date"].duplicated()]
        if not dupes.empty:
            raise ValueError(
                f"Source '{name}' has {len(dupes)} duplicate date(s), e.g. "
                f"{[d.strftime('%Y-%m-%d') for d in dupes.head()]}"
            )

    merged = crtm.merge(weather, on="date", how="inner").merge(
        calendar, on="date", how="inner"
    )

    # An inner join silently discards any date not common to all three sources. Compare
    # the survivor count against every source and name what was lost.
    for name, frame in sources.items():
        if len(merged) != len(frame):
            lost = _missing_dates(frame["date"], merged["date"])
            raise ValueError(
                f"Inner join dropped rows: source '{name}' has {len(frame)} rows but the "
                f"merge produced {len(merged)}. Date(s) in '{name}' missing from at least "
                f"one other source ({len(lost)} total): {lost[:20]}"
                + (" ..." if len(lost) > 20 else "")
            )

    merged = merged.sort_values("date").reset_index(drop=True)

    # Re-verify continuity at the unified level, not just per-source: three sources can
    # each be internally gap-free yet share a common hole.
    full_range = pd.date_range(merged["date"].min(), merged["date"].max(), freq="D")
    reindexed = merged.set_index("date").reindex(full_range)
    if len(reindexed) != len(merged):
        gaps = full_range.difference(pd.DatetimeIndex(merged["date"]))
        raise ValueError(
            f"Unified dataset has {len(gaps)} missing calendar day(s) between "
            f"{merged['date'].min():%Y-%m-%d} and {merged['date'].max():%Y-%m-%d}: "
            f"{[d.strftime('%Y-%m-%d') for d in gaps[:20]]}"
            + (" ..." if len(gaps) > 20 else "")
        )

    unified = reindexed.rename_axis("date").reset_index()

    for col in HOLIDAY_COLUMNS:
        if isinstance(unified[col].dtype, pd.CategoricalDtype):
            unified[col] = unified[col].cat.add_categories([NO_HOLIDAY])
        unified[col] = unified[col].fillna(NO_HOLIDAY)

    if unified["date"].duplicated().any():
        raise ValueError("Unified dataset contains duplicate dates after merge.")

    null_counts = unified.isna().sum()
    offenders = null_counts[null_counts > 0]
    if not offenders.empty:
        raise ValueError(f"Unified dataset contains nulls: {offenders.to_dict()}")

    return unified


OPERATOR_COLS: list[str] = ["metro", "emt", "carretera", "cercanias"]
TARGET_COL = "total"
# The operator columns are recorded as integer passenger counts, so their sum equals
# `total` EXACTLY -- not approximately. A tolerance of 0 is the honest assertion; anything
# looser would quietly accept a source file whose parts no longer add up to its whole.
OPERATOR_SUM_TOLERANCE = 0.0


def integrity_report(df: pd.DataFrame) -> pd.DataFrame:
    """Re-derive the dataset's integrity invariants as a displayable table.

    `unify_frames` already enforces gaps, duplicates and nulls as `raise` sites, so a frame
    that exists at all has passed them. This function exists because a guard that fires is
    invisible when it does NOT fire: the notebook needs to *show* the invariants holding,
    with their observed values, rather than assert that they were checked somewhere upstream.

    It also adds the one invariant that has no guard of its own -- the operator-sum identity
    metro+emt+carretera+cercanias == total. That identity is why same-day operator columns
    are perfect leakage of the target (see `features/lag_features.py`), so it is worth
    displaying as a measured fact rather than a claim in a docstring.

    Args:
        df: A unified daily frame as produced by `unify_frames`.

    Returns:
        DataFrame with columns [comprobacion, esperado, observado, veredicto], one row per
        invariant. `veredicto` is 'OK' or 'FALLO'; the caller decides what to do about it.
    """
    dates = pd.DatetimeIndex(df["date"])
    expected_span = pd.date_range(dates.min(), dates.max(), freq="D")
    n_gaps = len(expected_span) - len(dates)
    n_duplicates = int(df["date"].duplicated().sum())
    n_nulls = int(df.isna().sum().sum())

    missing_ops = [c for c in (*OPERATOR_COLS, TARGET_COL) if c not in df.columns]
    if missing_ops:
        raise ValueError(f"integrity_report: missing required column(s) {missing_ops}")
    residual = (df[OPERATOR_COLS].sum(axis=1) - df[TARGET_COL]).abs()
    max_residual = float(residual.max())

    rows = [
        ("filas", str(len(expected_span)), str(len(df)), len(df) == len(expected_span)),
        ("huecos de calendario", "0", str(n_gaps), n_gaps == 0),
        ("fechas duplicadas", "0", str(n_duplicates), n_duplicates == 0),
        ("valores nulos", "0", str(n_nulls), n_nulls == 0),
        (
            "suma de operadores == total",
            "desviacion maxima 0",
            f"desviacion maxima {max_residual:g}",
            max_residual <= OPERATOR_SUM_TOLERANCE,
        ),
    ]

    return pd.DataFrame(
        [
            {
                "comprobacion": name,
                "esperado": expected,
                "observado": observed,
                "veredicto": "OK" if ok else "FALLO",
            }
            for name, expected, observed, ok in rows
        ]
    )


def unify(
    save: bool = True, output_path: Path | str = UNIFIED_DAILY_FILE
) -> pd.DataFrame:
    """Load all three raw sources, unify them, and optionally persist to parquet.

    Args:
        save: Whether to write the result to `output_path`.
        output_path: Destination parquet file.

    Returns:
        The unified daily DataFrame.
    """
    unified = unify_frames(load_crtm(), load_weather(), load_calendar())

    if save:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Parquet, not CSV: it round-trips dtypes (datetime64, int64, Categorical) so
        # later phases do not have to re-infer or re-cast the schema.
        unified.to_parquet(output_path, index=False)

    return unified


def print_schema_summary(df: pd.DataFrame, output_path: Path | str | None = None) -> None:
    """Print column names, dtypes, date range and row count to stdout."""
    print("=" * 72)
    print("UNIFIED DAILY DATASET — schema summary")
    print("=" * 72)
    if output_path is not None:
        print(f"file        : {output_path}")
    print(f"rows        : {len(df)}")
    print(f"columns     : {len(df.columns)}")
    print(
        f"date range  : {df['date'].min():%Y-%m-%d} -> {df['date'].max():%Y-%m-%d} "
        f"(daily, {len(df)} observations)"
    )
    print(f"missing days: {len(pd.date_range(df['date'].min(), df['date'].max())) - len(df)}")
    print(f"duplicates  : {int(df['date'].duplicated().sum())}")
    print(f"total nulls : {int(df.isna().sum().sum())}")
    print("-" * 72)
    print(f"{'#':>3}  {'column':<34} {'dtype':<20} nulls")
    print("-" * 72)
    for i, col in enumerate(df.columns):
        print(f"{i:>3}  {col:<34} {str(df[col].dtype):<20} {int(df[col].isna().sum())}")
    print("=" * 72)


if __name__ == "__main__":
    frame = unify(save=True)
    print_schema_summary(frame, UNIFIED_DAILY_FILE)
