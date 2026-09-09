"""Phase 2 orchestrator: assemble the full daily feature table.

Column contract of the output (downstream code must never have to guess):

    date                    the daily index
    TARGET_COL              'total', raw same-day -- the modeling target, NOT a feature
    RAW_REFERENCE_COLS      raw same-day operator + calendar columns, kept for EDA and
                            multimodal analysis per CLAUDE.md. NEVER feed these to a model
                            same-day: metro+emt+carretera+cercanias == total exactly.
    PASSTHROUGH_WEATHER_COLS  same-day weather, carried through unmodified (see note below)
    feature_columns()       everything prefixed 'feat_' -- the model-ready matrix

Order of operations: leakage-free features (calendar, Fourier) first, then the
history-dependent ones (lags, rolling). This is presentational rather than semantic --
the modules are independent -- but it keeps the safe/unsafe boundary visible in the code.

NaN POLICY: the first 28 rows carry NaNs from the 28-day lag and 28-day rolling windows.
They are LEFT IN PLACE. Dropping or imputing is a training-time decision that depends on
the consumer -- an LSTM needs a contiguous NaN-free window, XGBoost handles NaNs natively
and would lose information if rows were dropped -- so it belongs to Phase 3, not here.

RESOLVED IN PHASE 3 -- same-day weather. The 21 raw weather columns remain unprefixed and
are therefore NOT in feature_columns(); they are kept only as reference/EDA columns. What
enters the model matrix is their LAGGED and rolling forms, built by weather_features.py.
Rationale in that module's docstring: same-day weather is observed data, not a forecast,
so using it unlagged assumes perfect foreknowledge. The same-day variant is deferred to a
labelled sensitivity analysis in the evaluation phase, not used as the reference model.
"""

from pathlib import Path

import pandas as pd

from src.features.calendar_features import add_calendar_features
from src.features.fourier_features import add_fourier_features
from src.features.lag_features import (
    FORBIDDEN_SAME_DAY_COLS,
    OPERATOR_COLS,
    add_lag_features,
)
from src.features.naming import FEATURE_PREFIX
from src.features.rolling_features import add_rolling_features
from src.features.weather_features import WEATHER_COLS, add_weather_features
from src.utils.paths import PROCESSED_DIR, UNIFIED_DAILY_FILE

TARGET_COL = "total"

RAW_REFERENCE_COLS: list[str] = [
    *OPERATOR_COLS,  # metro, emt, carretera, cercanias
    "day_of_week_es",
    "day_type",
    "holiday_type",
    "holiday_name",
]

FEATURES_DAILY_FILE: Path = PROCESSED_DIR / "features_daily.parquet"


def feature_columns(df: pd.DataFrame) -> list[str]:
    """The model-ready feature columns: everything carrying FEATURE_PREFIX."""
    return [c for c in df.columns if c.startswith(FEATURE_PREFIX)]


def passthrough_weather_columns(df: pd.DataFrame) -> list[str]:
    """Same-day weather columns carried through unmodified (see module docstring)."""
    known = {"date", TARGET_COL, *RAW_REFERENCE_COLS}
    return [c for c in df.columns if c not in known and not c.startswith(FEATURE_PREFIX)]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Assemble the full feature table from a unified daily frame.

    Args:
        df: Unified daily frame as produced by Phase 1.

    Returns:
        DataFrame with the column contract described in the module docstring, same row
        count as `df`, NaNs from the warm-up window left in place.

    Raises:
        ValueError: if a raw same-day operator column reaches the feature block, or if the
            row count changes during assembly.
    """
    n_input = len(df)
    base = df.sort_values("date").reset_index(drop=True)

    # Leakage-free first: deterministic functions of the calendar date only.
    calendar = add_calendar_features(base)
    fourier = add_fourier_features(base)
    # Then the history-dependent blocks, which introduce the warm-up NaNs.
    lags = add_lag_features(base)
    rolling = add_rolling_features(base)
    # Phase 3: weather promoted into the matrix, lagged only (never same-day).
    weather = add_weather_features(base)

    features = pd.concat([calendar, fourier, lags, rolling, weather], axis=1)

    ordered = [
        "date",
        TARGET_COL,
        *[c for c in RAW_REFERENCE_COLS if c in base.columns],
        *passthrough_weather_columns(base),
    ]
    out = pd.concat([base[ordered], features], axis=1)

    # Belt and braces: the per-module guard already ran, but re-assert on the assembled
    # frame, since concat is where a mis-named column would actually land in the matrix.
    #
    # BOTH checks scan for the PREFIXED name (`feat_metro`, `feat_temperature_2m_mean`),
    # never the bare one. The bare name is the wrong thing to look for twice over: it can
    # never appear in `feats`, because `feature_columns()` selects on the prefix and filters
    # it out by construction; and it is *expected* in `out`, because RAW_REFERENCE_COLS
    # deliberately carries the raw operator columns and TARGET_COL carries `total`. A check
    # against bare names is therefore either vacuous or would raise on every valid frame.
    #
    # What these two CAN catch is a builder regression -- `add_lag_features` gaining a lag 0,
    # or `add_weather_features` passing a same-day column through with the prefix attached --
    # which is exactly the concat-time defect this block exists for.
    #
    # The stronger protection is structural rather than defensive: the builders emit only
    # lag >= 1 and rolling forms, so a same-day column has no route into the matrix in the
    # first place. These guards are the secondary net for when that stops being true.
    feats = feature_columns(out)
    forbidden_prefixed = {f"{FEATURE_PREFIX}{c}": c for c in FORBIDDEN_SAME_DAY_COLS}
    leaked = [forbidden_prefixed[c] for c in feats if c in forbidden_prefixed]
    if leaked:
        raise ValueError(
            f"LEAKAGE GUARD: raw same-day column(s) {leaked} present in the feature block. "
            "metro+emt+carretera+cercanias == total exactly, so a same-day operator column "
            "is perfect leakage of the target."
        )
    weather_prefixed = {f"{FEATURE_PREFIX}{w}": w for w in WEATHER_COLS}
    leaked_wx = [weather_prefixed[c] for c in feats if c in weather_prefixed]
    if leaked_wx:
        raise ValueError(
            f"SAME-DAY WEATHER GUARD: unlagged weather column(s) {leaked_wx} present in "
            "the feature block. Same-day weather is observed data, not a forecast; using it "
            "assumes perfect foreknowledge and inflates apparent performance."
        )
    if len(out) != n_input:
        raise ValueError(
            f"build_features changed the row count: {n_input} -> {len(out)}. "
            "No row may be dropped in Phase 2; NaN handling belongs to Phase 3."
        )

    return out


def build(
    input_path: Path | str = UNIFIED_DAILY_FILE,
    output_path: Path | str = FEATURES_DAILY_FILE,
    save: bool = True,
) -> pd.DataFrame:
    """Load the unified dataset, build features, and optionally persist to parquet."""
    unified = pd.read_parquet(input_path)
    out = build_features(unified)

    if save:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(output_path, index=False)

    return out


def print_feature_report(df: pd.DataFrame, output_path: Path | str | None = None) -> None:
    """Print feature count, NaN count per feature, and first usable date per feature."""
    feats = feature_columns(df)
    weather = passthrough_weather_columns(df)

    print("=" * 78)
    print("FEATURE TABLE — Phase 2 report")
    print("=" * 78)
    if output_path is not None:
        print(f"file              : {output_path}")
    print(f"rows              : {len(df)}  (unchanged from data/interim)")
    print(f"columns total     : {len(df.columns)}")
    print(f"engineered feats  : {len(feats)}   (prefix '{FEATURE_PREFIX}')")
    print(f"raw reference     : {len([c for c in RAW_REFERENCE_COLS if c in df.columns])}")
    print(f"raw weather (ref) : {len(weather)}  (same-day, NOT in the model matrix)")
    print(
        f"date range        : {df['date'].min():%Y-%m-%d} -> {df['date'].max():%Y-%m-%d}"
    )
    print("-" * 78)
    print(f"{'feature':<44} {'dtype':<9} {'NaNs':>5}  first valid")
    print("-" * 78)
    for col in feats:
        n_nan = int(df[col].isna().sum())
        first_valid = df[col].first_valid_index()
        first_date = (
            df.loc[first_valid, "date"].strftime("%Y-%m-%d")
            if first_valid is not None
            else "NEVER"
        )
        print(f"{col:<44} {str(df[col].dtype):<9} {n_nan:>5}  {first_date}")
    print("-" * 78)

    nan_feats = [c for c in feats if df[c].isna().any()]
    print(f"features with NaNs: {len(nan_feats)} of {len(feats)}")
    if nan_feats:
        worst = max(int(df[c].isna().sum()) for c in nan_feats)
        first_clean = df.index[df[feats].notna().all(axis=1)]
        print(f"max NaNs in any feature : {worst}")
        print(
            "first fully-usable row  : index "
            f"{first_clean[0]} ({df.loc[first_clean[0], 'date']:%Y-%m-%d})"
        )
    print("NaN policy        : LEFT IN PLACE — drop/impute deferred to Phase 3.")
    print("=" * 78)


if __name__ == "__main__":
    frame = build(save=True)
    print_feature_report(frame, FEATURES_DAILY_FILE)
