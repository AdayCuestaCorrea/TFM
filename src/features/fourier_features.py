"""Deterministic Fourier seasonality terms (weekly and annual).

WHY THESE ARE THE ONE EXEMPTION TO THE "NO GLOBAL FEATURE ENGINEERING BEFORE SPLIT" RULE
(CLAUDE.md, Data Leakage Rule 4):

Rule 4 exists because a statistic computed over the full dataset embeds information from
the test period into the training features -- a rolling mean, a fitted scaler, or a
seasonal decomposition all *estimate parameters from observed y*, so computing them before
the split leaks the future.

These Fourier terms estimate nothing. The value at date t is a closed-form function of t
alone:

    sin(2*pi*k*day_index(t) / period),  cos(2*pi*k*day_index(t) / period)

No observed target value, no fitted coefficient, no dataset-dependent quantity enters the
computation. Computing them over the full date range produces byte-identical values to
computing them separately on each split, and the terms for a date one year in the future
are just as computable as for a date in the training set. There is therefore zero leakage
risk regardless of where the split point falls.

The contrast worth keeping in mind: fitting a *seasonal model* (amplitudes/phases regressed
against observed demand) on full data WOULD leak, because that estimates parameters from y.
Only the deterministic basis functions are exempt -- the regression coefficients on them
are learned by the model inside the training fold, as normal.

The epoch is a fixed constant rather than `df['date'].min()` so that appending future data
never retroactively changes the encoding of existing rows -- feature values must be stable
across re-runs for the seed-42 reproducibility convention to mean anything.
"""

import numpy as np
import pandas as pd

from src.features.naming import FEATURE_PREFIX

# Fixed anchor: the first date of the Phase 1 dataset. Deliberately a constant.
FOURIER_EPOCH = pd.Timestamp("2023-01-01")

WEEKLY_PERIOD = 7.0
WEEKLY_HARMONICS: tuple[int, ...] = (1,)  # fundamental only

# 365.25 rather than 365 so the annual phase does not drift by a day every leap year.
ANNUAL_PERIOD = 365.25
# k=1 is the annual cycle. k=2 adds the semi-annual harmonic, motivated by Madrid's
# academic calendar: demand has a summer/winter split-year structure that a single
# sinusoid cannot represent (one trough in August, a second smaller one at Christmas).
ANNUAL_HARMONICS: tuple[int, ...] = (1, 2)


def fourier_feature_names() -> list[str]:
    """The exact column names this module produces, in order."""
    names: list[str] = []
    for k in WEEKLY_HARMONICS:
        names += [
            f"{FEATURE_PREFIX}fourier_weekly_k{k}_sin",
            f"{FEATURE_PREFIX}fourier_weekly_k{k}_cos",
        ]
    for k in ANNUAL_HARMONICS:
        names += [
            f"{FEATURE_PREFIX}fourier_annual_k{k}_sin",
            f"{FEATURE_PREFIX}fourier_annual_k{k}_cos",
        ]
    return names


def day_index(dates: pd.Series) -> pd.Series:
    """Whole days elapsed since FOURIER_EPOCH (may be negative for earlier dates)."""
    return (pd.to_datetime(dates).dt.normalize() - FOURIER_EPOCH).dt.days


def add_fourier_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build weekly and annual Fourier terms from the calendar date alone.

    Args:
        df: Frame containing a 'date' column. No other column is read, and no target
            value is touched.

    Returns:
        A features-only DataFrame sharing `df`'s index with sin/cos pairs for the weekly
        fundamental and for annual harmonics k=1 and k=2. Contains no NaNs at any row,
        including the first: these features need no history.

    Raises:
        ValueError: if 'date' is absent.
    """
    if "date" not in df.columns:
        raise ValueError("fourier_features: missing required column 'date'")

    index = day_index(df["date"])
    features: dict[str, pd.Series] = {}

    for period, harmonics, label in (
        (WEEKLY_PERIOD, WEEKLY_HARMONICS, "weekly"),
        (ANNUAL_PERIOD, ANNUAL_HARMONICS, "annual"),
    ):
        for k in harmonics:
            angle = 2.0 * np.pi * k * index / period
            features[f"{FEATURE_PREFIX}fourier_{label}_k{k}_sin"] = np.sin(angle)
            features[f"{FEATURE_PREFIX}fourier_{label}_k{k}_cos"] = np.cos(angle)

    return pd.DataFrame(features, index=df.index)
