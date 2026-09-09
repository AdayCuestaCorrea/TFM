"""Persistence (naive) baseline: yhat_t = y_{t-1}.

Trivial by construction, and that is the point: it is the reference every other model must
beat to justify its complexity. On a strongly weekly series persistence is a *weak* naive
model -- it carries Monday's level into Tuesday and misses every weekend transition -- so
a seasonal-naive variant (yhat_t = y_{t-7}) is also provided for context, since beating
plain persistence is easy while beating seasonal-naive is the real bar at daily resolution.

It reads the already-built `feat_total_lag_1` column rather than recomputing a shift, so
that the baseline and the learned models consume byte-identical inputs. A baseline that
built its own lag could silently disagree with the feature table about alignment, and the
comparison would be quietly unfair.
"""

import pandas as pd

PERSISTENCE_COL = "feat_total_lag_1"
SEASONAL_NAIVE_COL = "feat_total_lag_7"


def predict_persistence(df: pd.DataFrame) -> pd.Series:
    """yhat_t = y_{t-1}, taken from the prebuilt lag-1 feature column.

    Args:
        df: Feature table containing PERSISTENCE_COL.

    Returns:
        Predictions aligned to `df`'s index (NaN wherever the lag is undefined).

    Raises:
        ValueError: if the required column is absent.
    """
    if PERSISTENCE_COL not in df.columns:
        raise ValueError(
            f"persistence: missing '{PERSISTENCE_COL}'. Build features first "
            "(python -m src.features.build_features)."
        )
    return df[PERSISTENCE_COL].rename("persistence")


def predict_seasonal_naive(df: pd.DataFrame) -> pd.Series:
    """yhat_t = y_{t-7}: same weekday last week. The honest bar for a weekly series."""
    if SEASONAL_NAIVE_COL not in df.columns:
        raise ValueError(f"seasonal naive: missing '{SEASONAL_NAIVE_COL}'")
    return df[SEASONAL_NAIVE_COL].rename("seasonal_naive")
