"""Moving-average baseline: yhat_t = mean(y_{t-7} .. y_{t-1}).

Consumes the prebuilt `feat_total_roll_mean_7` column rather than recomputing the window.
Single source of truth: that column was built shift-first in rolling_features.py and is
covered by the ordering tests there, so reusing it means this baseline cannot drift out of
alignment with the feature table or quietly reintroduce the unshifted-window leak that
those tests exist to prevent.
"""

import pandas as pd

MOVING_AVERAGE_COL = "feat_total_roll_mean_7"
MOVING_AVERAGE_28_COL = "feat_total_roll_mean_28"


def predict_moving_average(df: pd.DataFrame) -> pd.Series:
    """yhat_t = trailing 7-day mean of the target, ending at t-1.

    Args:
        df: Feature table containing MOVING_AVERAGE_COL.

    Returns:
        Predictions aligned to `df`'s index (NaN over the warm-up window).

    Raises:
        ValueError: if the required column is absent.
    """
    if MOVING_AVERAGE_COL not in df.columns:
        raise ValueError(
            f"moving_average: missing '{MOVING_AVERAGE_COL}'. Build features first "
            "(python -m src.features.build_features)."
        )
    return df[MOVING_AVERAGE_COL].rename("moving_average_7")


def predict_moving_average_28(df: pd.DataFrame) -> pd.Series:
    """yhat_t = trailing 28-day mean. Smoother, and included to show the trade-off:

    a longer window suppresses noise but lags regime changes (holiday periods, September
    return) more severely.
    """
    if MOVING_AVERAGE_28_COL not in df.columns:
        raise ValueError(f"moving_average: missing '{MOVING_AVERAGE_28_COL}'")
    return df[MOVING_AVERAGE_28_COL].rename("moving_average_28")
