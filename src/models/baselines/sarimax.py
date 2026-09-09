"""SARIMAX baseline with calendar exogenous regressors.

THREE DESIGN DECISIONS, EACH WITH A REASON:

1. EXOG IS CALENDAR-ONLY -- NO LAG FEATURES.
   SARIMAX already models autoregression through its own (p, d, q)(P, D, Q, s) structure.
   Feeding it the hand-built `feat_total_lag_*` columns as exog would express the same
   dependency twice, in two mechanisms that fight each other: the AR terms and the exog
   coefficients would compete to explain identical variance, producing unstable estimates
   and an uninterpretable model. What SARIMAX genuinely cannot derive on its own is the
   calendar -- it has no way to know a Thursday is a national holiday -- so exog carries
   exactly that. `assert_no_lag_features_in_exog` enforces this mechanically.

2. THE 'none' HOLIDAY LEVEL IS DROPPED FROM EXOG (and only here).
   The Phase 2 encoding is deliberately exhaustive: all four holiday_type dummies sum to 1
   on every row. That is right for tree models but singular for a regression with an
   intercept -- the dummy trap. So exog uses the three real holiday levels and treats
   'none' (not a holiday) as the reference. This is a property of THIS model, not a change
   to the feature table, which keeps its explicit encoding.

3. FORECASTING IS WALK-FORWARD ONE-STEP-AHEAD, NOT A SINGLE STATIC MULTI-STEP FORECAST.
   Fitting once on train and forecasting all ~390 remaining days in one shot would be
   badly misleading: with no new observations to condition on, a SARIMAX forecast decays
   toward the series mean within a few weeks, so the reported error would measure
   "how far into the future did we extrapolate" rather than model quality, and would
   flatter the other baselines by comparison.

   Implementation: parameters are estimated ONCE on the training set, then the fitted
   parameter vector is applied to the full series via `.filter()`. The Kalman filter then
   updates its state with each *observed* value as it walks forward, and each prediction
   at t conditions only on data through t-1 (`dynamic=False`). This is genuine
   one-step-ahead forecasting, and it mirrors realistic deployment -- a production model is
   refit periodically, not daily, but is fed yesterday's actuals every day. No target value
   from validation or test ever influences the estimated coefficients.
"""

import warnings
from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

TARGET_COL = "total"
SEASONAL_PERIOD = 7  # weekly cycle dominates at daily resolution

# Calendar-derived exog. 'feat_holiday_type_none' is deliberately absent: it is the
# reference level (see decision 2 above).
EXOG_COLS: list[str] = [
    "feat_is_weekend",
    "feat_is_bridge_day",
    "feat_holiday_type_festivo_nacional",
    "feat_holiday_type_festivo_de_la_comunidad_de_madrid",
    "feat_holiday_type_festivo_local_de_la_ciudad_de_madrid",
]

# Modest bounded search space, per the phase brief.
P_VALUES = [0, 1, 2]
D_VALUES = [0, 1]
Q_VALUES = [0, 1, 2]
SEASONAL_P_VALUES = [0, 1, 2]
SEASONAL_D_VALUES = [0, 1]
SEASONAL_Q_VALUES = [0, 1, 2]

FIT_KWARGS = dict(disp=False, maxiter=50, method="lbfgs")


@dataclass(frozen=True)
class SarimaxOrder:
    """A fitted SARIMAX specification and its training AIC."""

    order: tuple[int, int, int]
    seasonal_order: tuple[int, int, int, int]
    aic: float

    def __str__(self) -> str:
        return f"SARIMAX{self.order}x{self.seasonal_order} (AIC={self.aic:,.2f})"


def assert_no_lag_features_in_exog(exog: pd.DataFrame) -> None:
    """Fail loudly if any autoregressive feature reached the exog matrix.

    See decision 1: lag/rolling columns duplicate what the AR terms already model.
    """
    offenders = [
        c for c in exog.columns if "_lag_" in c or "_roll_mean_" in c or "_roll_std_" in c
    ]
    if offenders:
        raise ValueError(
            f"SARIMAX exog contains {len(offenders)} autoregressive feature(s): "
            f"{offenders[:5]}. SARIMAX models its own autoregression; exog must carry "
            "only calendar information."
        )


def build_exog(df: pd.DataFrame, exog_cols: list[str] | None = None) -> pd.DataFrame:
    """Assemble and validate the calendar exog matrix."""
    cols = exog_cols if exog_cols is not None else EXOG_COLS
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"SARIMAX: missing exog column(s) {missing}")

    exog = df[cols].astype("float64")
    assert_no_lag_features_in_exog(exog)
    return exog


def grid_search_order(
    y_train: pd.Series,
    exog_train: pd.DataFrame,
    seasonal_period: int = SEASONAL_PERIOD,
    verbose: bool = True,
) -> tuple[SarimaxOrder, pd.DataFrame]:
    """Select (p,d,q)(P,D,Q,s) by AIC on the TRAINING SET ONLY.

    Selecting on training AIC (rather than validation error) keeps the validation split
    genuinely untouched for model comparison: if the order were tuned on validation
    performance, validation metrics would no longer be an out-of-sample estimate.

    Returns:
        (best specification, full results table sorted by AIC).
    """
    combos = list(
        product(
            P_VALUES,
            D_VALUES,
            Q_VALUES,
            SEASONAL_P_VALUES,
            SEASONAL_D_VALUES,
            SEASONAL_Q_VALUES,
        )
    )

    rows: list[dict] = []
    for i, (p, d, q, sp, sd, sq) in enumerate(combos, start=1):
        if p == 0 and q == 0 and sp == 0 and sq == 0:
            continue  # no dynamics at all: not a time-series model

        order = (p, d, q)
        seasonal_order = (sp, sd, sq, seasonal_period)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = SARIMAX(
                    y_train,
                    exog=exog_train,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(**FIT_KWARGS)
            aic = float(res.aic)
            if not np.isfinite(aic):
                continue
            rows.append({"order": order, "seasonal_order": seasonal_order, "aic": aic})
        except Exception:
            # Non-convergence on an ill-posed combination is expected across a grid this
            # size; skip it rather than aborting the search.
            continue

        if verbose and i % 40 == 0:
            print(f"  ... {i}/{len(combos)} combinations evaluated", flush=True)

    if not rows:
        raise RuntimeError("SARIMAX grid search: no combination converged")

    table = pd.DataFrame(rows).sort_values("aic").reset_index(drop=True)
    best = table.iloc[0]
    return (
        SarimaxOrder(
            order=best["order"], seasonal_order=best["seasonal_order"], aic=best["aic"]
        ),
        table,
    )


def fit_and_forecast(
    df: pd.DataFrame,
    train_mask: pd.Series,
    spec: SarimaxOrder,
    exog_cols: list[str] | None = None,
    target_col: str = TARGET_COL,
) -> pd.Series:
    """Fit on train, then walk forward one step at a time over the remainder.

    Parameters are estimated on the training block only and then held fixed; the Kalman
    filter updates its state with observed values as it advances, so every prediction at t
    conditions on data through t-1 only. See decision 3.

    Args:
        df: Full feature table, sorted ascending by date.
        train_mask: Boolean mask selecting the training rows (a contiguous leading block).
        spec: The order selected by `grid_search_order`.
        exog_cols: Exog columns; defaults to EXOG_COLS.
        target_col: Target column name.

    Returns:
        One-step-ahead predictions aligned to `df`'s index. Training rows are filled with
        the model's in-sample one-step-ahead fit; validation/test rows are true
        out-of-sample forecasts.
    """
    y_full = df[target_col].astype("float64")
    exog_full = build_exog(df, exog_cols)

    y_train = y_full[train_mask]
    exog_train = exog_full[train_mask]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_train = SARIMAX(
            y_train,
            exog=exog_train,
            order=spec.order,
            seasonal_order=spec.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(**FIT_KWARGS)

        # Apply the TRAIN-ESTIMATED parameters to the full series without re-estimating.
        # .filter() runs the Kalman filter only -- it does not touch the likelihood
        # optimiser -- so no validation or test observation can influence a coefficient.
        full = SARIMAX(
            y_full,
            exog=exog_full,
            order=spec.order,
            seasonal_order=spec.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res_full = full.filter(res_train.params)

        # dynamic=False: each prediction uses observed history up to t-1, never its own
        # earlier predictions. This is what makes it one-step-ahead rather than a decaying
        # multi-step extrapolation.
        pred = res_full.get_prediction(dynamic=False)

    return pd.Series(
        np.asarray(pred.predicted_mean, dtype="float64"),
        index=df.index,
        name="sarimax",
    )
