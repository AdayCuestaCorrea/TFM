"""Forecast error metrics and the model comparison table.

Pure functions over (y_true, y_pred) arrays: no model objects, no I/O, no global state, so
every model in Phases 3-5 is scored by identical code and the comparison table cannot drift
between phases.
"""

import numpy as np
import pandas as pd

ArrayLike = np.ndarray | pd.Series | list[float]


def _as_pair(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Coerce to float arrays, validate shapes, and drop pairs with NaN on either side."""
    true = np.asarray(y_true, dtype="float64").ravel()
    pred = np.asarray(y_pred, dtype="float64").ravel()

    if true.shape != pred.shape:
        raise ValueError(f"shape mismatch: y_true {true.shape} vs y_pred {pred.shape}")
    if true.size == 0:
        raise ValueError("empty input: no observations to score")

    # A NaN prediction must not silently poison the metric into NaN; drop the pair and let
    # the caller see the reduced count via `n` in the comparison table.
    valid = ~(np.isnan(true) | np.isnan(pred))
    if not valid.any():
        raise ValueError("no valid (non-NaN) observation pairs to score")
    return true[valid], pred[valid]


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean absolute error, in passengers/day."""
    true, pred = _as_pair(y_true, y_pred)
    return float(np.mean(np.abs(true - pred)))


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Root mean squared error, in passengers/day. Penalises large misses more than MAE."""
    true, pred = _as_pair(y_true, y_pred)
    return float(np.sqrt(np.mean((true - pred) ** 2)))


def mape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean absolute percentage error (%).

    Undefined where y_true == 0. Daily demand is never zero in this dataset, but rather
    than divide and emit inf, zero-valued actuals are excluded and would show up as a
    reduced sample -- silent infinities are harder to notice than a missing row.
    """
    true, pred = _as_pair(y_true, y_pred)
    nonzero = true != 0
    if not nonzero.any():
        raise ValueError("MAPE undefined: every y_true value is zero")
    return float(np.mean(np.abs((true[nonzero] - pred[nonzero]) / true[nonzero])) * 100.0)


def r2(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Coefficient of determination.

    R² = 0 is the variance of the evaluation window itself, NOT a naive baseline. On a
    heavily weekly-cyclical series a persistence model can post a respectable R² purely by
    tracking the cycle, so read it alongside MAE rather than as a standalone verdict.
    """
    true, pred = _as_pair(y_true, y_pred)
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - np.mean(true)) ** 2))
    if ss_tot == 0:
        raise ValueError("R² undefined: y_true is constant over the evaluation window")
    return 1.0 - ss_res / ss_tot


def all_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    """Every metric at once, for one (y_true, y_pred) pair."""
    true, pred = _as_pair(y_true, y_pred)
    return {
        "n": float(true.size),
        "MAE": mae(true, pred),
        "RMSE": rmse(true, pred),
        "MAPE": mape(true, pred),
        "R2": r2(true, pred),
    }


def comparison_table(
    results: dict[str, tuple[ArrayLike, ArrayLike]],
    descriptions: dict[str, str] | None = None,
    sort_by: str | None = "RMSE",
) -> pd.DataFrame:
    """Build the model comparison table.

    Follows the layout of Table 1 in the reference paper (model name, description, RMSE,
    MAE) and extends it with MAPE and R² as required by the CLAUDE.md acceptance criteria.

    Args:
        results: {model_name: (y_true, y_pred)}.
        descriptions: Optional {model_name: short description} for the second column.
        sort_by: Metric to sort ascending by ('R2' sorts descending, since higher is
            better). None preserves insertion order.

    Returns:
        DataFrame with columns [Model, Description, RMSE, MAE, MAPE, R2, n].
    """
    descriptions = descriptions or {}

    rows = []
    for name, (y_true, y_pred) in results.items():
        scores = all_metrics(y_true, y_pred)
        rows.append(
            {
                "Model": name,
                "Description": descriptions.get(name, ""),
                "RMSE": scores["RMSE"],
                "MAE": scores["MAE"],
                "MAPE": scores["MAPE"],
                "R2": scores["R2"],
                "n": int(scores["n"]),
            }
        )

    table = pd.DataFrame(rows)
    if sort_by and sort_by in table.columns:
        table = table.sort_values(sort_by, ascending=(sort_by != "R2"))
    return table.reset_index(drop=True)


def format_table(table: pd.DataFrame) -> str:
    """Render a comparison table as fixed-width text for the console and the memoria."""
    display = table.copy()
    for col in ("RMSE", "MAE"):
        if col in display.columns:
            display[col] = display[col].map(lambda v: f"{v:,.0f}")
    if "MAPE" in display.columns:
        display["MAPE"] = display["MAPE"].map(lambda v: f"{v:.2f}%")
    if "R2" in display.columns:
        display["R2"] = display["R2"].map(lambda v: f"{v:.4f}")
    return display.to_string(index=False)
