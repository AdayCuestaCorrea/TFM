"""Shared XGBoost hyperparameter search, used identically by both Stage 2 models.

Factored out deliberately: the residual model and the XGBoost-alone baseline exist to be
COMPARED, so they must be tuned by the same procedure over the same grid. If one were
tuned more aggressively than the other, the comparison would measure tuning effort rather
than architecture, which is precisely the claim Phase 5 is supposed to test.

SELECTION USES A CHRONOLOGICAL INTERNAL HOLDOUT -- the last 15% of the model's own training
data by date. Three properties matter:
  - Chronological, not random: a random split would let the model select hyperparameters
    using future rows, the same leakage the whole pipeline is built to avoid.
  - Internal: carved out of training data only. It is NOT the official val split from
    src/utils/splits.py. Tuning on the official val split would consume it as a tuning
    signal and make every reported val metric optimistic.
  - Refit afterwards: the selected configuration is retrained on the FULL training set, so
    no rows are wasted on a permanent holdout.

The grid is modest on purpose. With ~552 residual rows and 161 features, the binding risk
is overfitting, which is why min_child_weight is included as an explicit regulariser
rather than left at its default of 1.
"""

from dataclasses import dataclass, field
from itertools import product

import numpy as np
import pandas as pd

from src.evaluation.metrics import mae, rmse
from src.utils.seed import GLOBAL_SEED

MAX_DEPTH = [3, 4, 5]
N_ESTIMATORS = [100, 300]
LEARNING_RATE = [0.01, 0.05, 0.1]
MIN_CHILD_WEIGHT = [3, 5, 10]

INTERNAL_HOLDOUT_FRACTION = 0.15


@dataclass
class TuningResult:
    """Selected configuration and the holdout scores that drove the choice."""

    params: dict
    holdout_mae: float
    holdout_rmse: float
    holdout_start: pd.Timestamp
    holdout_end: pd.Timestamp
    n_fit: int
    n_holdout: int
    n_candidates: int
    leaderboard: pd.DataFrame = field(default_factory=pd.DataFrame)

    def describe(self) -> str:
        p = self.params
        return (
            f"max_depth={p['max_depth']}, n_estimators={p['n_estimators']}, "
            f"learning_rate={p['learning_rate']}, min_child_weight={p['min_child_weight']}"
        )


def build_model(params: dict, seed: int = GLOBAL_SEED):
    """Construct an XGBRegressor with the shared fixed settings."""
    from xgboost import XGBRegressor

    return XGBRegressor(
        objective="reg:squarederror",
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
        **params,
    )


def grid_search(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    holdout_fraction: float = INTERNAL_HOLDOUT_FRACTION,
    seed: int = GLOBAL_SEED,
    verbose: bool = True,
) -> TuningResult:
    """Select hyperparameters on a chronological internal holdout.

    Args:
        X: Feature matrix, ordered by date ascending.
        y: Target (residual, or total for the alone-model).
        dates: Date column aligned to X, used only for reporting the holdout window.

    Returns:
        TuningResult with the winning parameters and a full leaderboard.
    """
    n = len(X)
    n_holdout = max(1, int(round(n * holdout_fraction)))
    split = n - n_holdout

    X_fit, X_hold = X.iloc[:split], X.iloc[split:]
    y_fit, y_hold = y.iloc[:split], y.iloc[split:]

    combos = list(product(MAX_DEPTH, N_ESTIMATORS, LEARNING_RATE, MIN_CHILD_WEIGHT))
    rows = []

    for depth, n_est, lr, mcw in combos:
        params = {
            "max_depth": depth,
            "n_estimators": n_est,
            "learning_rate": lr,
            "min_child_weight": mcw,
        }
        model = build_model(params, seed)
        model.fit(X_fit, y_fit)
        pred = model.predict(X_hold)
        rows.append(
            {**params, "holdout_mae": mae(y_hold, pred), "holdout_rmse": rmse(y_hold, pred)}
        )

    leaderboard = pd.DataFrame(rows).sort_values("holdout_mae").reset_index(drop=True)
    best = leaderboard.iloc[0]
    params = {
        "max_depth": int(best["max_depth"]),
        "n_estimators": int(best["n_estimators"]),
        "learning_rate": float(best["learning_rate"]),
        "min_child_weight": int(best["min_child_weight"]),
    }

    result = TuningResult(
        params=params,
        holdout_mae=float(best["holdout_mae"]),
        holdout_rmse=float(best["holdout_rmse"]),
        holdout_start=dates.iloc[split],
        holdout_end=dates.iloc[-1],
        n_fit=split,
        n_holdout=n_holdout,
        n_candidates=len(combos),
        leaderboard=leaderboard,
    )

    if verbose:
        print(f"  candidates evaluated : {result.n_candidates}")
        print(
            f"  internal holdout     : {result.holdout_start:%Y-%m-%d} -> "
            f"{result.holdout_end:%Y-%m-%d} ({result.n_holdout} rows; "
            f"{result.n_fit} used to fit)"
        )
        print("                         ^ carved from TRAINING; not the official val split")
        print(f"  selected             : {result.describe()}")
        print(
            f"  holdout MAE / RMSE   : {result.holdout_mae:,.0f} / "
            f"{result.holdout_rmse:,.0f}"
        )

    return result


def fit_final(X: pd.DataFrame, y: pd.Series, params: dict, seed: int = GLOBAL_SEED):
    """Retrain the selected configuration on the full training set."""
    model = build_model(params, seed)
    model.fit(X, y)
    return model


def feature_importance(model, feature_names: list[str], top_n: int = 15) -> pd.DataFrame:
    """Top-N gain-based importances, for interpreting what stage 2 actually uses."""
    importances = np.asarray(model.feature_importances_, dtype="float64")
    table = pd.DataFrame({"feature": feature_names, "importance": importances})
    return table.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)
