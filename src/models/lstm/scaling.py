"""Target scaling for the LSTM, fit on training data only.

CLAUDE.md Data Leakage Rule 3: scalers are fit ONLY on the training set and applied with
transform to validation and test. `TargetScaler.fit` therefore takes the training slice
explicitly rather than a full series with a split argument -- an API that cannot be handed
the whole dataset by accident is safer than one that documents you shouldn't.

MinMax is the default. For a strictly positive, bounded-range demand series it maps cleanly
onto the LSTM's activation range, and unlike standardisation it does not depend on the
training distribution being roughly symmetric.

NOTE ON OUT-OF-RANGE VALUES: transformed validation/test values may fall slightly outside
[0, 1] whenever they exceed the training min/max. That is correct and must NOT be clipped
-- clipping would silently distort exactly the extreme days the model most needs to get
right, and would leak the training range into the evaluation data as a ceiling.
"""

from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class TargetScaler:
    """Fit/transform/inverse_transform wrapper around a 1-D scaler, plus persistence."""

    def __init__(self, kind: str = "minmax") -> None:
        if kind not in {"minmax", "standard"}:
            raise ValueError(f"TargetScaler: unknown kind {kind!r}")
        self.kind = kind
        self._scaler = MinMaxScaler() if kind == "minmax" else StandardScaler()
        self._fitted = False

    def fit(self, train_values: np.ndarray) -> "TargetScaler":
        """Fit on the TRAINING slice only. Never pass the full series here."""
        values = np.asarray(train_values, dtype="float64").reshape(-1, 1)
        if values.size == 0:
            raise ValueError("TargetScaler.fit: empty training values")
        if np.isnan(values).any():
            raise ValueError("TargetScaler.fit: training values contain NaN")
        self._scaler.fit(values)
        self._fitted = True
        return self

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise ValueError("TargetScaler: must call fit() on training data first")

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Apply the fitted scaling. Used for train, validation and test alike."""
        self._check_fitted()
        arr = np.asarray(values, dtype="float64").reshape(-1, 1)
        return self._scaler.transform(arr).ravel()

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        """Map scaled predictions back to passenger counts."""
        self._check_fitted()
        arr = np.asarray(values, dtype="float64").reshape(-1, 1)
        return self._scaler.inverse_transform(arr).ravel()

    def save(self, path: Path | str) -> Path:
        """Persist so Phase 5 inverts predictions with the identical fitted scaler."""
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"kind": self.kind, "scaler": self._scaler}, path)
        return path

    @classmethod
    def load(cls, path: Path | str) -> "TargetScaler":
        payload = joblib.load(Path(path))
        instance = cls(kind=payload["kind"])
        instance._scaler = payload["scaler"]
        instance._fitted = True
        return instance


class FeatureScaler:
    """Multi-column MinMax for the Phase 8 informational-parity exogenous vector.

    DIAGNOSTIC only -- not persisted (no `save`/`load`), so the Phase 4
    `lstm_stage1_scaler.joblib` contract is untouched. Fit on training rows only
    (Leakage Rule 3). A constant column -- `feat_day_type_domingo_festivo` is all-zero
    across the whole project (Phase 1: 0 observations) -- maps to 0.0 with no NaN/inf,
    which `test_feature_scaler_handles_a_constant_column` pins.
    """

    def __init__(self) -> None:
        self._scaler = MinMaxScaler()
        self._fitted = False

    def fit(self, train_rows: np.ndarray) -> "FeatureScaler":
        values = np.asarray(train_rows, dtype="float64")
        if values.ndim != 2:
            raise ValueError("FeatureScaler.fit: expected a 2-D (n_rows, k) array")
        if values.size == 0:
            raise ValueError("FeatureScaler.fit: empty training rows")
        if np.isnan(values).any():
            raise ValueError("FeatureScaler.fit: training rows contain NaN")
        self._scaler.fit(values)
        self._fitted = True
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise ValueError("FeatureScaler: must call fit() on training rows first")
        arr = np.asarray(values, dtype="float64")
        scaled = self._scaler.transform(arr)
        # A constant column has range 0; sklearn maps it to 0.0 cleanly, but guard anyway.
        return np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
