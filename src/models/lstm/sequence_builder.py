"""Fixed-length sequence construction for the Stage 1 LSTM.

UNIVARIATE BY DESIGN. Per CLAUDE.md, Stage 1 models the target series' own temporal
dynamics and nothing else; the exogenous features (weather, calendar, Fourier) are Stage
2's responsibility. Mixing them in here would blur the two stages and make the residual
decomposition uninterpretable -- the whole point of the architecture is that stage 2
corrects what stage 1 structurally cannot see.

WINDOW LENGTH W=28 (default, but always a parameter -- never hardcoded).
It matches the longest lag already used elsewhere in the pipeline, so the LSTM's memory
and the tabular features cover the same horizon, and it spans four complete weekly cycles.
The reference paper's W=72 is not transferable: that work is minute-resolution, where 72
steps is roughly an hour of a smooth physical signal. Here 72 steps is ten weeks of daily
data, and with ~860 usable training rows it would both consume a tenth of the history as
warm-up and leave far fewer effectively independent training sequences. W is compared
empirically over [14, 28, 60] in window_comparison.py rather than asserted.

STRICT CHRONOLOGY. The sequence at position t is values[t-W:t] predicting values[t] --
the window ends at t-1 and never contains values[t]. Sequences are emitted in ascending
target order and are never shuffled here; any shuffling would have to be an explicit
caller decision, and for a walk-forward design it never is.
"""

import numpy as np

DEFAULT_WINDOW = 28


def build_sequences(
    values: np.ndarray,
    window: int = DEFAULT_WINDOW,
    target_positions: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (X, y, target_positions) sliding windows over a 1-D series.

    Args:
        values: The full 1-D series, in chronological order. For OOF folds this is the
            scaled series; the caller is responsible for having fit the scaler on training
            data only.
        window: Sequence length W. The window for target t spans [t-W, t).
        target_positions: Which indices of `values` to emit targets for. Defaults to every
            position with enough history, i.e. range(window, len(values)). Passing an
            explicit range is how a fold predicts a held-out forward block while drawing
            its input history from earlier observed values.

    Returns:
        X of shape (n_sequences, window, 1), y of shape (n_sequences,), and the target
        positions actually used, all aligned and in ascending chronological order.

    Raises:
        ValueError: on a non-1-D input, a window that is not positive, a series shorter
            than the window, or a requested target lacking full history.
    """
    series = np.asarray(values, dtype="float64").ravel()

    if series.ndim != 1:
        raise ValueError("build_sequences: values must be 1-D")
    if window < 1:
        raise ValueError(f"build_sequences: window must be >= 1, got {window}")
    if len(series) <= window:
        raise ValueError(
            f"build_sequences: series length {len(series)} must exceed window {window}"
        )

    if target_positions is None:
        positions = np.arange(window, len(series), dtype="int64")
    else:
        positions = np.asarray(target_positions, dtype="int64").ravel()
        if positions.size == 0:
            raise ValueError("build_sequences: target_positions is empty")
        if positions.min() < window:
            raise ValueError(
                f"build_sequences: target position {int(positions.min())} has fewer than "
                f"{window} prior observations; it cannot be given a full window."
            )
        if positions.max() >= len(series):
            raise ValueError(
                f"build_sequences: target position {int(positions.max())} is outside the "
                f"series (length {len(series)})"
            )
        if not np.all(np.diff(positions) > 0):
            raise ValueError(
                "build_sequences: target_positions must be strictly increasing; "
                "out-of-order targets would silently break chronology."
            )

    # Vectorised gather: row i spans [t_i - window, t_i), so values[t_i] is excluded by
    # construction rather than by a boundary condition that could be edited away.
    offsets = np.arange(-window, 0, dtype="int64")
    index_matrix = positions[:, None] + offsets[None, :]
    X = series[index_matrix][..., np.newaxis]
    y = series[positions]

    return X, y, positions


def usable_target_positions(n_values: int, window: int = DEFAULT_WINDOW) -> np.ndarray:
    """Positions that have a complete window of history available."""
    return np.arange(window, n_values, dtype="int64")


def build_exog_matrix(features: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Gather row-*t* exogenous vectors aligned to the target positions of `build_sequences`.

    Phase 8 informational-parity DIAGNOSTIC only. `build_sequences` is deliberately NOT
    touched -- its univariate-by-design contract stands. This helper returns, for each
    target position *t*, that day's own feature vector (row *t*, NOT the window that
    excludes it), so the parity LSTM sees exactly the `feat_`-prefixed row XGBoost sees.

    Args:
        features: 2-D array (n_rows, k), aligned index-for-index with the `values` series
            passed to `build_sequences`.
        positions: The target positions returned by `build_sequences`, in ascending order.

    Returns:
        Array of shape (len(positions), k), row *i* being `features[positions[i]]`.
    """
    matrix = np.asarray(features, dtype="float64")
    if matrix.ndim != 2:
        raise ValueError("build_exog_matrix: features must be 2-D (n_rows, k)")
    idx = np.asarray(positions, dtype="int64").ravel()
    if idx.size == 0:
        raise ValueError("build_exog_matrix: positions is empty")
    if idx.min() < 0 or idx.max() >= len(matrix):
        raise ValueError("build_exog_matrix: a position is outside the feature matrix")
    return matrix[idx]
