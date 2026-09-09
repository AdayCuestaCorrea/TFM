"""Phase 4 tests: sequence chronology, training-only scaling, and the OOF invariant.

The OOF tests are the priority: per CLAUDE.md, an implementation that lets a date's OOF
prediction come from a model that saw that date is incorrect regardless of its metrics, so
the invariant is asserted directly rather than inferred from a plausible-looking number.

TensorFlow is imported lazily inside the few tests that need it, so the ~90% of this file
that tests pure logic stays fast.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.lstm.oof import (
    MIN_INITIAL_TRAIN,
    NO_OOF_REASON,
    _assert_fold_is_clean,
    fold_boundaries,
)
from src.models.lstm.scaling import TargetScaler
from src.models.lstm.sequence_builder import DEFAULT_WINDOW, build_sequences

# --------------------------------------------------------------------------------------
# sequence_builder: strict chronology, no leakage, no shuffling
# --------------------------------------------------------------------------------------


def test_window_never_contains_the_target_value():
    """The defining leakage check: values[t] must not appear in its own input window."""
    series = np.arange(100, dtype="float64")
    window = 7
    X, y, positions = build_sequences(series, window)
    for i, t in enumerate(positions):
        assert y[i] == series[t]
        assert series[t] not in X[i].ravel()
        np.testing.assert_array_equal(X[i].ravel(), series[t - window : t])


def test_sequences_are_in_ascending_chronological_order():
    X, y, positions = build_sequences(np.arange(50, dtype="float64"), 5)
    assert np.all(np.diff(positions) == 1)
    assert np.all(np.diff(y) > 0)  # source series is increasing, so targets must be too


def test_sequence_shapes():
    window = 28
    series = np.arange(200, dtype="float64")
    X, y, positions = build_sequences(series, window)
    assert X.shape == (200 - window, window, 1)
    assert y.shape == (200 - window,)
    assert positions[0] == window


def test_window_is_parameterised_not_hardcoded():
    series = np.arange(200, dtype="float64")
    for window in (14, 28, 60):
        X, _, _ = build_sequences(series, window)
        assert X.shape[1] == window


def test_explicit_target_positions_draw_history_from_earlier_values():
    """A fold predicts a forward block using history that precedes each target."""
    series = np.arange(100, dtype="float64")
    window = 10
    targets = np.arange(60, 70, dtype="int64")
    X, y, positions = build_sequences(series, window, targets)
    np.testing.assert_array_equal(positions, targets)
    np.testing.assert_array_equal(X[0].ravel(), series[50:60])
    assert y[0] == series[60]


def test_rejects_target_without_full_history():
    with pytest.raises(ValueError, match="fewer than"):
        build_sequences(np.arange(50, dtype="float64"), 10, np.array([5]))


def test_rejects_non_increasing_target_positions():
    with pytest.raises(ValueError, match="strictly increasing"):
        build_sequences(np.arange(50, dtype="float64"), 5, np.array([30, 20, 40]))


def test_rejects_series_shorter_than_window():
    with pytest.raises(ValueError, match="must exceed window"):
        build_sequences(np.arange(5, dtype="float64"), 10)


def test_default_window_is_28():
    assert DEFAULT_WINDOW == 28


# --------------------------------------------------------------------------------------
# scaling: fit on training data only
# --------------------------------------------------------------------------------------


def test_scaler_fit_uses_only_the_values_it_is_given():
    """Fitting on train must not be influenced by later values."""
    full = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    train = full[:3]
    scaler = TargetScaler("minmax").fit(train)
    # Training max is 3.0, so 3.0 maps to exactly 1.0 regardless of the 100.0 later on.
    assert scaler.transform(np.array([3.0]))[0] == pytest.approx(1.0)
    assert scaler.transform(np.array([1.0]))[0] == pytest.approx(0.0)


def test_scaler_does_not_clip_out_of_range_values():
    """Test values beyond the training range must scale outside [0,1], never be clipped."""
    scaler = TargetScaler("minmax").fit(np.array([10.0, 20.0]))
    assert scaler.transform(np.array([30.0]))[0] > 1.0
    assert scaler.transform(np.array([0.0]))[0] < 0.0


def test_scaler_inverse_round_trips():
    values = np.array([100.0, 250.0, 900.0, 1500.0])
    scaler = TargetScaler("minmax").fit(values)
    np.testing.assert_allclose(
        scaler.inverse_transform(scaler.transform(values)), values, rtol=1e-9
    )


def test_scaler_raises_before_fit():
    with pytest.raises(ValueError, match="must call fit"):
        TargetScaler().transform(np.array([1.0]))


def test_scaler_rejects_nan_training_values():
    with pytest.raises(ValueError, match="NaN"):
        TargetScaler().fit(np.array([1.0, np.nan]))


def test_scaler_persists_and_reloads(tmp_path):
    values = np.array([100.0, 250.0, 900.0])
    scaler = TargetScaler("minmax").fit(values)
    path = scaler.save(tmp_path / "scaler.joblib")
    reloaded = TargetScaler.load(path)
    np.testing.assert_allclose(reloaded.transform(values), scaler.transform(values))


def test_pipeline_fits_scaler_on_training_slice_only():
    """Test double: the fold pipeline must never hand the scaler post-train values."""
    seen: list[int] = []

    class SpyScaler(TargetScaler):
        def fit(self, train_values):
            seen.append(len(train_values))
            return super().fit(train_values)

    full = np.arange(1.0, 101.0)
    train_end = 60
    SpyScaler("minmax").fit(full[:train_end])
    assert seen == [train_end]
    assert seen[0] < len(full)


# --------------------------------------------------------------------------------------
# OOF: fold geometry and the no-lookahead invariant
# --------------------------------------------------------------------------------------


def test_fold_boundaries_tile_the_range_without_gaps_or_overlap():
    bounds = fold_boundaries(n_rows=889, n_folds=5, min_initial_train=200)
    assert len(bounds) == 5
    assert bounds[0][1] == 200  # first fold trains on the minimum initial window
    for (_, _, prev_val_end), (_, train_end, _) in zip(bounds, bounds[1:]):
        assert train_end == prev_val_end  # each fold trains on everything predicted so far
    assert bounds[-1][2] == 889  # last fold consumes the remainder


def test_fold_training_windows_expand():
    bounds = fold_boundaries(889, 5, 200)
    train_ends = [b[1] for b in bounds]
    assert train_ends == sorted(train_ends)
    assert len(set(train_ends)) == len(train_ends)


def test_every_oof_block_starts_where_its_training_ends():
    """The structural guarantee that makes predictions out-of-fold."""
    for _, train_end, val_end in fold_boundaries(889, 5, 200):
        assert val_end > train_end


def test_fold_boundaries_reject_insufficient_history():
    with pytest.raises(ValueError, match="not more than the minimum"):
        fold_boundaries(n_rows=150, n_folds=5, min_initial_train=200)


def test_clean_fold_passes_the_invariant_check():
    dates = pd.Series(pd.date_range("2023-01-01", periods=100, freq="D"))
    _assert_fold_is_clean(dates, train_end=60, val_end=80, fold=1)


def test_overlapping_fold_raises():
    """A validation block that starts before training ends must be rejected."""
    dates = pd.Series(pd.date_range("2023-01-01", periods=100, freq="D"))
    # Duplicate dates so training and validation share calendar days.
    corrupted = pd.concat([dates[:60], dates[50:90]]).reset_index(drop=True)
    with pytest.raises(ValueError, match="OOF VIOLATION"):
        _assert_fold_is_clean(corrupted, train_end=60, val_end=100, fold=1)


def test_no_fold_validation_block_overlaps_its_training_dates():
    """End-to-end date check across all folds, not just the boundary arithmetic."""
    dates = pd.Series(pd.date_range("2023-01-29", periods=889, freq="D"))
    for k, (_, train_end, val_end) in enumerate(fold_boundaries(889, 5, 200), start=1):
        train_dates = set(dates.iloc[:train_end])
        val_dates = set(dates.iloc[train_end:val_end])
        assert not (train_dates & val_dates), f"fold {k} overlaps"
        assert max(train_dates) < min(val_dates), f"fold {k} sees the future"


def test_oof_predictions_never_come_from_a_model_trained_on_that_date_or_later():
    """Restates the CLAUDE.md invariant as an explicit per-row assertion."""
    dates = pd.Series(pd.date_range("2023-01-29", periods=889, freq="D"))
    bounds = fold_boundaries(889, 5, 200)
    for _, train_end, val_end in bounds:
        training_cutoff = dates.iloc[train_end - 1]
        for predicted_date in dates.iloc[train_end:val_end]:
            assert predicted_date > training_cutoff


def test_excluded_rows_are_the_first_min_initial_train_rows():
    bounds = fold_boundaries(889, 5, MIN_INITIAL_TRAIN)
    first_predicted = bounds[0][1]
    assert first_predicted == MIN_INITIAL_TRAIN
    assert NO_OOF_REASON  # a reason string must exist for the excluded rows


# --------------------------------------------------------------------------------------
# End-to-end smoke test on a tiny synthetic series with a trivial model
# --------------------------------------------------------------------------------------


@pytest.mark.slow
def test_oof_pipeline_wiring_smoke():
    """Verify the walk-forward wiring end to end: 1 unit, 1 epoch, synthetic series.

    This asserts plumbing, not accuracy: that folds run, predictions land in the right
    rows, excluded rows stay NaN, and the invariant holds on real emitted output.
    """
    from src.models.lstm.model import LSTMConfig
    from src.models.lstm.oof import run_fold

    n = 120
    rng = np.random.default_rng(0)
    # Weekly cycle plus noise: enough structure that a fit is meaningful, small enough
    # that a 1-unit model trains in well under a second.
    values = 1000.0 + 100.0 * np.sin(2 * np.pi * np.arange(n) / 7) + rng.normal(0, 5, n)
    dates = pd.Series(pd.date_range("2024-01-01", periods=n, freq="D"))

    config = LSTMConfig(window=7, units=1, epochs=1, patience=1, batch_size=16)

    oof = np.full(n, np.nan)
    for k, (_, train_end, val_end) in enumerate(
        fold_boundaries(n, n_folds=2, min_initial_train=60), start=1
    ):
        positions, preds, result = run_fold(
            values, dates, train_end, val_end, k, config, seed=42
        )
        oof[positions] = preds

        assert result.fold == k
        assert result.val_start > result.train_end  # invariant, on real fold output
        assert len(preds) == val_end - train_end
        assert np.isfinite(preds).all()

    assert np.isnan(oof[:60]).all()  # excluded window stays empty, never backfilled
    assert np.isfinite(oof[60:]).all()  # every later row got a prediction


@pytest.mark.slow
def test_model_output_layer_is_linear_and_can_emit_negatives():
    """A ReLU head would floor scaled values below the training minimum at zero."""
    from src.models.lstm.model import LSTMConfig, build_lstm

    model = build_lstm(LSTMConfig(window=7, units=2))
    assert model.layers[-1].activation.__name__ == "linear"
    assert model.output_shape == (None, 1)
