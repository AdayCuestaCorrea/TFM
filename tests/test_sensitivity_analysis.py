"""Phase 5b tests: guard that each experiment changes exactly ONE thing.

The point of a sensitivity analysis is attributability. If Experiment A differed from
Phase 5 in both sample weights and hyperparameters, no observed difference could be
assigned to either, and the experiment would produce a number with no interpretation.
These tests pin that discipline mechanically rather than trusting the docstrings.
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.ensemble_baseline import (
    COMPONENTS,
    build_ensembles,
    equal_weights,
    inverse_mae_weights,
)
from src.models.hybrid_residual.xgboost_residual_weighted import (
    BASE_WEIGHT,
    PHASE5_PARAMS,
    UPWEIGHT_COLS,
    UPWEIGHT_VALUE,
    build_sample_weights,
    load_phase5_params,
)


# --------------------------------------------------------------------------------------
# Experiment A: sample_weight is the ONLY difference from Phase 5
# --------------------------------------------------------------------------------------


def test_hyperparameters_are_identical_to_phase5_artifact():
    """Loaded from the saved model, so drift in either side fails loudly."""
    assert load_phase5_params() == PHASE5_PARAMS


def test_declared_params_match_phase5_selection():
    assert PHASE5_PARAMS == {
        "max_depth": 5,
        "n_estimators": 300,
        "learning_rate": 0.01,
        "min_child_weight": 3,
    }


def test_weighted_model_uses_the_same_feature_matrix():
    """Same 552-row residual set, same 161 features -- no new data, no new exclusions."""
    from src.models.hybrid_residual.xgboost_residual import load_residual_training_set

    df = load_residual_training_set()
    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    assert len(df) == 552
    assert len(feat_cols) == 161
    assert sorted(df["fold"].unique().tolist()) == [2, 3, 4, 5]


def test_sample_weights_are_three_on_irregular_rows_and_one_elsewhere():
    df = pd.DataFrame(
        {
            "feat_day_type_festivo": [1, 0, 0, 0],
            "feat_day_type_domingo_festivo": [0, 1, 0, 0],
            "feat_is_bridge_day": [0, 0, 1, 0],
        }
    )
    np.testing.assert_array_equal(
        build_sample_weights(df),
        [UPWEIGHT_VALUE, UPWEIGHT_VALUE, UPWEIGHT_VALUE, BASE_WEIGHT],
    )


def test_sample_weight_takes_only_two_distinct_values():
    """A row flagged by two categories must not compound to 9.0."""
    df = pd.DataFrame(
        {
            "feat_day_type_festivo": [1, 0],
            "feat_day_type_domingo_festivo": [0, 0],
            "feat_is_bridge_day": [1, 0],  # first row is flagged twice
        }
    )
    weights = build_sample_weights(df)
    assert set(np.unique(weights)) <= {BASE_WEIGHT, UPWEIGHT_VALUE}
    assert weights[0] == UPWEIGHT_VALUE


def test_upweight_value_is_the_documented_a_priori_choice():
    """Pinned so a later 'small tweak' to this constant is a visible test change."""
    assert UPWEIGHT_VALUE == 3.0
    assert BASE_WEIGHT == 1.0


def test_upweight_columns_are_the_irregular_calendar_ones():
    assert UPWEIGHT_COLS == [
        "feat_day_type_festivo",
        "feat_day_type_domingo_festivo",
        "feat_is_bridge_day",
    ]


def test_sample_weights_raise_on_missing_column():
    with pytest.raises(ValueError, match="missing column"):
        build_sample_weights(pd.DataFrame({"feat_day_type_festivo": [1]}))


def test_weighted_hybrid_uses_the_same_additive_rule():
    from src.models.hybrid_residual.xgboost_residual_weighted import (
        WEIGHTED_PREDICTIONS_FILE,
    )

    if not WEIGHTED_PREDICTIONS_FILE.exists():
        pytest.skip("experiment A not run yet")

    out = pd.read_parquet(WEIGHTED_PREDICTIONS_FILE)
    np.testing.assert_allclose(
        out["y_pred_hybrid_weighted"],
        out["y_pred_lstm"] + out["e_pred_xgb_weighted"],
        rtol=1e-9,
    )


def test_weighted_hybrid_reuses_phase4_lstm_predictions_unchanged():
    """Stage 1 must be the saved artifact, not a re-run network."""
    from src.models.hybrid_residual.xgboost_residual_weighted import (
        WEIGHTED_PREDICTIONS_FILE,
    )
    from src.models.lstm.final_model import VAL_TEST_PREDICTIONS_FILE

    if not WEIGHTED_PREDICTIONS_FILE.exists():
        pytest.skip("experiment A not run yet")

    weighted = pd.read_parquet(WEIGHTED_PREDICTIONS_FILE)
    lstm = pd.read_parquet(VAL_TEST_PREDICTIONS_FILE)
    merged = weighted.merge(lstm, on="date", suffixes=("_w", "_orig"))
    np.testing.assert_allclose(
        merged["y_pred_lstm_w"], merged["y_pred_lstm_orig"], rtol=1e-12
    )


# --------------------------------------------------------------------------------------
# Experiment B: a documented function of saved predictions, nothing retrained
# --------------------------------------------------------------------------------------


def test_equal_weights_are_half_and_half():
    assert equal_weights() == {"sarimax": 0.5, "xgboost_alone": 0.5}


def test_equal_ensemble_is_the_arithmetic_mean_on_a_fixture():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-18", periods=3, freq="D"),
            "split": ["test"] * 3,
            "y_true": [100.0, 200.0, 300.0],
            "sarimax": [90.0, 210.0, 280.0],
            "xgboost_alone": [110.0, 190.0, 320.0],
        }
    )
    # inverse_mae_weights needs a val block; supply one.
    val = df.copy()
    val["split"] = "val"
    out, _ = build_ensembles(pd.concat([val, df], ignore_index=True))
    test_rows = out[out["split"] == "test"]
    np.testing.assert_allclose(test_rows["ensemble_equal"], [100.0, 200.0, 300.0])


def test_inverse_mae_weights_favour_the_more_accurate_component():
    val = pd.DataFrame(
        {
            "y_true": [100.0, 100.0, 100.0, 100.0],
            "sarimax": [110.0, 110.0, 110.0, 110.0],  # MAE 10
            "xgboost_alone": [105.0, 105.0, 105.0, 105.0],  # MAE 5 -> better
        }
    )
    weights = inverse_mae_weights(val)
    assert weights["xgboost_alone"] > weights["sarimax"]
    assert sum(weights.values()) == pytest.approx(1.0)
    # MAE 5 vs 10 -> weights 2:1
    assert weights["xgboost_alone"] / weights["sarimax"] == pytest.approx(2.0)


def test_ensemble_weights_sum_to_one():
    for weights in (equal_weights(),):
        assert sum(weights.values()) == pytest.approx(1.0)


def test_ensemble_is_a_pure_function_of_saved_component_predictions():
    """No retraining: the output must be reproducible from the inputs by arithmetic."""
    from src.evaluation.ensemble_baseline import ENSEMBLE_PREDICTIONS_FILE

    if not ENSEMBLE_PREDICTIONS_FILE.exists():
        pytest.skip("experiment B not run yet")

    out = pd.read_parquet(ENSEMBLE_PREDICTIONS_FILE)
    np.testing.assert_allclose(
        out["ensemble_equal"],
        0.5 * out["sarimax"] + 0.5 * out["xgboost_alone"],
        rtol=1e-9,
    )


def test_ensemble_lies_between_its_components_pointwise():
    """A convex combination cannot fall outside the range its components span."""
    from src.evaluation.ensemble_baseline import ENSEMBLE_PREDICTIONS_FILE

    if not ENSEMBLE_PREDICTIONS_FILE.exists():
        pytest.skip("experiment B not run yet")

    out = pd.read_parquet(ENSEMBLE_PREDICTIONS_FILE)
    lower = out[COMPONENTS].min(axis=1)
    upper = out[COMPONENTS].max(axis=1)
    for col in ("ensemble_equal", "ensemble_inverse_mae"):
        assert (out[col] >= lower - 1e-6).all()
        assert (out[col] <= upper + 1e-6).all()


def test_ensemble_components_are_the_two_phase5_leaders():
    assert COMPONENTS == ["sarimax", "xgboost_alone"]
