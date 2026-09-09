"""Phase 6 tests: the dashboard must not disagree with the numbers already reported.

The recurring risk in a reporting layer is silent divergence -- a reshaping bug that makes
the figures show something subtly different from the tables in CLAUDE.md, with nothing
crashing. These tests re-derive the dashboard's outputs independently and compare.
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.dashboard_data import (
    FEATURE_GROUPS,
    MODEL_FAMILY,
    MODEL_NAMES,
    SMALL_SAMPLE_THRESHOLD,
    SPLITS,
    actual_vs_predicted,
    available_models,
    classify_feature,
    error_by_day_type,
    feature_importance_by_group,
    load_all_predictions,
    model_comparison_table,
    residual_distribution,
)
from src.evaluation.metrics import mae, mape, r2, rmse

VAL_TEST_ONLY = {"lstm_alone", "hybrid", "hybrid_weighted"}


@pytest.fixture(scope="session")
def predictions() -> pd.DataFrame:
    return load_all_predictions()


# --------------------------------------------------------------------------------------
# load_all_predictions
# --------------------------------------------------------------------------------------


def test_returns_expected_model_set(predictions):
    assert set(predictions["model"].unique()) == set(MODEL_NAMES)


def test_tidy_schema(predictions):
    assert list(predictions.columns) == ["date", "split", "model", "y_true", "y_pred"]


def test_no_duplicate_date_model_pairs(predictions):
    assert not predictions.duplicated(subset=["date", "model"]).any()


def test_every_row_has_a_valid_split(predictions):
    assert set(predictions["split"].unique()) <= set(SPLITS)
    assert "unassigned" not in set(predictions["split"].unique())
    assert predictions["split"].isna().sum() == 0


def test_no_null_predictions_survive(predictions):
    assert predictions["y_pred"].isna().sum() == 0
    assert predictions["y_true"].isna().sum() == 0


def test_lstm_family_covers_val_and_test_only(predictions):
    """Documented coverage gap: Phase 4 saved stage-1 predictions for val/test only."""
    for model in VAL_TEST_ONLY:
        splits = set(predictions.loc[predictions["model"] == model, "split"].unique())
        assert splits == {"val", "test"}, f"{model} covers {splits}"


def test_single_stage_models_cover_all_splits(predictions):
    for model in ("sarimax", "xgboost_alone", "ensemble_equal"):
        splits = set(predictions.loc[predictions["model"] == model, "split"].unique())
        assert splits == set(SPLITS), f"{model} covers {splits}"


def test_y_true_is_consistent_across_models_on_a_date(predictions):
    """Every model on a given date must be scored against the same actual value."""
    spread = predictions.groupby("date")["y_true"].nunique()
    assert (spread == 1).all()


def test_available_models_respects_coverage(predictions):
    assert set(available_models("test", predictions)) == set(MODEL_NAMES)
    train_models = set(available_models("train", predictions))
    assert not (train_models & VAL_TEST_ONLY)


def test_every_model_has_a_declared_family():
    assert set(MODEL_FAMILY) == set(MODEL_NAMES)


# --------------------------------------------------------------------------------------
# model_comparison_table must match metrics.py computed directly
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("split", SPLITS)
def test_comparison_table_matches_direct_metric_computation(split, predictions):
    table = model_comparison_table(split, predictions).set_index("model")
    subset = predictions[predictions["split"] == split]

    for model in table.index:
        pair = subset[subset["model"] == model]
        assert table.loc[model, "MAE"] == pytest.approx(mae(pair["y_true"], pair["y_pred"]))
        assert table.loc[model, "RMSE"] == pytest.approx(rmse(pair["y_true"], pair["y_pred"]))
        assert table.loc[model, "MAPE"] == pytest.approx(mape(pair["y_true"], pair["y_pred"]))
        assert table.loc[model, "R2"] == pytest.approx(r2(pair["y_true"], pair["y_pred"]))
        assert table.loc[model, "n"] == len(pair)


def test_comparison_table_reproduces_reported_test_numbers(predictions):
    """Pinned against the figures recorded in CLAUDE.md for Phase 5/5b."""
    table = model_comparison_table("test", predictions).set_index("model")
    assert table.loc["ensemble_equal", "MAE"] == pytest.approx(139_725, abs=1)
    assert table.loc["xgboost_alone", "MAE"] == pytest.approx(156_121, abs=1)
    assert table.loc["sarimax", "MAE"] == pytest.approx(163_627, abs=1)
    assert table.loc["hybrid", "MAE"] == pytest.approx(234_223, abs=1)
    assert table.loc["lstm_alone", "MAE"] == pytest.approx(280_952, abs=1)


def test_comparison_table_is_sorted_best_first(predictions):
    table = model_comparison_table("test", predictions)
    assert table["MAE"].is_monotonic_increasing
    assert table.iloc[0]["model"].startswith("ensemble")


def test_comparison_table_rejects_bad_split():
    with pytest.raises(ValueError, match="split must be one of"):
        model_comparison_table("holdout")


# --------------------------------------------------------------------------------------
# error_by_day_type
# --------------------------------------------------------------------------------------


def test_day_type_counts_match_value_counts(predictions):
    """Guard against silent misalignment in the features join."""
    from src.evaluation.dashboard_data import FEATURES_DAILY_FILE

    table = error_by_day_type("test", predictions)
    feats = pd.read_parquet(FEATURES_DAILY_FILE)

    test_dates = set(predictions.loc[predictions["split"] == "test", "date"])
    expected = (
        feats[feats["date"].isin(test_dates)]["day_type"].astype(str).value_counts()
    )

    for model in table["model"].unique():
        rows = table[table["model"] == model].set_index("group")
        for day_type, n in expected.items():
            if day_type in rows.index:
                assert rows.loc[day_type, "n"] == n, f"{model}/{day_type}"


def test_day_type_covers_every_model(predictions):
    table = error_by_day_type("test", predictions)
    assert set(table["model"].unique()) == set(MODEL_NAMES)


def test_small_sample_flag_marks_festivo_and_bridge_on_test(predictions):
    """CLAUDE.md caveat: festivo n=5 and bridge n=3 are indicative only."""
    table = error_by_day_type("test", predictions)
    flagged = table[table["small_sample"]]["group"].unique()
    assert "festivo" in flagged
    assert "laborable, bridge day" in flagged
    assert "laborable" not in flagged


def test_small_sample_threshold_is_applied_consistently(predictions):
    table = error_by_day_type("test", predictions)
    assert (table["small_sample"] == (table["n"] < SMALL_SAMPLE_THRESHOLD)).all()


def test_day_type_mae_matches_direct_computation(predictions):
    from src.evaluation.dashboard_data import FEATURES_DAILY_FILE

    table = error_by_day_type("test", predictions).set_index(["model", "group"])
    feats = pd.read_parquet(FEATURES_DAILY_FILE)[["date", "day_type"]]
    subset = predictions[predictions["split"] == "test"].merge(feats, on="date")

    rows = subset[(subset["model"] == "sarimax") & (subset["day_type"].astype(str) == "festivo")]
    assert table.loc[("sarimax", "festivo"), "MAE"] == pytest.approx(
        mae(rows["y_true"], rows["y_pred"])
    )


# --------------------------------------------------------------------------------------
# residual_distribution
# --------------------------------------------------------------------------------------


def test_residuals_equal_truth_minus_prediction(predictions):
    result = residual_distribution("sarimax", "test", predictions)
    rows = predictions[(predictions["model"] == "sarimax") & (predictions["split"] == "test")]
    np.testing.assert_allclose(
        result["test"].to_numpy(), (rows["y_true"] - rows["y_pred"]).to_numpy()
    )


def test_residuals_include_train_overlay_when_available(predictions):
    result = residual_distribution("xgboost_alone", "test", predictions)
    assert set(result) == {"test", "train"}
    # The overfit gap: train residuals are far tighter than test residuals.
    assert result["train"].abs().mean() < result["test"].abs().mean()


def test_residuals_omit_train_overlay_when_model_lacks_train_coverage(predictions):
    result = residual_distribution("hybrid", "test", predictions)
    assert set(result) == {"test"}


def test_residual_distribution_rejects_unknown_model():
    with pytest.raises(ValueError, match="unknown model"):
        residual_distribution("nonexistent", "test")


def test_residual_distribution_rejects_model_absent_from_split(predictions):
    with pytest.raises(ValueError, match="no predictions on split"):
        residual_distribution("hybrid", "train", predictions)


# --------------------------------------------------------------------------------------
# feature_importance_by_group
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("model", ["xgboost_residual", "xgboost_alone"])
def test_group_sums_equal_raw_total_importance(model):
    """No feature silently dropped or double-counted by the grouping."""
    import joblib

    from src.evaluation.dashboard_data import MODEL_ARTIFACTS

    payload = joblib.load(MODEL_ARTIFACTS[model])
    raw_total = float(np.sum(payload["model"].feature_importances_))

    result = feature_importance_by_group(model)
    # rel=1e-6, not tighter: XGBoost returns float32 importances, so summing 161 of them
    # in group order vs. flat order diverges at ~1e-7 relative purely from accumulation.
    # A tolerance below that tests float32 associativity, not the grouping. The stricter
    # guarantee -- that no feature is dropped or double-counted -- is exact and is asserted
    # by test_group_feature_counts_sum_to_the_full_matrix.
    assert result["by_group"]["importance"].sum() == pytest.approx(raw_total, rel=1e-6)


@pytest.mark.parametrize("model", ["xgboost_residual", "xgboost_alone"])
def test_group_feature_counts_sum_to_the_full_matrix(model):
    import joblib

    from src.evaluation.dashboard_data import MODEL_ARTIFACTS

    payload = joblib.load(MODEL_ARTIFACTS[model])
    result = feature_importance_by_group(model)
    assert result["by_group"]["n_features"].sum() == len(payload["features"])


@pytest.mark.parametrize("model", ["xgboost_residual", "xgboost_alone"])
def test_every_feature_classifies_to_a_known_group(model):
    import joblib

    from src.evaluation.dashboard_data import MODEL_ARTIFACTS

    payload = joblib.load(MODEL_ARTIFACTS[model])
    for feature in payload["features"]:
        assert classify_feature(feature) in FEATURE_GROUPS


def test_classify_feature_examples():
    assert classify_feature("feat_total_lag_7") == "lag_total"
    assert classify_feature("feat_metro_lag_1") == "lag_operator"
    assert classify_feature("feat_total_roll_mean_28") == "rolling"
    assert classify_feature("feat_fourier_weekly_k1_sin") == "fourier_weekly"
    assert classify_feature("feat_fourier_annual_k2_cos") == "fourier_annual"
    assert classify_feature("feat_day_type_festivo") == "calendar"
    assert classify_feature("feat_is_bridge_day") == "calendar"
    assert classify_feature("feat_temperature_2m_mean_lag_3") == "weather"
    assert classify_feature("feat_weather_code_roll_mean_7") == "weather"


def test_classify_feature_raises_on_unmapped_name():
    with pytest.raises(ValueError, match="matches no group"):
        classify_feature("feat_something_invented")


def test_top_features_are_sorted_and_limited():
    result = feature_importance_by_group("xgboost_residual", top_n=15)
    top = result["top_features"]
    assert len(top) == 15
    assert top["importance"].is_monotonic_decreasing


def test_group_shares_sum_to_one():
    result = feature_importance_by_group("xgboost_alone")
    assert result["by_group"]["share"].sum() == pytest.approx(1.0)


def test_feature_importance_rejects_unknown_model():
    with pytest.raises(ValueError, match="unknown model"):
        feature_importance_by_group("random_forest")


# --------------------------------------------------------------------------------------
# actual_vs_predicted
# --------------------------------------------------------------------------------------


def test_actual_vs_predicted_is_chronological(predictions):
    series = actual_vs_predicted("ensemble_equal", "test", predictions)
    assert series["date"].is_monotonic_increasing
    assert len(series) == 197


def test_actual_vs_predicted_rejects_missing_combination(predictions):
    with pytest.raises(ValueError, match="no rows"):
        actual_vs_predicted("hybrid", "train", predictions)


# --------------------------------------------------------------------------------------
# Missing-artifact error messages
# --------------------------------------------------------------------------------------


def test_missing_artifact_names_the_producing_phase(tmp_path):
    from src.evaluation.dashboard_data import _require

    with pytest.raises(FileNotFoundError, match="Produced by"):
        _require(tmp_path / "does_not_exist.parquet")
