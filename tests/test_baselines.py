"""Tests for the metrics, the baseline predictors, and the weather-feature promotion."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import all_metrics, comparison_table, mae, mape, r2, rmse
from src.features.naming import FEATURE_PREFIX
from src.features.weather_features import (
    WEATHER_COLS,
    WEATHER_LAGS,
    add_weather_features,
    weather_feature_names,
)
from src.models.baselines.moving_average import predict_moving_average
from src.models.baselines.persistence import (
    predict_persistence,
    predict_seasonal_naive,
)
from src.models.baselines.sarimax import (
    EXOG_COLS,
    assert_no_lag_features_in_exog,
    build_exog,
)


# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------


def _fixture(n: int = 20) -> pd.DataFrame:
    """Small frame with hand-checkable lag/rolling columns."""
    total = pd.Series(np.arange(1, n + 1, dtype="float64") * 100.0)
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "total": total,
            "feat_total_lag_1": total.shift(1),
            "feat_total_lag_7": total.shift(7),
            "feat_total_roll_mean_7": total.shift(1).rolling(7, min_periods=7).mean(),
            "feat_total_roll_mean_28": total.shift(1).rolling(28, min_periods=28).mean(),
        }
    )


def _weather_fixture(n: int = 20) -> pd.DataFrame:
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=n, freq="D")})
    for i, col in enumerate(WEATHER_COLS):
        df[col] = np.arange(n, dtype="float64") + i
    return df


# --------------------------------------------------------------------------------------
# Metrics against hand-computed / known reference values
# --------------------------------------------------------------------------------------


def test_perfect_prediction_gives_zero_error_and_unit_r2():
    y = np.array([10.0, 20.0, 30.0, 40.0])
    assert mae(y, y) == 0.0
    assert rmse(y, y) == 0.0
    assert mape(y, y) == 0.0
    assert r2(y, y) == 1.0


def test_mean_prediction_gives_r2_zero():
    """Predicting the mean is exactly the R² = 0 reference point."""
    y = np.array([10.0, 20.0, 30.0, 40.0])
    assert r2(y, np.full_like(y, y.mean())) == pytest.approx(0.0)


def test_worse_than_mean_gives_negative_r2():
    y = np.array([10.0, 20.0, 30.0, 40.0])
    assert r2(y, np.full_like(y, 1000.0)) < 0


def test_mae_and_rmse_hand_computed():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([2.0, 2.0, 2.0, 2.0])
    # errors: 1, 0, 1, 2  -> MAE = 4/4 = 1.0 ; RMSE = sqrt((1+0+1+4)/4) = sqrt(1.5)
    assert mae(y_true, y_pred) == pytest.approx(1.0)
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(1.5))


def test_rmse_penalises_outliers_more_than_mae():
    y_true = np.array([0.0, 0.0, 0.0, 0.0])
    spread = np.array([1.0, 1.0, 1.0, 1.0])
    spiky = np.array([0.0, 0.0, 0.0, 4.0])
    assert mae(y_true, spread) == mae(y_true, spiky)  # same MAE by construction
    assert rmse(y_true, spiky) > rmse(y_true, spread)


def test_mape_hand_computed():
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([110.0, 180.0])
    # |10/100| = 10%, |20/200| = 10% -> 10%
    assert mape(y_true, y_pred) == pytest.approx(10.0)


def test_metrics_drop_nan_pairs_rather_than_returning_nan():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, np.nan, 3.0])
    assert mae(y_true, y_pred) == 0.0
    assert all_metrics(y_true, y_pred)["n"] == 2


def test_metrics_reject_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        mae([1.0, 2.0], [1.0])


def test_r2_undefined_on_constant_truth():
    with pytest.raises(ValueError, match="constant"):
        r2([5.0, 5.0, 5.0], [1.0, 2.0, 3.0])


def test_comparison_table_shape_and_sorting():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    results = {
        "good": (y, y + 0.1),
        "bad": (y, y + 10.0),
    }
    table = comparison_table(results, {"good": "close", "bad": "far"})
    assert list(table.columns) == ["Model", "Description", "RMSE", "MAE", "MAPE", "R2", "n"]
    assert table.iloc[0]["Model"] == "good"  # sorted ascending by RMSE
    assert table.iloc[0]["Description"] == "close"


# --------------------------------------------------------------------------------------
# Persistence and moving average against hand-computed values
# --------------------------------------------------------------------------------------


def test_persistence_equals_previous_value():
    df = _fixture()
    pred = predict_persistence(df)
    assert pd.isna(pred.iloc[0])
    for t in range(1, len(df)):
        assert pred.iloc[t] == df["total"].iloc[t - 1]
    # Hand-computed: series is 100, 200, 300, ... so yhat_3 = 300.
    assert pred.iloc[3] == 300.0


def test_seasonal_naive_equals_value_seven_days_earlier():
    df = _fixture()
    pred = predict_seasonal_naive(df)
    assert pred.iloc[10] == df["total"].iloc[3]
    assert pred.iloc[:7].isna().all()


def test_moving_average_equals_trailing_seven_day_mean():
    df = _fixture()
    pred = predict_moving_average(df)
    # At t=7 the window is y_0..y_6 = 100..700 -> mean 400.
    assert pred.iloc[7] == pytest.approx(400.0)
    assert pred.iloc[:7].isna().all()
    for t in range(7, len(df)):
        assert pred.iloc[t] == pytest.approx(df["total"].iloc[t - 7 : t].mean())


def test_moving_average_window_excludes_the_current_day():
    """Direct leakage check on the baseline itself."""
    df = _fixture()
    pred = predict_moving_average(df)
    t = 10
    assert pred.iloc[t] != pytest.approx(df["total"].iloc[t - 6 : t + 1].mean())
    assert pred.iloc[t] == pytest.approx(df["total"].iloc[t - 7 : t].mean())


def test_baselines_raise_when_feature_table_missing_columns():
    with pytest.raises(ValueError, match="missing"):
        predict_persistence(pd.DataFrame({"total": [1.0, 2.0]}))
    with pytest.raises(ValueError, match="missing"):
        predict_moving_average(pd.DataFrame({"total": [1.0, 2.0]}))


# --------------------------------------------------------------------------------------
# SARIMAX exog contract
# --------------------------------------------------------------------------------------


def test_sarimax_exog_contains_no_lag_features():
    """SARIMAX models its own autoregression; exog must be calendar-only."""
    for col in EXOG_COLS:
        assert "_lag_" not in col
        assert "_roll_" not in col


def test_sarimax_exog_guard_raises_on_lag_feature():
    exog = pd.DataFrame({"feat_is_weekend": [0, 1], "feat_total_lag_1": [1.0, 2.0]})
    with pytest.raises(ValueError, match="autoregressive feature"):
        assert_no_lag_features_in_exog(exog)


def test_sarimax_exog_guard_raises_on_rolling_feature():
    exog = pd.DataFrame({"feat_total_roll_mean_7": [1.0, 2.0]})
    with pytest.raises(ValueError, match="autoregressive feature"):
        assert_no_lag_features_in_exog(exog)


def test_sarimax_exog_excludes_the_holiday_reference_level():
    """'none' is dropped to avoid the dummy trap against SARIMAX's intercept."""
    assert f"{FEATURE_PREFIX}holiday_type_none" not in EXOG_COLS
    assert f"{FEATURE_PREFIX}holiday_type_festivo_nacional" in EXOG_COLS


def test_build_exog_raises_on_missing_column():
    with pytest.raises(ValueError, match="missing exog"):
        build_exog(pd.DataFrame({"feat_is_weekend": [0, 1]}))


# --------------------------------------------------------------------------------------
# Weather promotion: lagged only, never same-day
# --------------------------------------------------------------------------------------


def test_weather_features_are_lagged_only():
    out = add_weather_features(_weather_fixture())
    for col in WEATHER_COLS:
        assert col not in out.columns
        assert f"{FEATURE_PREFIX}{col}" not in out.columns


def test_weather_feature_count_and_names():
    out = add_weather_features(_weather_fixture())
    assert list(out.columns) == weather_feature_names()
    # 21 variables x (4 lags + 1 rolling mean)
    assert len(out.columns) == len(WEATHER_COLS) * (len(WEATHER_LAGS) + 1) == 105


def test_weather_lag_values_come_from_the_past():
    df = _weather_fixture()
    out = add_weather_features(df)
    col = WEATHER_COLS[0]
    for lag in WEATHER_LAGS:
        assert out[f"{FEATURE_PREFIX}{col}_lag_{lag}"].iloc[15] == df[col].iloc[15 - lag]


def test_weather_rolling_is_shift_first():
    df = _weather_fixture()
    out = add_weather_features(df)
    col = WEATHER_COLS[0]
    t = 15
    assert out[f"{FEATURE_PREFIX}{col}_roll_mean_7"].iloc[t] == pytest.approx(
        df[col].iloc[t - 7 : t].mean()
    )


def test_weather_omits_long_lags():
    """Lags 14/21/28 are deliberately excluded for weather."""
    names = weather_feature_names()
    for lag in (14, 21, 28):
        assert not any(f"_lag_{lag}" in n for n in names)


def test_weather_same_day_guard_fires():
    from src.features.weather_features import _assert_no_same_day_weather

    leaky = pd.DataFrame({WEATHER_COLS[0]: [1.0, 2.0]})
    with pytest.raises(ValueError, match="SAME-DAY WEATHER GUARD"):
        _assert_no_same_day_weather(leaky)


def test_built_feature_table_has_no_same_day_weather_in_matrix():
    from src.features.build_features import FEATURES_DAILY_FILE, feature_columns

    df = pd.read_parquet(FEATURES_DAILY_FILE)
    feats = feature_columns(df)
    for col in WEATHER_COLS:
        assert col not in feats
        assert f"{FEATURE_PREFIX}{col}" not in feats
    # ...but the raw columns survive as reference data.
    assert all(col in df.columns for col in WEATHER_COLS)
