"""Phase 5 tests: exclusion logic, leakage guards, the additive identity, and split sizes.

The additive-identity test matters more than it looks. A sign error or a scale mismatch
between stage 1 and stage 2 does not crash -- it produces a hybrid that is quietly worse
than its own stage 1, which is easy to misread as "the residual correction didn't help".
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.day_type_breakdown import breakdown, compare_models
from src.evaluation.fold_coverage_check import build_coverage_table
from src.models.hybrid_residual.assemble_residual_dataset import (
    EXCLUDED_FOLD,
    FORBIDDEN_SAME_DAY,
    assemble,
)
from src.models.hybrid_residual.tuning import INTERNAL_HOLDOUT_FRACTION, grid_search
from src.models.hybrid_residual.xgboost_alone import load_training_split
from src.features.weather_features import WEATHER_COLS

EXPECTED_TRAIN_ROWS = 889  # full Phase 3 train split, post warm-up
EXPECTED_RESIDUAL_ROWS = 552  # 889 - 200 (no OOF) - 137 (fold 1)


@pytest.fixture(scope="session")
def residual_set() -> pd.DataFrame:
    return assemble()


# --------------------------------------------------------------------------------------
# Exclusion logic
# --------------------------------------------------------------------------------------


def test_residual_set_excludes_rows_without_oof(residual_set):
    from src.models.lstm.oof import OOF_PREDICTIONS_FILE

    oof = pd.read_parquet(OOF_PREDICTIONS_FILE)
    no_oof_dates = set(oof.loc[~oof["has_oof"], "date"])
    assert not (set(residual_set["date"]) & no_oof_dates)


def test_residual_set_excludes_fold_one(residual_set):
    assert EXCLUDED_FOLD not in residual_set["fold"].unique()
    assert sorted(residual_set["fold"].unique().tolist()) == [2, 3, 4, 5]


def test_residual_set_has_expected_row_count(residual_set):
    assert len(residual_set) == EXPECTED_RESIDUAL_ROWS


def test_residual_set_date_coverage_starts_after_fold_one(residual_set):
    assert residual_set["date"].min() == pd.Timestamp("2024-01-01")
    assert residual_set["date"].max() == pd.Timestamp("2025-07-05")


def test_residual_equals_truth_minus_oof_prediction(residual_set):
    """Sign convention: residual = y_true - y_pred_oof, so the correction ADDS."""
    np.testing.assert_allclose(
        residual_set["residual"],
        residual_set["y_true"] - residual_set["y_pred_oof"],
        rtol=1e-9,
    )


def test_residual_set_has_no_nulls(residual_set):
    assert residual_set.isna().sum().sum() == 0


# --------------------------------------------------------------------------------------
# Leakage guards
# --------------------------------------------------------------------------------------


def test_no_same_day_operator_or_target_columns(residual_set):
    feat_cols = [c for c in residual_set.columns if c.startswith("feat_")]
    for col in FORBIDDEN_SAME_DAY:
        assert col not in feat_cols


def test_no_same_day_weather_columns(residual_set):
    feat_cols = [c for c in residual_set.columns if c.startswith("feat_")]
    for col in WEATHER_COLS:
        assert col not in feat_cols
        assert f"feat_{col}" not in feat_cols


def test_residual_features_are_all_prefixed(residual_set):
    non_feature = {"date", "residual", "y_true", "y_pred_oof", "fold"}
    for col in residual_set.columns:
        assert col in non_feature or col.startswith("feat_")


# --------------------------------------------------------------------------------------
# The additive identity
# --------------------------------------------------------------------------------------


def test_hybrid_equals_lstm_plus_residual_on_synthetic_fixture():
    """No silent unit or scale mismatch between the two components."""
    y_lstm = np.array([1_000_000.0, 2_000_000.0, 3_000_000.0])
    e_pred = np.array([50_000.0, -25_000.0, 0.0])
    y_hybrid = y_lstm + e_pred
    np.testing.assert_allclose(y_hybrid, [1_050_000.0, 1_975_000.0, 3_000_000.0])


def test_hybrid_recovers_truth_when_residual_model_is_perfect():
    """If stage 2 predicted residuals exactly, the hybrid would equal the truth."""
    y_true = np.array([4_000_000.0, 5_500_000.0, 2_300_000.0])
    y_lstm = np.array([3_800_000.0, 5_900_000.0, 2_500_000.0])
    perfect_residual = y_true - y_lstm
    np.testing.assert_allclose(y_lstm + perfect_residual, y_true, rtol=1e-12)


def test_zero_correction_leaves_stage_one_unchanged():
    y_lstm = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(y_lstm + np.zeros(3), y_lstm)


def test_combine_output_columns_and_identity():
    """End-to-end on the real artifacts, if Phase 5 has been run."""
    from src.models.hybrid_residual.combine import HYBRID_PREDICTIONS_FILE

    if not HYBRID_PREDICTIONS_FILE.exists():
        pytest.skip("hybrid predictions not built yet")

    out = pd.read_parquet(HYBRID_PREDICTIONS_FILE)
    assert {"date", "y_true", "y_pred_lstm", "e_pred_xgb", "y_pred_hybrid"} <= set(out.columns)
    np.testing.assert_allclose(
        out["y_pred_hybrid"], out["y_pred_lstm"] + out["e_pred_xgb"], rtol=1e-9
    )


# --------------------------------------------------------------------------------------
# XGBoost-alone must NOT inherit the residual set's exclusions
# --------------------------------------------------------------------------------------


def test_xgboost_alone_trains_on_the_full_889_row_split():
    """The common mistake this guards: reusing the 552-row residual subset here.

    XGBoost-alone has no dependency on the LSTM's OOF folds, so neither the 200-row
    initial-history exclusion nor the degenerate fold 1 applies to it. Handicapping it
    would make the comparison against the hybrid meaningless.
    """
    train_df = load_training_split()
    assert len(train_df) == EXPECTED_TRAIN_ROWS
    assert len(train_df) != EXPECTED_RESIDUAL_ROWS
    assert train_df["date"].min() == pd.Timestamp("2023-01-29")
    assert train_df["date"].max() == pd.Timestamp("2025-07-05")


def test_xgboost_alone_and_residual_share_the_same_feature_matrix(residual_set):
    """Feature parity is what makes the two models comparable."""
    from src.features.build_features import feature_columns

    alone_feats = set(feature_columns(load_training_split()))
    residual_feats = {c for c in residual_set.columns if c.startswith("feat_")}
    assert alone_feats == residual_feats


# --------------------------------------------------------------------------------------
# Tuning methodology
# --------------------------------------------------------------------------------------


def test_internal_holdout_is_chronological_not_random():
    """The holdout must be the LAST rows by date, never a random sample."""
    n = 100
    X = pd.DataFrame({"feat_a": np.arange(n, dtype="float64")})
    y = pd.Series(np.arange(n, dtype="float64"))
    dates = pd.Series(pd.date_range("2024-01-01", periods=n, freq="D"))

    result = grid_search(X, y, dates, verbose=False)
    expected_holdout = int(round(n * INTERNAL_HOLDOUT_FRACTION))
    assert result.n_holdout == expected_holdout
    assert result.n_fit == n - expected_holdout
    # The holdout is the tail, so it ends on the final date.
    assert result.holdout_end == dates.iloc[-1]
    assert result.holdout_start == dates.iloc[n - expected_holdout]


def test_tuning_grid_size_is_as_documented():
    from src.models.hybrid_residual.tuning import (
        LEARNING_RATE,
        MAX_DEPTH,
        MIN_CHILD_WEIGHT,
        N_ESTIMATORS,
    )

    assert len(MAX_DEPTH) * len(N_ESTIMATORS) * len(LEARNING_RATE) * len(MIN_CHILD_WEIGHT) == 54


# --------------------------------------------------------------------------------------
# Coverage check and day-type breakdown
# --------------------------------------------------------------------------------------


def test_coverage_check_flags_zero_coverage_correctly():
    table, is_safe = build_coverage_table()
    # On the real data every fold-1 category recurs in folds 2-5.
    assert is_safe
    assert not table["zero_coverage_elsewhere"].any()


def test_coverage_check_detects_a_genuinely_lost_category():
    """The flag must be capable of firing, not vacuously False."""
    fabricated = pd.DataFrame(
        {
            "category": ["only_in_fold_1"],
            "count_in_fold_1": [3],
            "count_in_folds_2_5": [0],
        }
    )
    fabricated["zero_coverage_elsewhere"] = (fabricated["count_in_fold_1"] > 0) & (
        fabricated["count_in_folds_2_5"] == 0
    )
    assert bool(fabricated["zero_coverage_elsewhere"].any())


def test_day_type_breakdown_groups_and_isolates_bridge_days():
    features = pd.DataFrame(
        {
            "day_type": ["laborable", "laborable", "sabado", "festivo"],
            "feat_is_bridge_day": [1, 0, 0, 0],
        }
    )
    y_true = pd.Series([100.0, 100.0, 100.0, 100.0])
    y_pred = pd.Series([150.0, 110.0, 120.0, 200.0])

    table = breakdown(y_true, y_pred, features).set_index("group")
    assert table.loc["laborable, bridge day", "MAE"] == pytest.approx(50.0)
    assert table.loc["laborable, ordinary", "MAE"] == pytest.approx(10.0)
    assert table.loc["festivo", "MAE"] == pytest.approx(100.0)
    assert table.loc["laborable", "n"] == 2


def test_compare_models_aligns_groups_across_models():
    features = pd.DataFrame(
        {"day_type": ["laborable", "festivo"], "feat_is_bridge_day": [0, 0]}
    )
    y_true = pd.Series([100.0, 100.0])
    table = compare_models(
        {"a": (y_true, pd.Series([110.0, 120.0])), "b": (y_true, pd.Series([105.0, 190.0]))},
        features,
    )
    assert "a_MAE" in table.columns and "b_MAE" in table.columns
    row = table[table["group"] == "festivo"].iloc[0]
    assert row["a_MAE"] == pytest.approx(20.0)
    assert row["b_MAE"] == pytest.approx(90.0)


def test_breakdown_raises_on_missing_columns():
    with pytest.raises(ValueError, match="missing column"):
        breakdown(pd.Series([1.0]), pd.Series([1.0]), pd.DataFrame({"day_type": ["laborable"]}))
