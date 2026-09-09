"""Phase 8 tests -- the feature-block ablation is a diagnostic control, not a model.

The load-bearing test is `test_phase8_adds_no_model_to_the_master_comparison`: this whole
phase is explanatory, and nothing it produces may enter chapter 5's master comparison.
The rest pin the pre-registered subset design, the same-day leakage guards at the subset
boundary, and that the materiality threshold is a module constant.
"""

import inspect
from pathlib import Path

import pandas as pd
import pytest

from src.evaluation import feature_block_ablation as fba
from src.evaluation.dashboard_data import MODEL_NAMES
from src.features.build_features import feature_columns
from src.models.hybrid_residual.xgboost_alone import load_training_split
from src.utils.paths import MODELS_DIR, PROCESSED_DIR

EXPECTED_MASTER_MODELS = {
    "persistence",
    "seasonal_naive",
    "moving_average_7",
    "moving_average_28",
    "sarimax",
    "lstm_alone",
    "xgboost_alone",
    "hybrid",
    "hybrid_weighted",
    "ensemble_equal",
    "ensemble_inverse_mae",
}

PHASE8_NAMES = {
    "solo_temporal",
    "solo_exogeno",
    "sin_meteo",
    "solo_calendario",
    "lstm_parity_calendario",
    "lstm_parity_completo",
    "ablation",
}


@pytest.fixture(scope="module")
def train_df() -> pd.DataFrame:
    return load_training_split()


# --------------------------------------------------------------------------------------
# The mechanical no-candidate guard
# --------------------------------------------------------------------------------------


def test_phase8_adds_no_model_to_the_master_comparison():
    assert set(MODEL_NAMES) == EXPECTED_MASTER_MODELS

    full = pd.read_parquet(PROCESSED_DIR / "full_comparison.parquet")
    sensitivity = pd.read_parquet(PROCESSED_DIR / "sensitivity_comparison.parquet")

    assert set(full["Model"]) <= EXPECTED_MASTER_MODELS
    assert set(sensitivity["Model"]) == EXPECTED_MASTER_MODELS

    assert PHASE8_NAMES.isdisjoint(set(full["Model"]))
    assert PHASE8_NAMES.isdisjoint(set(sensitivity["Model"]))


def test_phase8_does_not_extend_the_ordering_constants():
    from src.evaluation.sensitivity_report import ORDER as SENSITIVITY_ORDER

    assert len(MODEL_NAMES) == 11
    assert set(SENSITIVITY_ORDER) == EXPECTED_MASTER_MODELS


def test_ablation_module_persists_no_model_artifact():
    source = Path(fba.__file__).read_text(encoding="utf-8")
    assert "MODELS_DIR" not in source
    assert ".save(" not in source
    assert "joblib.dump" not in source
    assert ".keras" not in source


# --------------------------------------------------------------------------------------
# Pre-registered subset design
# --------------------------------------------------------------------------------------


def test_feature_subset_none_reproduces_the_phase5_feature_set(train_df):
    from src.models.hybrid_residual import xgboost_alone

    sig = inspect.signature(xgboost_alone.train)
    assert sig.parameters["feature_subset"].default is None
    # None -> the full feature_columns(df) set, 161 columns.
    assert fba.subset_columns(train_df, "completo") == feature_columns(train_df)
    assert len(feature_columns(train_df)) == 161


def test_ablation_subsets_are_the_pre_registered_design(train_df):
    sizes = {k: len(fba.subset_columns(train_df, k)) for k in fba.SUBSETS}
    assert sizes["solo_temporal"] == 39
    assert sizes["solo_exogeno"] == 122
    assert sizes["sin_meteo"] == 56
    assert sizes["solo_calendario"] == 17

    temporal = set(fba.subset_columns(train_df, "solo_temporal"))
    exogeno = set(fba.subset_columns(train_df, "solo_exogeno"))
    completo = set(fba.subset_columns(train_df, "completo"))
    assert temporal.isdisjoint(exogeno)
    assert temporal | exogeno == completo

    weather = {c for c in completo if c not in fba.subset_columns(train_df, "sin_meteo")}
    assert set(fba.subset_columns(train_df, "sin_meteo")) == completo - weather


def test_ablation_subsets_contain_only_feat_prefixed_columns(train_df):
    operators = ("metro", "emt", "carretera", "cercanias", "total")
    for key in fba.SUBSETS:
        cols = fba.subset_columns(train_df, key)
        assert all(c.startswith("feat_") for c in cols)
        # No same-day operator column (only *_lag_* operator features are allowed).
        assert not any(c in operators for c in cols)
        # No same-day weather: every weather feature carries a lag/roll suffix.
        assert not any(c == f"feat_{op}" for op in operators for c in cols)


def test_block_materiality_threshold_is_a_module_constant():
    assert fba.BLOCK_MATERIAL_PCT == 5.0
    src = inspect.getsource(fba.evaluate_block_criteria)
    assert "BLOCK_MATERIAL_PCT" in src


def test_evaluate_block_criteria_uses_the_threshold_mechanically():
    synthetic = pd.DataFrame(
        [
            {"subset": "completo", "n_features": 161, "split": "val", "MAE": 100_000.0},
            {"subset": "completo", "n_features": 161, "split": "test", "MAE": 90_000.0},
            {"subset": "solo_temporal", "n_features": 39, "split": "val", "MAE": 120_000.0},
            {"subset": "solo_temporal", "n_features": 39, "split": "test", "MAE": 110_000.0},
            {"subset": "solo_exogeno", "n_features": 122, "split": "val", "MAE": 101_000.0},
            {"subset": "solo_exogeno", "n_features": 122, "split": "test", "MAE": 99_000.0},
        ]
    )
    out = fba.evaluate_block_criteria(synthetic).set_index("subset")
    assert out.loc["solo_temporal", "delta_val_MAE_pct"] == pytest.approx(20.0)
    assert out.loc["solo_temporal", "verdict"] == fba.VERDICT_MATERIAL
    assert out.loc["solo_exogeno", "verdict"] == fba.VERDICT_IMMATERIAL
    assert out.loc["completo", "verdict"] == "referencia"


def test_completo_reference_matches_the_published_phase5_metrics():
    ref = fba.completo_reference().set_index("split")
    assert ref.loc["val", "MAE"] == pytest.approx(244_843, abs=50)
    assert ref.loc["test", "MAE"] == pytest.approx(156_121, abs=50)
    assert ref.loc["val", "n_features"] == 161


# --------------------------------------------------------------------------------------
# End-to-end (slow: retrains XGBoost on one subset)
# --------------------------------------------------------------------------------------


@pytest.mark.slow
def test_run_ablation_writes_nothing_under_models():
    models_before = (
        {p.name for p in MODELS_DIR.glob("*")} if MODELS_DIR.exists() else set()
    )
    out = fba.run_ablation(subsets={"solo_calendario": fba.SUBSETS["solo_calendario"]})
    assert set(out["subset"]) == {"solo_calendario"}
    assert set(out["split"]) == {"val", "test"}
    for col in ("MAE", "RMSE", "MAPE", "R2", "max_depth", "holdout_mae"):
        assert col in out.columns
    models_after = (
        {p.name for p in MODELS_DIR.glob("*")} if MODELS_DIR.exists() else set()
    )
    assert models_before == models_after
