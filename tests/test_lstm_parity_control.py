"""Phase 8 tests -- the informational-parity LSTM is a val-only diagnostic control.

The fits here use a 1-unit / 1-epoch config: these tests assert wiring (two inputs, the
declared exog width, no test-split leakage, no model artifact) and the pre-registered
verdict constants, not accuracy.
"""

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation import lstm_parity_control as pc
from src.evaluation.dashboard_data import MODEL_NAMES
from src.models.lstm.model import LSTMConfig
from src.models.lstm.scaling import FeatureScaler
from src.models.lstm.sequence_builder import build_exog_matrix
from src.utils.paths import MODELS_DIR, PROCESSED_DIR

TINY = LSTMConfig(window=28, units=1, epochs=1, patience=1, batch_size=32)

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


# --------------------------------------------------------------------------------------
# No-candidate guard + source-text structural guards
# --------------------------------------------------------------------------------------


def test_phase8_parity_adds_no_model_to_the_master_comparison():
    assert set(MODEL_NAMES) == EXPECTED_MASTER_MODELS
    full = pd.read_parquet(PROCESSED_DIR / "full_comparison.parquet")
    sensitivity = pd.read_parquet(PROCESSED_DIR / "sensitivity_comparison.parquet")
    leaked = {"lstm_parity_calendario", "lstm_parity_completo"}
    assert leaked.isdisjoint(set(full["Model"]))
    assert leaked.isdisjoint(set(sensitivity["Model"]))


def test_parity_module_persists_no_model_artifact():
    source = Path(pc.__file__).read_text(encoding="utf-8")
    assert "MODELS_DIR" not in source
    assert ".save(" not in source
    assert "joblib.dump" not in source
    assert ".keras" not in source


def test_baseline_artifact_check_runs_at_entry_points_not_at_import():
    source = Path(pc.__file__).read_text(encoding="utf-8")
    # No bare module-scope call (CLAUDE.md forbids top-level I/O in src/).
    assert not any(
        line.strip() == "_assert_baseline_matches_artifact()" and line == line.lstrip()
        for line in source.splitlines()
    )
    # But it IS wired into the real entry points, so drift still fails loudly.
    assert "_assert_baseline_matches_artifact()" in inspect.getsource(pc.run_parity_control)
    assert "_assert_baseline_matches_artifact()" in inspect.getsource(pc.main)


def test_parity_control_never_reads_the_test_split():
    source = Path(pc.__file__).read_text(encoding="utf-8")
    assert '"val"' in source
    assert '["test"]' not in source
    assert 'metrics["test"]' not in source
    assert 'split"] == "test"' not in source


# --------------------------------------------------------------------------------------
# Pre-registered constants (the section 2.5 correction)
# --------------------------------------------------------------------------------------


def test_parity_thresholds_are_module_constants():
    assert pc.PARITY_GAP_CLOSED_HIGH == 60.0
    assert pc.PARITY_GAP_CLOSED_LOW == 25.0
    assert pc.UNDERTRAINED_EPOCH_MAX == 20
    src = inspect.getsource(pc.evaluate_parity_criteria)
    assert "PARITY_GAP_CLOSED_HIGH" in src
    assert "PARITY_GAP_CLOSED_LOW" in src


def test_parity_baseline_is_the_five_seed_median_not_the_best_seed():
    assert pc.SEEDS == [42, 1, 2, 3, 4]
    curve = pd.read_parquet(pc.LEARNING_CURVE_FILE)
    frac1 = curve[np.isclose(curve["fraction"], 1.0)]
    # The learning curve's fraction-1.00 point carries the same 5 seeds as the parity run.
    assert sorted(frac1["seed"].tolist()) == sorted(pc.SEEDS)
    assert round(float(frac1["val_MAE"].median())) == pc.UNIVARIATE_VAL_MAE
    assert pc.UNIVARIATE_VAL_MAE == 416_324
    assert pc.UNIVARIATE_VAL_MAE != 411_142
    assert pc.UNIVARIATE_VAL_MAE_BEST_SEED == 411_142


def test_gap_closure_uses_the_published_reference_maes():
    assert pc.UNIVARIATE_VAL_MAE == 416_324
    assert pc.XGBOOST_ALONE_VAL_MAE == 244_843
    assert pc._GAP == 171_481


def test_degeneracy_thresholds_are_imported_not_redeclared():
    source = Path(pc.__file__).read_text(encoding="utf-8")
    assert "from src.evaluation.lstm_learning_curve import" in source
    assert "0.50" not in source
    assert pc.DEGENERATE_STD_RATIO_MAX == 0.50
    assert pc.DEGENERATE_CORR_MAX == 0.50


def test_verdict_ordering_degeneracy_first_then_band_then_median():
    # Majority degenerate -> capacity verdict regardless of MAE.
    degen = pd.DataFrame(
        {
            "scope": ["calendario"] * 3,
            "seed": [42, 1, 2],
            "n_exog": [17] * 3,
            "val_MAE": [300_000.0, 305_000.0, 310_000.0],
            "degenerate": [True, True, False],
            "undertrained": [False] * 3,
            "gap_closed_pct": [pc._gap_closed_pct(v) for v in (300_000.0, 305_000.0, 310_000.0)],
            "best_epoch": [5, 5, 30],
        }
    )
    out = pc.evaluate_parity_criteria(degen).set_index("scope")
    assert out.loc["calendario", "verdict"] == pc.VERDICT_CAPACITY

    # Healthy, tight band well below the 60% MAE -> asimetria informativa.
    good = degen.copy()
    good["degenerate"] = False
    good["val_MAE"] = [300_000.0, 302_000.0, 304_000.0]
    out = pc.evaluate_parity_criteria(good).set_index("scope")
    assert out.loc["calendario", "verdict"] == pc.VERDICT_ASYMMETRY

    # Healthy, tight band well above the 25% MAE -> limitacion arquitectonica.
    bad = degen.copy()
    bad["degenerate"] = False
    bad["val_MAE"] = [400_000.0, 402_000.0, 404_000.0]
    out = pc.evaluate_parity_criteria(bad).set_index("scope")
    assert out.loc["calendario", "verdict"] == pc.VERDICT_ARCHITECTURE

    # Band straddling the 60% threshold -> downgrade to ambiguous (O6).
    straddle = degen.copy()
    straddle["degenerate"] = False
    straddle["val_MAE"] = [300_000.0, 313_435.0, 340_000.0]
    out = pc.evaluate_parity_criteria(straddle).set_index("scope")
    assert out.loc["calendario", "verdict"] == pc.VERDICT_AMBIGUOUS


# --------------------------------------------------------------------------------------
# Scope columns
# --------------------------------------------------------------------------------------


def test_scope_columns_are_the_pre_registered_widths():
    df = pd.read_parquet(pc.FEATURES_DAILY_FILE)
    assert len(pc.scope_columns(df, "calendario")) == 17
    assert len(pc.scope_columns(df, "completo")) == 161
    assert all(c.startswith("feat_") for c in pc.scope_columns(df, "completo"))


# --------------------------------------------------------------------------------------
# Sequence builder + scaler maths (no fit)
# --------------------------------------------------------------------------------------


def test_exog_matrix_gathers_row_t_not_the_window():
    features = np.arange(20).reshape(10, 2).astype("float64")
    positions = np.array([3, 5, 9])
    out = build_exog_matrix(features, positions)
    assert out.shape == (3, 2)
    np.testing.assert_array_equal(out[0], features[3])   # row t, never t-1
    np.testing.assert_array_equal(out[2], features[9])


def test_feature_scaler_handles_a_constant_column():
    rows = np.column_stack([np.linspace(0, 10, 50), np.zeros(50)])
    scaler = FeatureScaler().fit(rows)
    scaled = scaler.transform(rows)
    assert np.isfinite(scaled).all()
    assert not np.isnan(scaled).any()
    assert np.allclose(scaled[:, 1], 0.0)


# --------------------------------------------------------------------------------------
# Wiring (slow, tiny fits)
# --------------------------------------------------------------------------------------


@pytest.mark.slow
def test_parity_model_has_two_inputs_and_the_declared_exog_width():
    df = pd.read_parquet(pc.FEATURES_DAILY_FILE)
    cols = pc.scope_columns(df, "calendario")
    from src.models.lstm.final_model import train_final_model

    _, result, _, model = train_final_model(config=TINY, verbose=False, exog_columns=cols)
    assert len(model.inputs) == 2
    assert tuple(model.inputs[1].shape) == (None, 17)
    assert result.n_train_rows == 889
    assert result.n_train_sequences == 861


@pytest.mark.slow
def test_exog_columns_none_reproduces_the_phase4_training_set():
    from src.models.lstm.final_model import train_final_model

    _, result, _, model = train_final_model(config=TINY, verbose=False, exog_columns=None)
    assert len(model.inputs) == 1
    assert tuple(model.inputs[0].shape) == (None, 28, 1)
    assert result.n_train_rows == 889
    assert result.n_train_sequences == 861


@pytest.mark.slow
def test_parity_control_reports_every_scope_and_seed():
    models_before = (
        {p.name for p in MODELS_DIR.glob("*")} if MODELS_DIR.exists() else set()
    )
    df = pd.read_parquet(pc.FEATURES_DAILY_FILE)
    small = {"calendario": pc.SCOPES["calendario"]}
    out = pc.run_parity_control(
        scopes=small, seeds=[42, 1], verbose=False, config=TINY
    )
    assert len(out) == 2
    assert set(out["seed"]) == {42, 1}
    for col in ("val_MAE", "degenerate", "undertrained", "gap_closed_pct", "best_epoch"):
        assert col in out.columns
    assert out["degenerate"].dtype == bool
    assert out["undertrained"].dtype == bool
    assert not any("test" in c.lower() for c in out.columns)

    models_after = (
        {p.name for p in MODELS_DIR.glob("*")} if MODELS_DIR.exists() else set()
    )
    assert models_before == models_after
