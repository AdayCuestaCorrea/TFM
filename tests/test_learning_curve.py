"""Phase 7 tests — the LSTM learning curve is a val-only diagnostic that persists no model.

The fits here use a 1-unit / 1-epoch config: these tests assert wiring (trailing window
anchored at train_end, the fixed eval set, no test-split leakage, no model artifact), not
accuracy. `@pytest.mark.slow` marks the ones that build a TensorFlow model.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.evaluation import lstm_learning_curve as lc
from src.models.lstm.model import LSTMConfig
from src.utils.paths import MODELS_DIR


TINY = LSTMConfig(window=28, units=1, epochs=1, patience=1, batch_size=32)


# --------------------------------------------------------------------------------------
# Constants and source-level guarantees (no fit)
# --------------------------------------------------------------------------------------


def test_fractions_and_seeds_are_the_pre_registered_design():
    assert lc.FRACTIONS == [0.25, 0.40, 0.55, 0.70, 0.85, 1.00]
    assert lc.SEEDS[0] == 42  # GLOBAL_SEED stays primary, not redefined
    assert len(lc.SEEDS) == 3
    # Fraction 1.00 alone carries two extra seeds -- the Phase 8 parity-control baseline.
    assert lc.EXTRA_FRACTION1_SEEDS == [3, 4]


def test_degeneracy_thresholds_sit_between_phase4_regimes():
    # Phase 4: fold 1 degenerate at std ratio 0.199 / corr 0.300; healthy folds >= 0.75 / 0.79.
    assert 0.30 < lc.DEGENERATE_STD_RATIO_MAX < 0.75
    assert 0.30 < lc.DEGENERATE_CORR_MAX < 0.79


def test_learning_curve_never_reads_the_test_split():
    source = Path(lc.__file__).read_text(encoding="utf-8")
    # Degeneracy and metrics are read from the val split only.
    assert '"val"' in source
    assert '["test"]' not in source and 'metrics["test"]' not in source
    assert 'split"] == "test"' not in source


def test_learning_curve_module_persists_no_model_artifact():
    source = Path(lc.__file__).read_text(encoding="utf-8")
    assert "MODELS_DIR" not in source
    assert ".save(" not in source
    assert ".keras" not in source


def test_summarise_by_fraction_reports_median_and_band():
    synthetic = pd.DataFrame(
        {
            "fraction": [0.5, 0.5, 0.5, 1.0, 1.0, 1.0],
            "train_rows": [400, 400, 400, 889, 889, 889],
            "seed": [42, 1, 2, 42, 1, 2],
            "val_MAE": [500.0, 400.0, 600.0, 300.0, 350.0, 250.0],
            "degenerate": [True, False, False, False, False, False],
        }
    )
    out = lc.summarise_by_fraction(synthetic).set_index("fraction")
    assert out.loc[0.5, "val_MAE_median"] == 500.0
    assert out.loc[0.5, "val_MAE_min"] == 400.0
    assert out.loc[0.5, "val_MAE_max"] == 600.0
    assert out.loc[0.5, "n_degenerate"] == 1
    assert out.loc[1.0, "n_seeds"] == 3


# --------------------------------------------------------------------------------------
# Wiring (tiny fits)
# --------------------------------------------------------------------------------------


@pytest.mark.slow
def test_max_train_rows_none_reproduces_the_phase4_training_set():
    from src.models.lstm.final_model import train_final_model

    _, result, _, _ = train_final_model(config=TINY, verbose=False, max_train_rows=None)
    assert result.n_train_rows == 889
    assert result.n_train_sequences == 861  # 889 - window(28)


@pytest.mark.slow
def test_train_rows_are_a_trailing_window_anchored_at_train_end():
    from src.models.lstm.final_model import train_final_model

    predictions, result, _, _ = train_final_model(
        config=TINY, verbose=False, max_train_rows=222
    )
    assert result.n_train_rows == 222
    assert result.n_train_sequences == 222 - 28
    # The evaluation set is held fixed: val + test coverage is unchanged by training size.
    eval_rows = predictions[predictions["split"].isin(["val", "test"])]
    assert len(eval_rows) == 393


@pytest.mark.slow
def test_assemble_full_curve_has_five_seeds_at_fraction_one():
    # One deterministic pass reproduces the whole artifact, incl. the 5-seed fraction-1.00
    # point that lstm_parity_control.UNIVARIATE_VAL_MAE depends on.
    curve = lc.assemble_full_curve(fractions=[1.0], verbose=False, config=TINY)
    frac1 = curve[curve["fraction"] == 1.0]
    assert sorted(frac1["seed"].tolist()) == [1, 2, 3, 4, 42]
    assert len(frac1) == 5
    # Idempotent: de-duplicated on (fraction, seed), so no seed is double-counted.
    assert frac1["seed"].is_unique


@pytest.mark.slow
def test_curve_reports_every_fraction_and_seed_combination(tmp_path):
    models_before = set(p.name for p in MODELS_DIR.glob("*")) if MODELS_DIR.exists() else set()

    curve = lc.run_learning_curve(
        fractions=[0.5, 1.0], seeds=[42, 1], verbose=False, config=TINY
    )
    assert len(curve) == 4
    assert set(zip(curve["fraction"], curve["seed"])) == {
        (0.5, 42), (0.5, 1), (1.0, 42), (1.0, 1)
    }
    for col in ("pred_std_ratio", "pred_actual_corr", "degenerate", "val_MAE", "train_rows"):
        assert col in curve.columns
    assert curve["degenerate"].dtype == bool
    # No test-split column leaked into the persisted schema.
    assert not any("test" in c.lower() for c in curve.columns)
    # Fraction 1.0 trains on the full block; 0.5 on roughly half.
    assert curve[curve["fraction"] == 1.0]["train_rows"].iloc[0] == 889
    assert curve[curve["fraction"] == 0.5]["train_rows"].iloc[0] == round(0.5 * 889)

    models_after = set(p.name for p in MODELS_DIR.glob("*")) if MODELS_DIR.exists() else set()
    assert models_before == models_after, "learning curve must not write under models/"
