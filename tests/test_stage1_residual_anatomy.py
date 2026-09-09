"""Phase 7 tests — Stage 1 residual anatomy is a diagnostic control, not a model.

The load-bearing test here is `test_phase7_adds_no_model_to_the_master_comparison`: this
whole phase is explanatory, and nothing it produces may enter chapter 5's master
comparison. The rest pin the statistics (ACF on a known AR(1), a synthetic period-7 sine,
BH monotonicity) and the reuse contracts (calendar breakdown, same-day-weather guard).
"""

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation import stage1_residual_anatomy as sra
from src.evaluation.dashboard_data import MODEL_NAMES
from src.evaluation.metrics import mae
from src.utils.paths import PROCESSED_DIR


@pytest.fixture(scope="session")
def primary_residuals() -> pd.DataFrame:
    return sra.load_stage1_residuals(sra.PRIMARY_SERIES)


@pytest.fixture(scope="session")
def weather_corr() -> pd.DataFrame:
    return sra.residual_weather_correlation(sra.PRIMARY_SERIES)


# --------------------------------------------------------------------------------------
# The mechanical guard: this phase adds no model to the master comparison
# --------------------------------------------------------------------------------------

# Frozen expectation, independent of dashboard_data, so an accidental edit there is caught
# here too. Matches CLAUDE.md's 11-model master comparison as of Phase 5b.
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


def test_phase7_adds_no_model_to_the_master_comparison():
    assert set(MODEL_NAMES) == EXPECTED_MASTER_MODELS

    full = pd.read_parquet(PROCESSED_DIR / "full_comparison.parquet")
    sensitivity = pd.read_parquet(PROCESSED_DIR / "sensitivity_comparison.parquet")

    # full_comparison covers the Phase 5 subset; sensitivity_comparison covers all 11.
    assert set(full["Model"]) <= EXPECTED_MASTER_MODELS
    assert set(sensitivity["Model"]) == EXPECTED_MASTER_MODELS

    # No Phase 7 name leaked into either table.
    phase7_names = {"stage1_residual", "error_chain", "learning_curve", "fold_size"}
    assert phase7_names.isdisjoint(set(full["Model"]))
    assert phase7_names.isdisjoint(set(sensitivity["Model"]))


def test_phase7_does_not_extend_the_ordering_constants():
    """dashboard_data.MODEL_NAMES and sensitivity_report.ORDER stay at 11 entries."""
    from src.evaluation.sensitivity_report import ORDER as SENSITIVITY_ORDER

    assert len(MODEL_NAMES) == 11
    assert set(SENSITIVITY_ORDER) == EXPECTED_MASTER_MODELS


# --------------------------------------------------------------------------------------
# ACF / PACF maths
# --------------------------------------------------------------------------------------


def test_acf_of_a_known_ar1_series_recovers_its_coefficient():
    rng = np.random.default_rng(42)
    phi = 0.7
    n = 4000
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.standard_normal()

    table = sra.autocorrelation_frame(x, max_lag=10).set_index("lag")
    # AR(1) theoretical ACF is phi**k; lag 1 ~ phi, lag 2 ~ phi**2.
    assert table.loc[1, "acf"] == pytest.approx(phi, abs=0.05)
    assert table.loc[2, "acf"] == pytest.approx(phi**2, abs=0.05)
    # PACF of an AR(1) cuts off after lag 1.
    assert abs(table.loc[2, "pacf"]) < 0.1
    assert table.loc[1, "outside_band"]


def test_acf_of_white_noise_stays_inside_the_bartlett_band():
    rng = np.random.default_rng(7)
    x = rng.standard_normal(2000)
    table = sra.autocorrelation_frame(x, max_lag=sra.MAX_LAG)
    # At alpha=0.05 a handful of exceedances by chance is expected; demand well under half.
    assert table["outside_band"].sum() <= 4
    assert table["acf"].abs().max() < 0.15


def test_periodogram_of_a_synthetic_period_7_sine_peaks_at_period_7():
    t = np.arange(700)
    x = np.sin(2 * np.pi * t / 7.0) + 0.01 * np.random.default_rng(0).standard_normal(700)
    table = sra.periodogram_frame(x)
    top = table.iloc[0]
    assert top["period_days"] == pytest.approx(7.0, abs=0.2)
    # The peak bin should carry the overwhelming majority of the power.
    assert top["power_share"] > 0.8


# --------------------------------------------------------------------------------------
# Series construction (objection O2: the primary series is ONE model, contiguous)
# --------------------------------------------------------------------------------------


def test_primary_residual_series_is_contiguous_and_single_model(primary_residuals):
    assert len(primary_residuals) == 393
    gaps = primary_residuals["date"].diff().dropna().dt.days
    assert (gaps == 1).all(), "val_test residual series must have no date gaps"
    assert primary_residuals["fold"].isna().all()
    assert set(primary_residuals["split"]) == {"val", "test"}
    assert (primary_residuals["residual"] == primary_residuals["y_true"] - primary_residuals["y_pred"]).all()


def test_oof_residual_series_is_labelled_with_its_fold_boundaries():
    oof = sra.load_stage1_residuals(sra.SECONDARY_SERIES)
    # Fold 1 is degenerate and excluded (matches Phase 5's residual training set).
    assert set(oof["fold"]) == {2, 3, 4, 5}
    assert len(oof) == 552
    assert sra.DEGENERATE_OOF_FOLD == 1


def test_load_stage1_residuals_rejects_unknown_series():
    with pytest.raises(ValueError):
        sra.load_stage1_residuals("in_sample")


# --------------------------------------------------------------------------------------
# Reuse contracts
# --------------------------------------------------------------------------------------


def test_calendar_breakdown_reuses_day_type_breakdown():
    source = Path(sra.__file__).read_text(encoding="utf-8")
    assert "from src.evaluation.day_type_breakdown import" in source
    assert "breakdown(" in source
    # It must not reimplement a day-type groupby of its own.
    assert 'groupby("day_type")' not in source and "groupby('day_type')" not in source


def test_calendar_group_has_festivo_concentration_over_laborable(primary_residuals):
    table = sra.residual_by_calendar_group().set_index("group")
    ratio = table.loc["festivo", "mean_abs_residual"] / table.loc["laborable, ordinary", "mean_abs_residual"]
    # CLAUDE.md Phase 4 step-0 diagnostic: the univariate residual concentrates hard on
    # festivo days. This is the one criterion expected to land firmly in "structure".
    assert ratio >= sra.CALENDAR_RATIO_STRUCTURE_MIN


# --------------------------------------------------------------------------------------
# Weather correlation
# --------------------------------------------------------------------------------------


def test_weather_correlation_covers_all_105_lagged_features_and_no_same_day_column(weather_corr):
    assert len(weather_corr) == 105
    assert set(weather_corr["lag_kind"]) == set(sra.WEATHER_LAG_KINDS)
    # Re-assert the Phase 3 same-day-weather guard at the diagnostic layer.
    assert weather_corr["feature"].str.contains(r"_lag_\d+$|_roll_mean_7$", regex=True).all()
    assert not weather_corr["lag_kind"].eq("").any()


def test_partial_correlation_controls_for_the_annual_fourier_terms():
    assert sra.ANNUAL_FOURIER_COLS == [
        "feat_fourier_annual_k1_sin",
        "feat_fourier_annual_k1_cos",
        "feat_fourier_annual_k2_sin",
        "feat_fourier_annual_k2_cos",
    ]
    # Constructed case: a feature that is pure annual-Fourier signal plus noise must have
    # its correlation with an annual-Fourier-driven target collapse after partialling.
    rng = np.random.default_rng(1)
    n = 400
    season = np.sin(2 * np.pi * np.arange(n) / 365.25)
    controls = np.column_stack([season, np.cos(2 * np.pi * np.arange(n) / 365.25),
                                np.sin(4 * np.pi * np.arange(n) / 365.25),
                                np.cos(4 * np.pi * np.arange(n) / 365.25)])
    y = season + 0.1 * rng.standard_normal(n)
    x = season + 0.1 * rng.standard_normal(n)
    ry = sra._rank_residualise(y, controls)
    rx = sra._rank_residualise(x, controls)
    partial = np.corrcoef(ry, rx)[0, 1]
    raw = np.corrcoef(y, x)[0, 1]
    assert raw > 0.9
    assert abs(partial) < 0.3


def test_bh_adjustment_is_monotone_and_never_below_the_raw_pvalue(weather_corr):
    ordered = weather_corr.sort_values("p_partial").reset_index(drop=True)
    assert (ordered["p_adj_bh"] + 1e-12 >= ordered["p_partial"]).all()
    assert (ordered["p_adj_bh"].diff().dropna() >= -1e-9).all()
    assert weather_corr["p_adj_bh"].between(0, 1).all()


# --------------------------------------------------------------------------------------
# Error chain (analysis 2)
# --------------------------------------------------------------------------------------


def test_error_chain_matches_the_reported_phase5_maes():
    table = sra.error_chain_table("test").set_index("Modelo")
    assert table.loc["lstm_alone", "MAE"] == pytest.approx(280_952, abs=50)
    assert table.loc["hybrid", "MAE"] == pytest.approx(234_223, abs=50)
    assert table.loc["xgboost_alone", "MAE"] == pytest.approx(156_121, abs=50)


def test_error_chain_gap_share_sums_to_one_hundred_percent():
    for split in ("test", "val"):
        table = sra.error_chain_table(split)
        assert table["gap_share_pct"].sum() == pytest.approx(100.0, abs=1e-6)
        # lstm_alone is the reference row: zero step, zero share.
        assert table.iloc[0]["gap_share_pct"] == pytest.approx(0.0, abs=1e-9)


def test_error_chain_deliverable_sentence_numbers():
    """Stage 2 closes ~37% of the test gap; ~63% still separates it from xgboost_alone."""
    table = sra.error_chain_table("test").set_index("Modelo")
    assert table.loc["hybrid", "gap_share_pct"] == pytest.approx(37.4, abs=1.0)
    assert table.loc["xgboost_alone", "gap_share_pct"] == pytest.approx(62.6, abs=1.0)


# --------------------------------------------------------------------------------------
# Per-fold sample-size context (analysis 3a)
# --------------------------------------------------------------------------------------


def test_fold_skill_ratio_uses_seasonal_naive_on_the_identical_dates():
    table = sra.fold_size_vs_error()
    assert list(table["fold"]) == [1, 2, 3, 4, 5]

    oof = pd.read_parquet(PROCESSED_DIR / "lstm_oof_predictions.parquet")
    baselines = pd.read_parquet(PROCESSED_DIR / "baseline_predictions.parquet")
    for _, row in table.iterrows():
        block = oof[oof["fold"] == row["fold"]]
        naive = block[["date"]].merge(baselines, on="date", how="left")
        expected = mae(naive["total"], naive["seasonal_naive"])
        assert row["mae_seasonal_naive_same_block"] == pytest.approx(expected, rel=1e-9)
        assert row["skill_ratio"] == pytest.approx(row["mae"] / expected, rel=1e-9)

    # Expanding window: training size grows monotonically across folds.
    assert (table["train_rows"].diff().dropna() > 0).all()
    # ... but fold MAE does NOT (the confound this table exists to expose, objection O1).
    assert not (table["mae"].diff().dropna() < 0).all()


# --------------------------------------------------------------------------------------
# Acceptance criteria
# --------------------------------------------------------------------------------------


def test_acceptance_criteria_thresholds_are_module_constants():
    table = sra.evaluate_acceptance_criteria()
    assert len(table) == 4
    assert set(table["verdict"]) <= {
        sra.VERDICT_STRUCTURE,
        sra.VERDICT_WHITE,
        sra.VERDICT_AMBIGUOUS,
    }
    # The thresholds echoed into the table must be the module constants, not re-typed.
    by_q = table.set_index("question")
    assert by_q.loc["Memoria temporal (ACF/PACF)", "threshold_structure"] == sra.ACF_STRUCTURE_MIN
    assert by_q.loc["Memoria temporal (ACF/PACF)", "threshold_white"] == sra.ACF_WHITE_MAX
    assert by_q.loc["Concentracion en calendario", "threshold_structure"] == sra.CALENDAR_RATIO_STRUCTURE_MIN
    assert by_q.loc["Senal meteorologica (parcial, BH)", "threshold_structure"] == float(
        sra.WEATHER_MIN_SIGNIFICANT
    )

    # The evaluator must actually reference the constants in its source, not literals.
    src = inspect.getsource(sra.evaluate_acceptance_criteria)
    for name in ("ACF_STRUCTURE_MIN", "CALENDAR_RATIO_STRUCTURE_MIN", "WEATHER_MIN_SIGNIFICANT"):
        assert name in src


def test_calendar_concentration_verdict_is_structure():
    """The one criterion CLAUDE.md's diagnostics make near-certain."""
    table = sra.evaluate_acceptance_criteria().set_index("question")
    assert table.loc["Concentracion en calendario", "verdict"] == sra.VERDICT_STRUCTURE


def test_ljung_box_returns_the_requested_lags():
    table = sra.ljung_box(lags=(7, 14, 28))
    assert list(table["lag"]) == [7, 14, 28]
    assert table["lb_pvalue"].between(0, 1).all()


def test_weekday_profile_covers_seven_days_and_sums_close_to_zero(primary_residuals):
    table = sra.weekday_residual_profile()
    assert list(table["day_of_week_es"]) == sra.WEEKDAY_ORDER
    assert table["n"].sum() == len(primary_residuals)


def test_summarise_returns_every_table():
    out = sra.summarise()
    expected = {
        "autocorrelation",
        "ljung_box",
        "weekday_profile",
        "periodogram",
        "calendar_group",
        "weather_correlation",
        "error_chain_test",
        "error_chain_val",
        "fold_size",
        "acceptance",
    }
    assert set(out) == expected
    assert all(isinstance(v, pd.DataFrame) and not v.empty for v in out.values())
