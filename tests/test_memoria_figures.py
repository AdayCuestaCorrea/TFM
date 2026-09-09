"""Phase 8 tests: memoria-only figures share the VIU theme and touch no model code.

These figures illustrate data already produced by Phases 2-5b (features, OOF predictions,
ensemble predictions). Nothing here trains, refits, or loads a Keras/XGBoost model, and the
test that greps for `src.models` imports exists specifically to keep that true as the module
grows.
"""

import re
from pathlib import Path

import plotly.graph_objects as go
import pytest

from src.evaluation import memoria_figures
from src.evaluation.memoria_figures import (
    annual_seasonality_figure,
    block_ablation_figure,
    error_dispersion_figure,
    oof_fold1_degenerate_figure,
    oof_folds_figure,
    operator_demand_figure,
    parity_control_figure,
    period_rolling_mae_figure,
    splits_figure,
    stage1_acf_pacf_figure,
    stage1_error_chain_figure,
    stage1_fold_size_figure,
    stage1_learning_curve_figure,
    stage1_periodogram_figure,
    stage1_weather_corr_figure,
    stage1_weekday_figure,
    subgroup_delta_ci_figure,
    subgroup_mae_figure,
    total_series_figure,
    weather_vs_demand_figure,
    weekly_seasonality_figure,
)
from src.evaluation.theme import VIU_TEMPLATE, theme_fingerprint


# --------------------------------------------------------------------------------------
# No model logic sneaks in here
# --------------------------------------------------------------------------------------


def test_module_imports_no_model_code():
    source = Path(memoria_figures.__file__).read_text(encoding="utf-8")
    import_lines = [
        line for line in source.splitlines() if line.strip().startswith(("import ", "from "))
    ]
    assert not any("src.models" in line for line in import_lines)


def test_no_hardcoded_hex_colours():
    source = Path(memoria_figures.__file__).read_text(encoding="utf-8")
    code = "\n".join(line.split("#")[0] for line in source.splitlines())
    assert not re.search(r"#[0-9a-fA-F]{6}\b", code)


# --------------------------------------------------------------------------------------
# Every builder returns a themed figure
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "builder",
    [
        total_series_figure,
        operator_demand_figure,
        weekly_seasonality_figure,
        annual_seasonality_figure,
        weather_vs_demand_figure,
        splits_figure,
        oof_folds_figure,
        oof_fold1_degenerate_figure,
        error_dispersion_figure,
        stage1_acf_pacf_figure,
        stage1_periodogram_figure,
        stage1_weekday_figure,
        stage1_weather_corr_figure,
        stage1_error_chain_figure,
        stage1_fold_size_figure,
        stage1_learning_curve_figure,
        block_ablation_figure,
        parity_control_figure,
        subgroup_mae_figure,
        subgroup_delta_ci_figure,
        period_rolling_mae_figure,
    ],
)
def test_every_builder_returns_a_themed_figure(builder):
    fig = builder()
    assert isinstance(fig, go.Figure)
    assert fig.layout.template.layout.colorway == theme_fingerprint()


@pytest.mark.parametrize(
    "builder",
    [
        stage1_acf_pacf_figure,
        stage1_periodogram_figure,
        stage1_weekday_figure,
        stage1_weather_corr_figure,
        stage1_error_chain_figure,
        stage1_fold_size_figure,
        stage1_learning_curve_figure,
        block_ablation_figure,
        parity_control_figure,
        subgroup_mae_figure,
        subgroup_delta_ci_figure,
        period_rolling_mae_figure,
    ],
)
def test_stage1_anatomy_builders_have_no_embedded_title(builder):
    # The descriptive title lives only in the LaTeX \caption, never duplicated in the PNG.
    assert not builder().layout.title.text


def test_builders_share_the_project_template_object():
    fig = total_series_figure()
    assert fig.layout.template is VIU_TEMPLATE or (
        fig.layout.template.layout.colorway == VIU_TEMPLATE.layout.colorway
    )


# --------------------------------------------------------------------------------------
# Content sanity checks
# --------------------------------------------------------------------------------------


def test_total_series_figure_spans_the_full_history():
    fig = total_series_figure()
    x = fig.data[0].x
    assert len(x) == 1310


def test_operator_demand_figure_has_four_operator_traces():
    fig = operator_demand_figure()
    assert len(fig.data) == 4


def test_weekly_seasonality_figure_has_seven_boxes():
    fig = weekly_seasonality_figure()
    assert len(fig.data) == 7


def test_annual_seasonality_figure_has_twelve_months():
    fig = annual_seasonality_figure()
    assert len(fig.data[0].x) == 12


def test_splits_figure_matches_the_official_split_boundaries():
    # The descriptive title was removed from the figure itself (it now lives only in the
    # LaTeX caption); this test instead verifies the three shaded regions are the ones
    # actually labelled, via the vrect annotations that survive in the figure.
    fig = splits_figure()
    region_labels = {a.text for a in fig.layout.annotations}
    assert region_labels == {"Train", "Val", "Test"}


def test_splits_figure_has_no_embedded_title():
    fig = splits_figure()
    assert not fig.layout.title.text


def test_oof_folds_figure_excludes_the_no_oof_rows():
    fig = oof_folds_figure()
    # 1 actual trace + 5 OOF fold traces (folds 1-5); fold -1 (no OOF) never appears.
    assert len(fig.data) == 6
    for trace in fig.data[1:]:
        assert "fold -1" not in trace.name.lower()


def test_oof_fold1_degenerate_figure_reports_the_documented_std_ratio():
    """CLAUDE.md pins fold 1's OOF std ratio at ~0.199 and correlation at ~0.300."""
    fig = oof_fold1_degenerate_figure()
    title_text = fig.layout.annotations[0].text
    assert "0.19" in title_text or "0.20" in title_text


def test_error_dispersion_figure_reports_a_correlation_annotation():
    fig = error_dispersion_figure()
    annotations = [a.text for a in fig.layout.annotations]
    assert any("corr" in a for a in annotations)


def test_error_dispersion_figure_shows_weak_correlation():
    """The whole point of the ensemble: SARIMAX and XGBoost-alone errors are decorrelated."""
    fig = error_dispersion_figure()
    corr_text = next(a.text for a in fig.layout.annotations if "corr" in a.text)
    value = float(corr_text.split("=")[1].strip())
    assert abs(value) < 0.6


# --------------------------------------------------------------------------------------
# Stage 1 residual anatomy builders (Phase 7 diagnostic)
# --------------------------------------------------------------------------------------


def test_acf_pacf_figure_covers_lags_to_35():
    fig = stage1_acf_pacf_figure()
    # Two band traces + two bar traces (ACF, PACF).
    bars = [t for t in fig.data if t.type == "bar"]
    assert len(bars) == 2
    assert max(bars[0].x) == 35


def test_periodogram_axis_stays_within_the_representable_band():
    # REGRESSION (§5.9 review): the axis once ran to ~4e7 days because add_vline's
    # annotation is read as an already-log10 coordinate on a log axis. With 393
    # observations no Fourier bin exists above 393 days, so no part of the axis may.
    import math

    from src.evaluation import stage1_residual_anatomy as sra

    table = sra.residual_periodogram()
    lo, hi = float(table["period_days"].min()), float(table["period_days"].max())

    fig = stage1_periodogram_figure()
    axis_lo, axis_hi = fig.layout.xaxis.range          # log10 units
    assert axis_hi <= math.log10(hi) + 0.1
    assert axis_lo >= math.log10(lo) - 0.1
    # The band is the full representable one (Nyquist to fundamental), not N/2: the
    # period-196.5 bin is the second-highest peak and must stay on the page.
    assert axis_hi >= math.log10(hi) - 0.1
    # Nothing the figure draws may sit outside the plotted band either.
    for ann in fig.layout.annotations:
        if ann.xref in ("x", None):
            assert axis_lo <= ann.x <= axis_hi


def test_periodogram_period7_marker_is_placed_in_log_coordinates():
    import math

    fig = stage1_periodogram_figure()
    label = next(a for a in fig.layout.annotations if "7" in a.text)
    assert label.x == pytest.approx(math.log10(7.0), abs=1e-9)
    # The rule itself is a shape, which Plotly *does* convert, so it stays in data units.
    line = next(s for s in fig.layout.shapes if s.type == "line")
    assert line.x0 == pytest.approx(7.0)


def test_weekday_figure_annotates_sample_size_under_each_tick():
    fig = stage1_weekday_figure()
    ticktext = "".join(fig.layout.xaxis.ticktext)
    assert "n = " in ticktext


def test_weather_corr_figure_shows_raw_and_partial():
    fig = stage1_weather_corr_figure()
    names = {t.name for t in fig.data}
    assert "ρ parcial" in names and "ρ bruto" in names


def test_error_chain_figure_has_three_models():
    fig = stage1_error_chain_figure("test")
    bar = next(t for t in fig.data if t.type == "bar")
    assert set(bar.y) == {"lstm_alone", "hybrid", "xgboost_alone"}


def test_error_chain_annotations_sit_between_bars():
    # A step label describes the PAIR of bars it separates, so it belongs in the gap
    # between them — never anchored on one bar (which collided with its neighbour and
    # placed the label on the side of the pair it does not describe).
    fig = stage1_error_chain_figure("test")
    bar = next(t for t in fig.data if t.type == "bar")
    positions = set(range(len(bar.y)))

    steps = [a for a in fig.layout.annotations if a.text.startswith("Δ")]
    assert len(steps) == 2
    for ann in steps:
        assert ann.y not in positions          # not on a bar
        assert min(positions) < ann.y < max(positions)
        assert ann.y * 2 == int(ann.y * 2)     # exactly half-way between two categories
        # Both pieces of information survive the repositioning.
        assert "% del hueco" in ann.text
        assert re.search(r"Δ\s-?[\d,]+", ann.text)


def test_fold_size_figure_marks_fold_one_degenerate():
    fig = stage1_fold_size_figure()
    # Fold 1 uses the 'x' marker symbol in both panels.
    for trace in fig.data:
        assert trace.marker.symbol[0] == "x"


def test_learning_curve_figure_draws_both_reference_lines():
    fig = stage1_learning_curve_figure()
    ref_texts = " ".join(a.text for a in fig.layout.annotations)
    assert "SARIMAX" in ref_texts and "xgboost_alone" in ref_texts


def test_learning_curve_figure_marks_degenerate_fractions_hollow():
    fig = stage1_learning_curve_figure()
    median_trace = next(t for t in fig.data if t.mode == "lines+markers")
    # Fraction 0.25 (222 rows) collapsed on every seed -> first marker is hollow.
    assert median_trace.marker.symbol[0] == "circle-open"
    assert median_trace.marker.symbol[-1] == "circle"


# --------------------------------------------------------------------------------------
# Phase 8 diagnostic controls (informational parity and block ablation)
# --------------------------------------------------------------------------------------


def test_block_ablation_figure_has_completo_reference_and_five_bars():
    fig = block_ablation_figure("val")
    bar = next(t for t in fig.data if t.type == "bar")
    assert len(bar.y) == 5
    vlines = [s for s in fig.layout.shapes if s.type == "line"]
    assert vlines, "expected an add_vline reference at the completo MAE"


def test_parity_control_figure_draws_three_reference_lines():
    fig = parity_control_figure()
    ref_texts = " ".join(a.text for a in fig.layout.annotations)
    assert "univariante" in ref_texts
    assert "SARIMAX" in ref_texts and "xgboost_alone" in ref_texts
    scatter = next(t for t in fig.data if t.mode == "markers")
    assert len(scatter.x) == 2  # calendario, completo


# --------------------------------------------------------------------------------------
# Phase 9 diagnostic (subgroup and period breakdown)
# --------------------------------------------------------------------------------------


def test_subgroup_mae_figure_has_one_trace_per_reduced_model():
    from src.evaluation.memoria_figures import SUBGROUP_FIGURE_MODELS

    fig = subgroup_mae_figure("test")
    assert {t.name for t in fig.data} == set(SUBGROUP_FIGURE_MODELS)


def test_subgroup_delta_ci_figure_marks_the_zero_line():
    fig = subgroup_delta_ci_figure("test")
    vlines = [s for s in fig.layout.shapes if s.type == "line"]
    assert any(s.x0 == 0 for s in vlines)
    # Inferential and non-inferential subgroups are drawn as separate traces.
    names = {t.name for t in fig.data}
    assert any("no inferencial" in n for n in names)


def test_period_rolling_mae_figure_has_two_panels_and_a_boundary_rule():
    fig = period_rolling_mae_figure()
    # Panel 2 carries the zero reference; a dotted val|test rule sits in both panels.
    hlines = [s for s in fig.layout.shapes if s.type == "line"]
    assert len(hlines) >= 3
    ann = " ".join(a.text for a in fig.layout.annotations)
    assert "val" in ann and "test" in ann
