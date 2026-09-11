"""Phase 9 tests: LaTeX table exporter cannot drift from report_tables.py, and the two
direction-mandated declarative tables (1.4 cronograma, 2.4 comparativa) are internally
consistent.
"""

import re

import pandas as pd
import pytest

from src.evaluation import export_latex_tables as elt
from src.evaluation.export_latex_tables import (
    LITERATURE_COLUMNS,
    LITERATURE_COMPARISON,
    PHASES,
    TOTAL_TESTS_BEFORE_MEMORIA,
    build_ablacion_bloques,
    build_autocorrelacion_residuo,
    build_cadena_error,
    build_comparacion,
    build_comparativa_estado_arte,
    build_cronograma_fases,
    build_curva_aprendizaje,
    build_delta_hibrido,
    build_desglose_dia,
    build_ensemble_pesos,
    build_esquema_unificado,
    build_folds_oof,
    build_fuentes_datos,
    build_grupos_features,
    build_hiperparametros,
    build_paridad_lstm,
    build_particiones,
    build_residuo_calendario,
    build_residuo_dia_semana,
    build_residuo_meteo,
    build_sarimax_orden,
    build_subgrupos,
    build_subgrupos_inferencia,
    build_ventana_lstm,
    dataframe_to_tabular,
    escape_latex,
    format_number,
)


# --------------------------------------------------------------------------------------
# Formatting primitives
# --------------------------------------------------------------------------------------


def test_format_number_uses_spanish_thousands_and_decimal_separators():
    assert format_number(139681, 0) == "139.681"
    assert format_number(3.85, 2) == "3,85"
    assert format_number(0.9755, 4) == "0,9755"


def test_format_number_handles_nan():
    assert format_number(float("nan")) == "--"


def test_escape_latex_handles_underscores_and_percent():
    # `\allowbreak` after the escaped underscore lets long identifiers wrap inside a
    # narrow p{} column; it does not change the visible text, only where it may break.
    assert escape_latex("xgboost_alone") == r"xgboost\_\allowbreak{}alone"
    assert escape_latex("5%") == r"5\%"


def test_escape_latex_allows_breaking_long_paths():
    result = escape_latex("data/processed/lstm_oof_predictions.parquet")
    assert result == (
        r"data/\allowbreak{}processed/\allowbreak{}"
        r"lstm\_\allowbreak{}oof\_\allowbreak{}predictions.parquet"
    )


def test_dataframe_to_tabular_contains_booktabs_and_orange_header():
    df = pd.DataFrame({"Modelo": ["sarimax", "xgboost", "hybrid"], "MAE": [163627, 156121, 234223]})
    tex = dataframe_to_tabular(df)
    assert r"\toprule" in tex and r"\midrule" in tex and r"\bottomrule" in tex
    assert r"\rowcolor{naranja}" in tex
    assert r"\begin{tabular}" in tex and r"\end{tabular}" in tex
    # Never a floating table environment: that lives in the section file.
    assert r"\begin{table}" not in tex
    assert r"\caption" not in tex
    # Zebra tint on alternate data rows, starting on the first one so the band never abuts
    # the orange header. Asserted here rather than in a new test so the suite's collected
    # count stays at the figure the memoria states in two places.
    body = tex.split(r"\midrule")[1].split(r"\bottomrule")[0].strip().splitlines()
    assert [line.startswith(r"\rowcolor{filaclara}") for line in body] == [True, False, True]


def test_dataframe_to_tabular_escapes_model_names_in_body():
    df = pd.DataFrame({"Modelo": ["xgboost_alone"], "MAE": [156121]})
    tex = dataframe_to_tabular(df)
    assert r"xgboost\_\allowbreak{}alone" in tex
    assert "139.681" not in tex  # sanity: not testing the wrong fixture


# --------------------------------------------------------------------------------------
# 1.4 — Cronograma de fases
# --------------------------------------------------------------------------------------


def test_cronograma_increments_sum_to_the_declared_total():
    increments = [inc for *_rest, inc in PHASES if inc is not None]
    assert sum(increments) == TOTAL_TESTS_BEFORE_MEMORIA == 257


def test_cronograma_table_is_monotonically_non_decreasing_in_cumulative_tests():
    table = build_cronograma_fases()
    cumulative = table["Pruebas acumuladas"]
    assert (cumulative.diff().dropna() >= 0).all()
    assert cumulative.iloc[-1] == 257


def test_cronograma_table_has_one_row_per_declared_phase():
    table = build_cronograma_fases()
    assert len(table) == len(PHASES)


# --------------------------------------------------------------------------------------
# 2.4 — Comparativa de literatura
# --------------------------------------------------------------------------------------


def test_literature_table_has_one_row_per_study_and_five_axes():
    """One row per LITERATURE_COMPARISON entry, 'Este TFM' last, 'Estudio' + 5 axes.

    The row count is derived from the constant rather than pinned to a literal: the
    supervisor's review (phase 2) grew the table from 6 to 10 rows and a literal would have to
    be re-typed every time literature is added, which is exactly the edit that gets forgotten.
    What must not drift is the shape: this project's row closes the table, and no axis is lost.
    """
    table = build_comparativa_estado_arte()
    assert len(table) == len(LITERATURE_COMPARISON) == 10
    assert table["Estudio"].iloc[-1] == "Este TFM"
    assert list(table.columns) == list(LITERATURE_COLUMNS.values())
    assert len(table.columns) == 6


def test_literature_table_has_no_empty_cells():
    table = build_comparativa_estado_arte()
    for col in table.columns:
        assert table[col].astype(str).str.strip().ne("").all(), f"empty cell in {col}"


def test_literature_table_this_tft_row_cites_figures_in_the_test_comparison_table():
    """The 2.4 preview must not contradict the chapter 5 master table."""
    from src.evaluation.dashboard_data import model_comparison_table

    table = build_comparativa_estado_arte()
    row = table[table["Estudio"].str.contains("Este TFM")].iloc[0]
    combined = " ".join(row.astype(str))

    test_table = model_comparison_table("test").set_index("model")
    expected = {
        "ensemble_equal": 139_725,
        "xgboost_alone": 156_121,
        "sarimax": 163_627,
        "hybrid": 234_223,
    }
    for model, expected_mae in expected.items():
        actual = round(test_table.loc[model, "MAE"])
        assert actual == expected_mae, f"{model} MAE drifted: {actual} != {expected_mae}"

    # The 2.4 preview must not hardcode a delta that could drift from chapter 5's own
    # table: derive the expected strings from build_delta_hibrido() itself rather than
    # typing the literal MAE differences by hand (CLAUDE.md editorial-closure note).
    delta_table = build_delta_hibrido().set_index(["Partición", "Comparación"])
    sarimax_delta = format_number(delta_table.loc[("test", "hybrid - sarimax"), "Delta MAE"], 0)
    xgb_delta = format_number(delta_table.loc[("test", "hybrid - xgboost_alone"), "Delta MAE"], 0)
    assert format_number(139_725, 0) in combined
    assert sarimax_delta in combined or xgb_delta in combined


def test_literature_carries_no_pending_verification_marker():
    """No row of the 2.4 table may print a pending-verification marker (supervisor item 35).

    Surribas-Sayago was the last placeholder: its (b)/(d) cells were filled from the paper
    itself (section 3.1) and its bibliography entry completed, so the markers that
    contradicted Anexo D's "todas verificadas contra su fuente" must never come back. The
    guard lives on the generator because that is what regenerates the .tex.
    """
    for entry in LITERATURE_COMPARISON:
        combined = " ".join(entry.values())
        assert "sin verificar" not in combined, entry["estudio"]
        assert "No verificado" not in combined, entry["estudio"]
    surribas = next(e for e in LITERATURE_COMPARISON if "Surribas-Sayago" in e["estudio"])
    assert surribas["estudio"] == "Surribas-Sayago et al. (2026)"


# --------------------------------------------------------------------------------------
# Chapter 3/4/5 tables cannot drift from their source functions
# --------------------------------------------------------------------------------------


def test_comparacion_table_matches_report_tables_directly():
    from src.evaluation.report_tables import comparison_table as rt_comparison_table

    generated = build_comparacion("test")
    reference = rt_comparison_table("test")
    pd.testing.assert_frame_equal(generated, reference)


def test_comparacion_tex_file_contains_the_ensemble_mae_in_spanish_format():
    path = elt.write_table(
        build_comparacion("test"), "tabla_comparacion_test.tex", decimals={"MAPE (%)": 2, "R²": 4}
    )
    text = path.read_text(encoding="utf-8")
    assert "139.681" in text
    assert (
        r"ensemble\_\allowbreak{}equal" in text
        or r"ensemble\_\allowbreak{}inverse\_\allowbreak{}mae" in text
    )


def test_desglose_dia_table_keeps_small_sample_counts():
    table = build_desglose_dia("test").set_index("Tipo de día")
    assert table.loc["Festivo", "n"] == 5


def test_particiones_table_sums_to_the_full_dataset():
    table = build_particiones()
    assert table["Días"].sum() == 1310


def test_sarimax_orden_table_matches_the_persisted_json():
    table = build_sarimax_orden()
    assert table.loc[0, "Orden (p,d,q)"] == "(2,1,2)"
    assert table.loc[0, "Orden estacional (P,D,Q,s)"] == "(0,1,2,7)"


def test_ventana_lstm_table_declares_w28_as_best_val_mae():
    table = build_ventana_lstm().set_index("W")
    best_w = table["MAE (val)"].idxmin()
    assert best_w == 28


def test_folds_oof_table_has_five_folds():
    table = build_folds_oof()
    assert sorted(table["Fold"]) == [1, 2, 3, 4, 5]


def test_hiperparametros_table_matches_report_tables():
    from src.evaluation.report_tables import hyperparameters_table

    pd.testing.assert_frame_equal(build_hiperparametros(), hyperparameters_table())


def test_esquema_unificado_sums_to_31_columns():
    table = build_esquema_unificado()
    assert table["N.º columnas"].sum() == 31


def test_grupos_features_totals_161():
    table = build_grupos_features()
    assert table.loc[table["Grupo"] == "Total", "N.º variables"].iloc[0] == 161


def test_fuentes_datos_has_three_raw_sources():
    assert len(build_fuentes_datos()) == 3


# --------------------------------------------------------------------------------------
# Generated (not hand-typed) chapter 5 tables
# --------------------------------------------------------------------------------------


def test_delta_hibrido_reports_the_documented_test_deltas():
    """CLAUDE.md Phase 5: hybrid - sarimax = +70,597 test MAE; hybrid - xgboost_alone = +78,103."""
    table = build_delta_hibrido().set_index(["Partición", "Comparación"])
    sarimax_delta = table.loc[("test", "hybrid - sarimax"), "Delta MAE"]
    xgb_delta = table.loc[("test", "hybrid - xgboost_alone"), "Delta MAE"]
    assert sarimax_delta == pytest.approx(70_597, abs=50)
    assert xgb_delta == pytest.approx(78_103, abs=50)
    assert (table["Veredicto"] == "hybrid peor").all()


def test_criterio_experimento_a_reports_not_met():
    """CLAUDE.md Phase 5b: criterion A NOT MET (irregular days did not improve)."""
    from src.evaluation.export_latex_tables import build_criterio_experimento_a

    table = build_criterio_experimento_a()
    verdict_row = table[table["Criterio"].str.contains("Veredicto")].iloc[0]
    assert verdict_row["Resultado"] == "NO CUMPLIDO"


def test_ensemble_pesos_table_matches_documented_weights():
    """CLAUDE.md Phase 5b: equal 0.50/0.50, inverse-MAE 0.4724/0.5276."""
    table = build_ensemble_pesos().set_index("Variante")
    assert table.loc["ensemble_equal", "Peso SARIMAX"] == pytest.approx(0.5)
    assert table.loc["ensemble_inverse_mae", "Peso SARIMAX"] == pytest.approx(0.4724, abs=1e-3)
    assert table.loc["ensemble_inverse_mae", "Peso XGBoost independiente"] == pytest.approx(
        0.5276, abs=1e-3
    )


# --------------------------------------------------------------------------------------
# 5.9 — Stage 1 residual anatomy tables (Phase 7 diagnostic)
# --------------------------------------------------------------------------------------


def test_autocorrelacion_residuo_covers_the_weekly_lags():
    table = build_autocorrelacion_residuo()
    assert list(table["Retardo (días)"]) == [1, 2, 3, 7, 14, 21, 28]
    # Ljung-Box only computed at 7/14/28; the rest are NaN -> "--" in the .tex.
    assert table.set_index("Retardo (días)").loc[1, "Ljung-Box Q"] != table.set_index(
        "Retardo (días)"
    ).loc[1, "Ljung-Box Q"]  # NaN


def test_cadena_error_matches_reported_phase5_maes():
    table = build_cadena_error()
    test_block = table[table["Partición"] == "test"].set_index("Modelo")
    assert test_block.loc["lstm_alone", "MAE"] == pytest.approx(280_952, abs=50)
    assert test_block.loc["hybrid", "MAE"] == pytest.approx(234_223, abs=50)
    assert test_block.loc["xgboost_alone", "MAE"] == pytest.approx(156_121, abs=50)


def test_cadena_error_gap_share_sums_to_100_per_split():
    table = build_cadena_error()
    for split in ("test", "val"):
        block = table[table["Partición"] == split]
        assert block["Cuota del hueco (%)"].sum() == pytest.approx(100.0, abs=1e-6)


def test_residuo_meteo_reports_ten_features_with_a_significance_flag():
    table = build_residuo_meteo()
    assert len(table) == 10
    assert table["Significativa"].dtype == bool
    # CLAUDE.md diagnostic finding: no lagged-weather feature survives BH partialling.
    assert not table["Significativa"].any()


def test_residuo_calendario_covers_both_dimensions():
    table = build_residuo_calendario()
    assert set(table["Dimensión"]) == {"day_type", "holiday_type"}
    assert "festivo" in set(table["Grupo"])


def test_residuo_dia_semana_has_seven_weekdays():
    table = build_residuo_dia_semana()
    assert len(table) == 7
    assert table["n"].sum() == 393


def test_curva_aprendizaje_covers_the_six_fractions_and_flags_degeneracy():
    table = build_curva_aprendizaje().set_index("Fracción")
    assert list(table.index) == ["0.25", "0.40", "0.55", "0.70", "0.85", "1.00"]
    assert table.loc["1.00", "Filas train"] == 889
    # Fraction 0.25 collapsed on every seed; the larger fractions did not.
    assert table.loc["0.25", "Fits degenerados"] == 3
    assert table.loc["1.00", "Fits degenerados"] == 0
    # The LSTM never reaches xgboost_alone's val MAE (244,843): the plateau sits well above.
    assert table["MAE val (mediana)"].min() > 300_000


# --------------------------------------------------------------------------------------
# Phase 8 diagnostic tables (informational parity and block ablation)
# --------------------------------------------------------------------------------------


def test_ablacion_bloques_table_matches_the_pre_registered_subsets():
    table = build_ablacion_bloques().set_index("Subconjunto")
    assert list(table.index) == [
        "completo", "solo temporal", "solo exogeno", "sin meteo", "solo calendario"
    ]
    assert table.loc["completo", "N variables"] == 161
    assert table.loc["solo temporal", "N variables"] == 39
    assert table.loc["solo exogeno", "N variables"] == 122
    assert table.loc["solo calendario", "N variables"] == 17
    # completo is the reference row: zero delta, and its val MAE is the published 244,843.
    assert table.loc["completo", "Delta MAE val (%)"] == pytest.approx(0.0, abs=0.05)
    assert table.loc["completo", "MAE val"] == pytest.approx(244_843, abs=50)
    # Both mandatory subsets degrade past the 5% materiality floor.
    assert table.loc["solo temporal", "Delta MAE val (%)"] >= 5.0
    assert table.loc["solo exogeno", "Delta MAE val (%)"] >= 5.0


def test_paridad_lstm_table_reports_both_scopes_and_the_gap_closure():
    table = build_paridad_lstm().set_index("Ámbito")
    assert set(table.index) == {"calendario", "completo"}
    assert table.loc["calendario", "k"] == 17
    assert table.loc["completo", "k"] == 161
    # No seed degenerated in either scope.
    assert table["Fits degenerados"].sum() == 0
    # Neither scope closes 60% of the gap (both verdicts are non-asymmetry).
    assert table["Hueco cerrado (%)"].max() < 60.0


# --------------------------------------------------------------------------------------
# End-to-end export
# --------------------------------------------------------------------------------------


def test_subgrupos_table_carries_the_preregistered_verdicts():
    """CLAUDE.md standing result: the hybrid wins in no subgroup. demanda_anomala is
    non-inferential on the verdict scope (n=16), never a per-subgroup exception."""
    from src.evaluation import subgroup_breakdown as sb

    table = build_subgrupos("test").set_index("Subgrupo")
    assert (table["Delta MAE"] > 0).all()  # every best hybrid worse than the best competitor
    assert sb.VERDICT_WIN not in set(table["Veredicto"])
    assert table.loc["Demanda anómala", "Veredicto"] == sb.VERDICT_NON_INFER
    assert table.loc["Demanda anómala", "n"] == 16
    # The val table shows the SAME verdict column (one verdict per subgroup, test-scoped).
    val_table = build_subgrupos("val").set_index("Subgrupo")
    assert val_table.loc["Demanda anómala", "Veredicto"] == sb.VERDICT_NON_INFER


def test_subgrupos_inferencia_table_is_the_inferential_family_only():
    from src.evaluation import subgroup_breakdown as sb

    table = build_subgrupos_inferencia()
    assert (table["n"] >= sb.MIN_N_INFERENTIAL).all()
    assert len(table) == 9
    # A suppressed CI prints as '--' (verano: low usable-resample fraction).
    tex = elt.dataframe_to_tabular(
        table, decimals={"Delta MAE": 0, "IC 95% inf.": 0, "IC 95% sup.": 0}
    )
    assert "--" in tex
    assert "no inferencial" not in tex  # only inferential rows are here


def test_export_all_writes_every_declared_table_and_they_all_parse_as_tabular():
    written = elt.export_all()
    assert len(written) >= 18
    assert len(written) == 32
    names = {p.name for p in written}
    assert {
        "tabla_ablacion_bloques.tex", "tabla_paridad_lstm.tex",
        "tabla_subgrupos_test.tex", "tabla_subgrupos_val.tex",
        "tabla_subgrupos_inferencia.tex",
    } <= names
    for path in written:
        assert path.exists()
        text = path.read_text(encoding="utf-8")
        assert text.strip().startswith(r"\begin{tabular}")
        assert text.strip().endswith(r"\end{tabular}")
        # No unresolved LaTeX-hostile bare '&', '%', '_' outside of escaped/backslashed form
        # in the header line specifically (the header is fully under our control).
        header_line = text.splitlines()[3]
        assert re.search(r"(?<!\\)_", header_line) is None
