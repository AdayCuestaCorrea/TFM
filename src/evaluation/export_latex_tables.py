"""Fase 9 — dump every memoria table to `docs/LaTeX/tables/*.tex` as a bare `tabular`.

Run as a standalone script:

    python -m src.evaluation.export_latex_tables

DESIGN. Each `.tex` file contains ONLY the `tabular` environment (booktabs rules +
`\\rowcolor{naranja}` header) — never `\\begin{table}`, `\\caption` or `\\label`. Those live
in the section files (`docs/LaTeX/sections/*.tex`) so a `\\caption` can be hand-edited without
being clobbered by a re-run, and a re-run never overwrites prose.

NOTHING IS RETRAINED. Every number comes from `report_tables.py` / `dashboard_data.py`
(themselves reading Phase 3-5b artifacts) or from two declarative constants below
(`PHASES`, `LITERATURE_COMPARISON`) whose content is documentary, not derived. A test pins
that the "Este TFM" row of the literature table cites figures that also exist in
`comparison_table("test")`, so the chapter 2 preview cannot contradict chapter 5.

NUMBER FORMAT — Spanish convention: point as thousands separator, comma as decimal
separator (`139.681`, `3,85`, `0,9755`). `_format_number` does the swap explicitly rather
than relying on locale, which is unset/inconsistent across machines that might run this.
"""

import json
from pathlib import Path

import pandas as pd

from src.evaluation import report_tables
from src.evaluation.day_type_breakdown import BRIDGE_COL, DAY_TYPE_COL
from src.evaluation.dashboard_data import (
    MODEL_NAMES,
    XGB_ALONE_MODEL,
    XGB_RESIDUAL_MODEL,
    _require,
    model_comparison_table,
)
from src.evaluation.ensemble_baseline import equal_weights, inverse_mae_weights
from src.evaluation.metrics import mae
from src.features.build_features import FEATURES_DAILY_FILE
from src.utils.paths import PROCESSED_DIR, PROJECT_ROOT
from src.utils.splits import chronological_split

TABLES_DIR: Path = PROJECT_ROOT / "docs" / "LaTeX" / "tables"

SARIMAX_ORDER_FILE: Path = PROCESSED_DIR / "sarimax_selected_order.json"
LSTM_WINDOW_COMPARISON_FILE: Path = PROCESSED_DIR / "lstm_window_comparison.parquet"
LSTM_OOF_FOLD_LOG_FILE: Path = PROCESSED_DIR / "lstm_oof_fold_log.parquet"
ENSEMBLE_PREDICTIONS_FILE: Path = PROCESSED_DIR / "ensemble_predictions.parquet"


# --------------------------------------------------------------------------------------
# LaTeX formatting primitives
# --------------------------------------------------------------------------------------

_LATEX_SPECIAL = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def escape_latex(text: object) -> str:
    """Escape LaTeX-special characters in a table cell. Numbers pass through untouched.

    Also inserts `\\allowbreak` after path separators (`/`) and escaped underscores, so
    that long filesystem paths and identifiers (e.g. 'data/processed/lstm_oof_predictions
    .parquet') can wrap inside a narrow `p{}` column instead of producing an unbreakable
    token that overflows into the margin. LaTeX's default hyphenation never breaks at
    those characters on its own, since `_` becomes the control sequence `\\_` and `/`
    carries no hyphenation points for a technical string like this.
    """
    if not isinstance(text, str):
        return str(text)
    out = []
    for ch in text:
        out.append(_LATEX_SPECIAL.get(ch, ch))
    escaped = "".join(out)
    escaped = escaped.replace("/", r"/\allowbreak{}")
    escaped = escaped.replace(r"\_", r"\_\allowbreak{}")
    return escaped


def format_number(value: float, decimals: int = 0) -> str:
    """Format a number in Spanish convention: '.' for thousands, ',' for decimals.

    `139681` at 0 decimals -> '139.681'. `3.85` at 2 decimals -> '3,85'. Implemented as an
    explicit swap rather than a locale, because locale availability is not guaranteed on
    every machine that might run this exporter.
    """
    if pd.isna(value):
        return "--"
    text = f"{value:,.{decimals}f}"
    return text.translate(str.maketrans({",": "\x00", ".": ","})).replace("\x00", ".")


def dataframe_to_tabular(
    df: pd.DataFrame,
    decimals: dict[str, int] | None = None,
    column_spec: str | None = None,
    url_columns: set[str] | None = None,
) -> str:
    """Render `df` as a bare booktabs `tabular` with an orange header row.

    Args:
        df: Table to render. Column names become the (escaped) header labels.
        decimals: {column: decimal places} for float columns formatted in Spanish
            convention. Columns not listed use `format_number` with 0 decimals if numeric,
            or plain `escape_latex` if not.
        column_spec: Explicit LaTeX column spec (e.g. 'lrrr'). Defaults to 'l' for
            object/string columns and 'r' for numeric columns, inferred per column.
        url_columns: Columns whose cells hold a bare URL. They are wrapped in `\\url{}`
            and NOT passed through `escape_latex`: hyperref's `\\url` is verbatim-like, so
            escaping first would print the escape sequences literally, and its own break
            points after `/` and `-` are what let a long address wrap inside a `p{}`
            column. The DataFrame keeps the plain URL, so the value stays checkable
            against the live address without stripping markup.

    Returns:
        The full `\\begin{tabular}...\\end{tabular}` block as a string, ending with a
        newline, ready to be written directly to a `.tex` file.
    """
    decimals = decimals or {}
    url_columns = url_columns or set()

    if column_spec is None:
        column_spec = "".join(
            "r" if pd.api.types.is_numeric_dtype(df[col]) else "l" for col in df.columns
        )

    header = " & ".join(rf"\textbf{{{escape_latex(col)}}}" for col in df.columns) + r" \\"

    # Zebra tint on alternate data rows: with six numeric columns and up to 15 rows, plain
    # booktabs rules give the eye nothing to track a row by, and readers slip a line. A tint
    # band is the only one of the three candidate treatments (the others being \addlinespace
    # at group boundaries and a larger \arraystretch) that costs zero vertical space, which a
    # memoria against a hard page limit cannot spend. It is a tint of the same institutional
    # orange already carrying the header, so the table still reads as one object, and it
    # survives the \resizebox wrappers at the \input sites because colortbl rules scale with
    # the box instead of reflowing. Emitted here rather than at the \input site: these files
    # are regenerated by export_all() and a hand edit would be silently reverted.
    body_lines = []
    for position, (_, row) in enumerate(df.iterrows()):
        cells = []
        for col in df.columns:
            value = row[col]
            if col in url_columns:
                cells.append(rf"\url{{{value}}}")
            elif pd.api.types.is_bool_dtype(type(value)) or isinstance(value, bool):
                cells.append("Sí" if value else "No")
            elif pd.api.types.is_numeric_dtype(df[col]):
                cells.append(format_number(value, decimals.get(col, 0)))
            else:
                cells.append(escape_latex(value))
        # Start on the first data row, so the tint alternates against the orange header
        # rather than abutting it.
        prefix = r"\rowcolor{filaclara} " if position % 2 == 0 else ""
        body_lines.append(prefix + " & ".join(cells) + r" \\")

    lines = [
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        r"\rowcolor{naranja}",
        header,
        r"\midrule",
        *body_lines,
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return "\n".join(lines) + "\n"


def write_table(
    df: pd.DataFrame,
    filename: str,
    decimals: dict[str, int] | None = None,
    column_spec: str | None = None,
    url_columns: set[str] | None = None,
) -> Path:
    """Render and write one table to `docs/LaTeX/tables/<filename>`."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLES_DIR / filename
    path.write_text(
        dataframe_to_tabular(df, decimals, column_spec, url_columns), encoding="utf-8"
    )
    return path


# --------------------------------------------------------------------------------------
# 1.4 — Project lifecycle table (declarative: NOT derived from data/processed/)
# --------------------------------------------------------------------------------------

# (fase, denominacion, entregable, artefacto, tests_incremento). tests_incremento is None
# for phases with no pytest suite of their own (Fase 0, and the ongoing memoria phases
# 7-11, which are writing/exporting, not modelling code under test).
PHASES: list[tuple[str, str, str, str, int | None]] = [
    ("Fase 0", "Andamiaje",
     "Estructura src/, venv, requirements.txt, contexto técnico del proyecto",
     "Documento de contexto técnico", None),
    ("Fase 1", "Ingesta y unificación", "unified_daily.parquet (1.310 x 31)",
     "data/interim/unified_daily.parquet", 30),
    ("Fase 2", "Ingeniería de características anti-fuga",
     "features_daily.parquet (1.310 x 87)", "data/processed/features_daily.parquet", 36),
    ("Fase 3", "Particiones, promoción meteorológica y baselines",
     "105 vars. meteorológicas retardadas, chronological_split, SARIMAX",
     "data/processed/baseline_predictions.parquet", 45),
    ("Fase 4", "LSTM etapa 1 con validación OOF",
     "Walk-forward expansivo 5 folds, lstm_stage1_final.keras",
     "data/processed/lstm_oof_predictions.parquet", 27),
    ("Fase 5", "Híbrido residual",
     "Conjunto residual 552 filas, xgboost_residual.joblib",
     "data/processed/hybrid_predictions.parquet", 22),
    ("Fase 5b", "Análisis de sensibilidad prerregistrado",
     "Experimento A (repesado 3x) y B (ensamble)",
     "data/processed/sensitivity_comparison.parquet", 17),
    ("Fase 6", "Cuadro de mando de resultados", "Plotly + ipywidgets, 13 PNG",
     "reports/figures/*.png", 40),
    ("Fase 6b", "Refinamiento visual", "Tema VIU compartido, tablas copiables",
     "src/evaluation/theme.py", 40),
    ("Fases 7-11", "Redacción de la memoria",
     "Modularización LaTeX, figuras y tablas de memoria, REPORT_MAPPING.md",
     "docs/LaTeX/", None),
]

TOTAL_TESTS_BEFORE_MEMORIA = 257


def build_cronograma_fases() -> pd.DataFrame:
    """Fase / Denominación / Entregable principal / Artefacto persistido / Pruebas acumuladas.

    'Pruebas acumuladas' is a running sum, not a per-phase count: it is the evidence that the
    lifecycle was incremental and verified at every step, not a narration written after the
    fact. Rows with no test suite of their own (Fase 0, Fases 7-11) repeat the last
    cumulative total instead of leaving a gap.
    """
    rows = []
    cumulative = 0
    for fase, nombre, entregable, artefacto, increment in PHASES:
        if increment is not None:
            cumulative += increment
        rows.append(
            {
                "Fase": fase,
                "Denominación": nombre,
                "Entregable principal": entregable,
                "Artefacto persistido": artefacto,
                "Pruebas acumuladas": cumulative,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# 2.4 — Literature comparison table (declarative)
# --------------------------------------------------------------------------------------

LITERATURE_COMPARISON: list[dict[str, str]] = [
    {
        "estudio": "Monje et al. (2022)",
        "ambito": "Una línea de autobús (EMT línea 1), solo franja de tarde",
        "horizonte": "26 meses pre-pandemia (ene. 2015 - feb. 2017)",
        "exogenas": "Festivo binario + calendario (mes, día de semana); cero meteorología",
        "protocolo": "Holdout cronológico simple 80/20, una sola ejecución, sin CV",
        "aporte": "Reporta el ganador (LSTM R²=0,89) sin MAE/RMSE/MAPE absolutos",
    },
    {
        # Fase 2 de la revision de direccion. Celdas leidas del propio articulo por el autor
        # (seccion de datos y de resultados): 10.440 observaciones horarias de validaciones de
        # tarjeta, 1 jul. 2021 - 31 ene. 2023; particion 80/20; seleccion de variables por
        # ganancia de XGBoost; en ese ranking dominan hora, dia escolar, dia de la semana,
        # festivo nacional y mes, y precipitacion y nieve quedan entre las menos importantes.
        "estudio": "Yazıcıoğlu y Akgüngör (2025)",
        "ambito": "Autobús y ferrocarril de la orilla asiática de Estambul, a escala regional y de ruta; validaciones horarias",
        "horizonte": "10.440 observaciones horarias (jul. 2021 - ene. 2023)",
        "exogenas": "Hora, día, mes y día de semana; festivos nacionales y días lectivos; temperatura, precipitación, humedad, viento, nubosidad y nieve",
        "protocolo": "Partición 80/20; selección de variables por ganancia de XGBoost; LSTM, GRU, RNN y NARX con parada temprana",
        "aporte": "NARX con menor error; la meteorología reduce el RMSE solo un 2,2 % regional (autobús) y un 2,4-3,7 % por ruta; el calendario domina el ranking de ganancia",
    },
    {
        "estudio": "Toqué et al. (2017)",
        "ambito": "Transporte multimodal por estación (París)",
        "horizonte": "Logs de billetaje, 2017",
        "exogenas": "No especificada en la fuente disponible",
        "protocolo": "No especificado en la fuente disponible",
        "aporte": "Comparativa de métodos de ML para flujos multimodales, sin arquitectura híbrida residual",
    },
    {
        "estudio": "Cardozo et al. (2012)",
        "ambito": "Entradas por estación del Metro de Madrid, corte transversal espacial",
        "horizonte": "Sin dimensión temporal (regresión espacial)",
        "exogenas": "Entorno construido; sin calendario ni clima",
        "protocolo": "Ajuste espacial in-sample",
        "aporte": "Regresión espacial por estación; no aborda predicción temporal",
    },
    {
        # Fase 2. Celdas segun la ficha verificada por el autor: tarjetas inteligentes de todos
        # los operadores de autobus de Gipuzkoa, fines de semana de 2010 y 2011, regresion
        # lineal multiple. DOI tomado del registro de Crossref (el que circula con prefijo
        # j.trb es erroneo). Como Cardozo, no es un trabajo de prediccion temporal.
        "estudio": "Arana et al. (2014)",
        "ambito": "Todos los operadores de autobús de Gipuzkoa; viajes de tarjeta inteligente en fin de semana",
        "horizonte": "Fines de semana de 2010 y 2011",
        "exogenas": "Viento, lluvia y temperatura; viajeros habituales y ocasionales por separado",
        "protocolo": "Regresión lineal múltiple explicativa; sin partición de pronóstico",
        "aporte": "Viento y lluvia reducen los viajes y la temperatura los aumenta; cuantifica el efecto meteorológico, no aborda predicción",
    },
    {
        # Fase 2. Celdas confirmadas por el autor sobre el resumen y los "highlights" del
        # articulo: metro de Nueva York (MTA NYCT), modelos a escala de sistema y de estacion
        # sobre demanda diaria y horaria de 2010-2011; meteorologia de NOAA y Weather
        # Underground; regresion OLS explicativa, sin particion temporal de validacion. Es el
        # unico trabajo revisado que modela la misma demanda a escala diaria y horaria y
        # compara ambas, lo que respalda la agregacion diaria de este TFM como decision.
        "estudio": "Singhal et al. (2014)",
        "ambito": "Metro de Nueva York (MTA NYCT); modelos a escala de sistema y de estación",
        "horizonte": "Demanda diaria y horaria, 2010-2011 (dos años)",
        "exogenas": "Meteorología (NOAA, Weather Underground); segmentación por día de semana y franja horaria; características de estación (protección, accesibilidad, autobús de conexión)",
        "protocolo": "Regresión OLS explicativa, no predictiva; sin partición temporal de validación",
        "aporte": "Los modelos diario y horario difieren sustancialmente en la variabilidad que explican; el efecto meteorológico varía por franja y localización",
    },
    {
        # Fase 2. Celdas leidas del propio articulo (seccion de datos y tabla 2 de resultados):
        # estacion Xibu (inicio de la linea 2 del metro de Chengdu), flujo entrante cada 15 min
        # de 06:00 a 22:00, 9-29 de abril de 2018 (1.344 observaciones); 14 dias laborables
        # de entrenamiento y un unico viernes de prueba; sin factores externos por decision
        # declarada. MAE de test 24,193 (EMD-LSTM) frente a 28,164 (LSTM), 34,248 (BPN) y
        # 39,451 (ARIMA). Una correccion de 2020 (e0231199) retira un fichero de apoyo
        # incluido por error y no altera resultados.
        "estudio": "Chen et al. (2019)",
        "ambito": "Flujo entrante de una estación del metro de Chengdu (Xibu, línea 2), intervalos de 15 min",
        "horizonte": "Tres semanas de abril de 2018 (1.344 observaciones de 15 min)",
        "exogenas": "Ninguna por decisión declarada: solo el propio historial de flujo (sin meteorología ni calendario)",
        "protocolo": "Holdout cronológico: 14 días laborables de entrenamiento y un único viernes de prueba; sin CV",
        "aporte": "El híbrido EMD+LSTM bate a LSTM, BPN y ARIMA en el día de prueba (MAE 24,2 frente a 28,2, 34,2 y 39,5)",
    },
    {
        "estudio": "Zhang (2003)",
        "ambito": "Series univariantes ajenas al transporte (manchas solares, linces, GBP/USD)",
        "horizonte": "288 / 114 / 731 observaciones",
        "exogenas": "Ninguna (residuo modelado a partir de sus propios rezagos)",
        "protocolo": "Holdout único, residuos ARIMA in-sample (sin OOF)",
        "aporte": "El híbrido ARIMA+red neuronal bate a ambos componentes en las tres series",
    },
    {
        # (b) y (d) tomados de la propia fuente, seccion 3.1 "Experimental Setup" (p. 5):
        # observaciones minutales de radiacion difusa mas temperatura, humedad y presion
        # (CESAR, Paises Bajos), diciembre de 2023; primer 80 % para entrenar y 20 % final
        # para probar, una sola ejecucion, sin validacion cruzada. Redactado en paralelo a
        # las filas de Monje y Zhang para que el lector compare los protocolos de un vistazo.
        "estudio": "Surribas-Sayago et al. (2026)",
        "ambito": "Radiación solar difusa (dominio meteorológico, no transporte)",
        "horizonte": "Diciembre de 2023, resolución minutal (CESAR, Países Bajos)",
        "exogenas": "Retardos y términos de Fourier (fuente de inspiración de la ingeniería de variables de este TFM, no de su arquitectura)",
        "protocolo": "Holdout cronológico único 80/20, una sola ejecución, sin CV",
        "aporte": "Arquitectura paralela CNN+LSTM+MLP; citada solo por su ingeniería de características",
    },
    {
        "estudio": "Este TFM",
        "ambito": "Demanda diaria agregada de los cuatro operadores del CRTM (Madrid)",
        "horizonte": "1.310 días continuos 2023-2026, cero huecos, cero nulos, íntegramente post-pandemia",
        # Fase 3 de la revision de direccion (item 14): la columna (c) es la matriz EXOGENA, y
        # de las 161 predictoras solo 122 lo son (11 calendario + 6 Fourier + 105 meteo); las
        # otras 39 son retardos y estadisticos moviles de la propia serie y de sus operadores.
        "exogenas": "122 exógenas de las 161 predictoras: 105 meteorológicas físicas retardadas (lags 1/2/3/7 + media móvil 7), 6 términos de Fourier y 11 de calendario (55 días puente detectados); meteorología del mismo día explícitamente prohibida",
        "protocolo": "Partición cronológica 70/15/15 de frontera única; walk-forward OOF en 5 bloques expansivos; SARIMAX walk-forward de un paso (un día hacia adelante) sin reestimación; escalado ajustado solo en train; 257 pruebas automatizadas fijando las reglas anti-fuga",
        "aporte": "Auditoría del fallo: el híbrido no bate a SARIMAX (+70.597 MAE test) ni a XGBoost independiente (+78.103); el mejor modelo es un ensamble 50/50 (MAE 139.725, -10,5% vs. el mejor componente)",
    },
]

LITERATURE_COLUMNS = {
    "estudio": "Estudio",
    "ambito": "(a) Ámbito y granularidad",
    "horizonte": "(b) Horizonte temporal",
    "exogenas": "(c) Matriz exógena",
    "protocolo": "(d) Protocolo de validación",
    "aporte": "(e) Aporte empírico",
}


def build_comparativa_estado_arte() -> pd.DataFrame:
    """Literature comparison table for section 2.4: one row per study plus 'Este TFM' last, 5 axes."""
    df = pd.DataFrame(LITERATURE_COMPARISON)
    return df.rename(columns=LITERATURE_COLUMNS)[list(LITERATURE_COLUMNS.values())]


# --------------------------------------------------------------------------------------
# Chapter 3 — data sources, unified schema, feature groups, splits
# --------------------------------------------------------------------------------------


def build_fuentes_datos() -> pd.DataFrame:
    """Static inventory of the three raw sources (CLAUDE.md Data Inventory).

    `Origen` carries the page that actually serves each file. It exists because the
    reproduction sequence of Annex C cannot run without these three files and the
    repository does not redistribute them (`data/` is gitignored and was never versioned),
    so without an address the very first step is unreachable. The column states where the
    data comes from and nothing more: no licence or redistribution claim is made here or
    in the annex, because the terms were not established and asserting them would be worse
    than omitting them.

    Landing pages, not direct download links: the file-level URLs carry a resource id that
    the portals rotate, whereas the dataset page survives it and is one click away from the
    download.
    """
    rows = [
        {
            "Fichero": "CRTM_Evolucion_demanda_diaria.xlsx",
            "Contenido": "Demanda diaria por operador (metro, EMT, carretera, cercanías) y total",
            "Formato": "Excel, hoja 'diaria', cabecera en fila 2",
            "Origen": "https://datos.crtm.es/documents/crtm::crtm-evolucion-demanda-diaria/about",
        },
        {
            "Fichero": "open-meteo-40.39N3.68W666m.csv",
            "Contenido": "22 variables meteorológicas diarias (temperatura, precipitación, viento, presión, radiación)",
            "Formato": "CSV, cabecera real en línea 4",
            "Origen": "https://open-meteo.com/en/docs/historical-weather-api",
        },
        {
            "Fichero": "300082-1-calendario_laboral-csv.csv",
            "Contenido": "Día de la semana, tipo de día (laborable/sábado/domingo/festivo) y festividad",
            "Formato": "CSV separado por ';', UTF-8 con BOM",
            "Origen": "https://datos.madrid.es/dataset/300082-0-calendario_laboral",
        },
    ]
    return pd.DataFrame(rows)


def build_esquema_unificado() -> pd.DataFrame:
    """Static group summary of the 31-column unified schema (data/interim)."""
    rows = [
        {"Grupo": "Fecha", "N.º columnas": 1, "Ejemplo": "date"},
        {"Grupo": "Demanda (target + operadores)", "N.º columnas": 5,
         "Ejemplo": "total, metro, emt, carretera, cercanias"},
        {"Grupo": "Meteorología", "N.º columnas": 21,
         "Ejemplo": "temperature_2m_mean, precipitation_sum, wind_speed_10m_max"},
        {"Grupo": "Calendario", "N.º columnas": 4,
         "Ejemplo": "day_of_week_es, day_type, holiday_type, holiday_name"},
    ]
    return pd.DataFrame(rows)


def build_grupos_features() -> pd.DataFrame:
    """Static group summary of the 161-column model feature matrix (CLAUDE.md Phase 2/3)."""
    rows = [
        {"Grupo": "Retardos de la demanda total", "N.º variables": 7,
         "Descripción": "lags 1, 2, 3, 7, 14, 21, 28"},
        {"Grupo": "Retardos de operadores", "N.º variables": 28,
         "Descripción": "4 operadores x 7 retardos"},
        {"Grupo": "Medias/desv. móviles de la demanda", "N.º variables": 4,
         "Descripción": "media y desv. típica, ventanas 7 y 28, shift-first"},
        {"Grupo": "Términos de Fourier", "N.º variables": 6,
         "Descripción": "semanal k=1, anual k=1 y k=2"},
        {"Grupo": "Calendario", "N.º variables": 11,
         "Descripción": "day_type (5), holiday_type (4), is_weekend, is_bridge_day"},
        {"Grupo": "Meteorología retardada", "N.º variables": 105,
         "Descripción": "21 variables x (lags 1/2/3/7 + media móvil 7)"},
    ]
    df = pd.DataFrame(rows)
    total = df["N.º variables"].sum()
    df.loc[len(df)] = {"Grupo": "Total", "N.º variables": total, "Descripción": ""}
    return df


def build_particiones() -> pd.DataFrame:
    """Chronological split boundaries, read live from `chronological_split`."""
    feats = pd.read_parquet(_require(FEATURES_DAILY_FILE))
    bounds = chronological_split(feats["date"].sort_values())
    total = bounds.n_train + bounds.n_val + bounds.n_test
    rows = [
        {"Partición": "Train", "Inicio": f"{bounds.train_start:%Y-%m-%d}",
         "Fin": f"{bounds.train_end:%Y-%m-%d}", "Días": bounds.n_train,
         "Porcentaje": 100 * bounds.n_train / total},
        {"Partición": "Val", "Inicio": f"{bounds.val_start:%Y-%m-%d}",
         "Fin": f"{bounds.val_end:%Y-%m-%d}", "Días": bounds.n_val,
         "Porcentaje": 100 * bounds.n_val / total},
        {"Partición": "Test", "Inicio": f"{bounds.test_start:%Y-%m-%d}",
         "Fin": f"{bounds.test_end:%Y-%m-%d}", "Días": bounds.n_test,
         "Porcentaje": 100 * bounds.n_test / total},
    ]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Chapter 4 — baselines, LSTM window/OOF, XGBoost hyperparameters
# --------------------------------------------------------------------------------------


def build_sarimax_orden() -> pd.DataFrame:
    """Selected SARIMAX order, read live from the persisted JSON artifact."""
    payload = json.loads(_require(SARIMAX_ORDER_FILE).read_text(encoding="utf-8"))
    p, d, q = payload["order"]
    P, D, Q, s = payload["seasonal_order"]
    return pd.DataFrame(
        [
            {
                "Orden (p,d,q)": f"({p},{d},{q})",
                "Orden estacional (P,D,Q,s)": f"({P},{D},{Q},{s})",
                "AIC": payload["aic"],
                "N.º variables exógenas": len(payload["exog_cols"]),
            }
        ]
    )


def build_ventana_lstm() -> pd.DataFrame:
    """W=14/28/60 comparison, read live from the persisted parquet."""
    df = pd.read_parquet(_require(LSTM_WINDOW_COMPARISON_FILE)).sort_values("window")
    return df.rename(
        columns={
            "window": "W",
            "train_sequences": "Secuencias train",
            "epochs_run": "Épocas ejecutadas",
            "best_epoch": "Mejor época",
            "val_MAE": "MAE (val)",
            "val_RMSE": "RMSE (val)",
            "val_MAPE": "MAPE (val, %)",
            "val_R2": "R² (val)",
        }
    ).reset_index(drop=True)


def build_folds_oof() -> pd.DataFrame:
    """Per-fold OOF epochs/losses/metrics, read live from the persisted parquet."""
    df = pd.read_parquet(_require(LSTM_OOF_FOLD_LOG_FILE)).sort_values("fold")
    return df.rename(
        columns={
            "fold": "Fold",
            "epochs_run": "Épocas ejecutadas",
            "best_epoch": "Mejor época",
            "final_train_loss": "Loss train (final)",
            "final_val_loss": "Loss val (final)",
            "mae": "MAE",
            "rmse": "RMSE",
        }
    ).reset_index(drop=True)


def build_hiperparametros() -> pd.DataFrame:
    return report_tables.hyperparameters_table()


# --------------------------------------------------------------------------------------
# Chapter 5 — master tables, breakdowns, importances, delta, sensitivity
# --------------------------------------------------------------------------------------


def build_comparacion(split: str) -> pd.DataFrame:
    return report_tables.comparison_table(split)


def build_desglose_dia(split: str) -> pd.DataFrame:
    return report_tables.day_type_table(split)


def build_importancia_grupo(model: str) -> pd.DataFrame:
    return report_tables.group_importance_table(model)


def build_top15(model: str) -> pd.DataFrame:
    return report_tables.feature_importance_table(model)


def build_delta_hibrido() -> pd.DataFrame:
    """hybrid - sarimax and hybrid - xgboost_alone, on val and test.

    Generated (not typed by hand) so this table can never disagree with `comparison_table`
    — the exact failure mode a hand-copied number is prone to.
    """
    rows = []
    for split in ("val", "test"):
        table = model_comparison_table(split).set_index("model")
        hybrid_mae = table.loc["hybrid", "MAE"]
        for competitor in ("sarimax", "xgboost_alone"):
            delta = hybrid_mae - table.loc[competitor, "MAE"]
            rows.append(
                {
                    "Partición": split,
                    "Comparación": f"hybrid - {competitor}",
                    # ASCII column name: 'Delta MAE', not the Unicode Greek letter --
                    # pdflatex's default OT1/T1 fonts have no glyph for U+0394 and a
                    # LaTeX macro (\Delta) would be double-escaped by escape_latex(),
                    # which treats every header/cell as plain text (see its docstring).
                    "Delta MAE": delta,
                    "Veredicto": "hybrid peor" if delta > 0 else "hybrid mejor",
                }
            )
    return pd.DataFrame(rows)


def build_criterio_experimento_a() -> pd.DataFrame:
    """Pre-registered verdict for Phase 5b Experiment A, computed via sensitivity_report."""
    from src.evaluation.sensitivity_report import (
        PHASE5_HYBRID_TEST_MAE,
        assemble_all,
        build_breakdown,
        build_master_tables,
        evaluate_criterion_a,
    )

    df = assemble_all()
    breakdown = build_breakdown(df)
    master = build_master_tables(df)
    verdict = evaluate_criterion_a(breakdown, master["test"])

    rows = [
        {"Criterio": "MAE en días irregulares (festivo + puente), n-ponderado",
         "hybrid": verdict["irregular_mae_hybrid"],
         "hybrid_weighted": verdict["irregular_mae_weighted"],
         "Resultado": "Mejora" if verdict["irregular_improved"] else "No mejora"},
        {"Criterio": f"MAE global test (techo +{5}% sobre {PHASE5_HYBRID_TEST_MAE:,.0f})".replace(",", "."),
         "hybrid": PHASE5_HYBRID_TEST_MAE,
         "hybrid_weighted": verdict["overall_test_mae"],
         "Resultado": "Dentro del margen" if verdict["within_budget"] else "Fuera del margen"},
        {"Criterio": "Veredicto del criterio prerregistrado (ambas condiciones)",
         "hybrid": float("nan"), "hybrid_weighted": float("nan"),
         "Resultado": "CUMPLIDO" if verdict["criterion_met"] else "NO CUMPLIDO"},
    ]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# 5.9 (fase diagnóstica 7) — anatomía del residuo de la etapa 1 LSTM
#
# Every number below comes from src/evaluation/stage1_residual_anatomy.py, which retrains
# nothing. Non-ASCII Greek (rho, sigma, Delta) is spelled out: pdflatex's T1 fonts have no
# glyph for it and it aborts the build, the same incident documented for U+0394 in
# REPORT_MAPPING.md §5.
# --------------------------------------------------------------------------------------

_ANATOMY_LAGS = [1, 2, 3, 7, 14, 21, 28]


def build_autocorrelacion_residuo() -> pd.DataFrame:
    """ACF/PACF of the Stage 1 residual at the weekly lags, with Ljung-Box columns."""
    from src.evaluation.stage1_residual_anatomy import ljung_box, residual_autocorrelation

    acf_table = residual_autocorrelation().set_index("lag")
    lb = ljung_box(lags=(7, 14, 28)).set_index("lag")

    rows = []
    for lag in _ANATOMY_LAGS:
        rows.append(
            {
                "Retardo (días)": lag,
                "ACF": acf_table.loc[lag, "acf"],
                "PACF": acf_table.loc[lag, "pacf"],
                "Fuera de banda": bool(acf_table.loc[lag, "outside_band"]),
                "Ljung-Box Q": lb.loc[lag, "lb_stat"] if lag in lb.index else float("nan"),
                "p-valor (LB)": lb.loc[lag, "lb_pvalue"] if lag in lb.index else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def build_residuo_dia_semana() -> pd.DataFrame:
    """Weekday profile of the Stage 1 residual (mean, median, spread, effect size)."""
    from src.evaluation.memoria_figures import WEEKDAY_LABELS
    from src.evaluation.stage1_residual_anatomy import weekday_residual_profile

    df = weekday_residual_profile()
    return pd.DataFrame(
        {
            "Día": [WEEKDAY_LABELS[d] for d in df["day_of_week_es"]],
            "n": df["n"].to_numpy(),
            "Residuo medio": df["mean_residual"].to_numpy(),
            "Mediana": df["median_residual"].to_numpy(),
            "Desv. típica": df["sd"].to_numpy(),
            "|Residuo| medio": df["mean_abs_residual"].to_numpy(),
            "Media / sigma": df["mean_over_sigma"].to_numpy(),
        }
    )


def build_residuo_calendario() -> pd.DataFrame:
    """Stage 1 residual magnitude by calendar regime (day type + holiday type)."""
    from src.evaluation.stage1_residual_anatomy import residual_by_calendar_group

    df = residual_by_calendar_group()
    return df.rename(
        columns={
            "dimension": "Dimensión",
            "group": "Grupo",
            "n": "n",
            "mean_abs_residual": "|Residuo| medio",
            "mean_abs_pct": "Error abs. medio (%)",
        }
    )


def build_residuo_meteo(top_n: int = 10) -> pd.DataFrame:
    """Top-N lagged-weather features by |partial correlation| with the Stage 1 residual."""
    from src.evaluation.stage1_residual_anatomy import residual_weather_correlation

    df = residual_weather_correlation().head(top_n)
    return pd.DataFrame(
        {
            "Variable meteorológica": df["weather_var"].to_numpy(),
            "Retardo": df["lag_kind"].to_numpy(),
            "Rho bruto": df["spearman_raw"].to_numpy(),
            "Rho parcial": df["spearman_partial"].to_numpy(),
            "p parcial": df["p_partial"].to_numpy(),
            "p ajustado (BH)": df["p_adj_bh"].to_numpy(),
            "Significativa": df["significant"].to_numpy(),
        }
    )


def build_cadena_error() -> pd.DataFrame:
    """The lstm_alone -> hybrid -> xgboost_alone error chain, one block per split.

    Generated, not typed: `delta` and `gap_share` are derived from the same
    `metrics.all_metrics` path as chapter 5, so this table cannot contradict the master
    comparison. NOT a decomposition — three separate models (REPORT_MAPPING.md O6).
    """
    from src.evaluation.stage1_residual_anatomy import error_chain_table

    frames = []
    for split in ("test", "val"):
        block = error_chain_table(split).copy()
        block.insert(0, "Partición", split)
        frames.append(block)
    df = pd.concat(frames, ignore_index=True)
    return df.rename(
        columns={
            "Modelo": "Modelo",
            "MAPE": "MAPE (%)",
            "R2": "R²",
            "delta_MAE_abs": "Delta MAE",
            "delta_MAE_pct": "Delta MAE (%)",
            "gap_share_pct": "Cuota del hueco (%)",
        }
    )


LEARNING_CURVE_FILE: Path = PROCESSED_DIR / "lstm_learning_curve.parquet"


def build_curva_aprendizaje() -> pd.DataFrame:
    """Median (min-max) val MAE of the Stage 1 LSTM per training-size fraction.

    Source: `data/processed/lstm_learning_curve.parquet` (3 seeds per fraction; 5 at
    fraction 1.00 -- the point Phase 8's parity control reads as its baseline; both built
    by one run of `lstm_learning_curve.main`; val only, trailing training window). The
    `Fits degenerados` column carries the count of seeds
    whose fit collapsed to a near-constant prediction, so the reader can tell an
    undertrained model from one that failed to train at all.
    """
    curve = pd.read_parquet(_require(LEARNING_CURVE_FILE))
    grouped = curve.groupby("fraction")
    return pd.DataFrame(
        {
            "Fracción": [f"{f:.2f}" for f in grouped.groups],
            "Filas train": grouped["train_rows"].first().to_numpy(),
            "Secuencias": grouped["train_sequences"].first().to_numpy(),
            "MAE val (mediana)": grouped["val_MAE"].median().to_numpy(),
            "MAE val (mín)": grouped["val_MAE"].min().to_numpy(),
            "MAE val (máx)": grouped["val_MAE"].max().to_numpy(),
            "R² val (mediana)": grouped["val_R2"].median().to_numpy(),
            "Fits degenerados": grouped["degenerate"].sum().to_numpy(),
        }
    )


def build_ensemble_pesos() -> pd.DataFrame:
    """Component weights for both ensemble variants, computed the same way as Phase 5b."""
    ensemble_df = pd.read_parquet(_require(ENSEMBLE_PREDICTIONS_FILE))
    val_df = ensemble_df[ensemble_df["split"] == "val"]

    w_equal = equal_weights()
    w_inverse = inverse_mae_weights(val_df)

    rows = []
    for variant, weights in (("ensemble_equal", w_equal), ("ensemble_inverse_mae", w_inverse)):
        rows.append(
            {
                "Variante": variant,
                "Peso SARIMAX": weights["sarimax"],
                "Peso XGBoost independiente": weights["xgboost_alone"],
            }
        )
    return pd.DataFrame(rows)


BLOCK_ABLATION_FILE: Path = PROCESSED_DIR / "feature_block_ablation.parquet"
PARITY_CONTROL_FILE: Path = PROCESSED_DIR / "lstm_parity_control.parquet"

_ABLATION_ROW_ORDER = ["completo", "solo_temporal", "solo_exogeno", "sin_meteo", "solo_calendario"]


def build_ablacion_bloques() -> pd.DataFrame:
    """Feature-block ablation of `xgboost_alone` -- diagnostic control (Phase 8, Q2).

    Source: `data/processed/feature_block_ablation.parquet`. Each subset re-runs the same
    54-point grid and internal holdout as Phase 5; the `completo` row is read from the
    published master comparison, not retrained. Block-removal study, NOT a decomposition
    (objection O2): the deltas do not sum to anything.
    """
    table = pd.read_parquet(_require(BLOCK_ABLATION_FILE))
    val = table[table["split"] == "val"].set_index("subset")
    test = table[table["split"] == "test"].set_index("subset")

    rows = []
    for subset in _ABLATION_ROW_ORDER:
        v = val.loc[subset]
        rows.append(
            {
                "Subconjunto": subset.replace("_", " "),
                "N variables": int(v["n_features"]),
                "prof.": int(v["max_depth"]),
                "n_est.": int(v["n_estimators"]),
                "lr": float(v["learning_rate"]),
                "MAE val": float(v["MAE"]),
                "MAE test": float(test.loc[subset, "MAE"]),
                "Delta MAE val (%)": float(v["delta_val_MAE_pct"]),
            }
        )
    return pd.DataFrame(rows)


def _subgroup_label(key: str) -> str:
    from src.evaluation.memoria_figures import SUBGROUP_LABELS

    return SUBGROUP_LABELS[key]


def build_subgrupos(split: str) -> pd.DataFrame:
    """Per-subgroup: best competitor + MAE, best hybrid + MAE, Δ, pre-registered verdict.

    Source: `subgroup_breakdown.py` (retrains nothing). The `Veredicto` column is the SAME
    pre-registered verdict on both the test and val tables -- there is one verdict per
    subgroup, read on the verdict scope (test) with val as the replication requirement; the
    two tables differ only in the MAE/Δ columns. Non-ASCII Greek (Δ) is spelled 'Delta',
    the U+0394 incident documented in REPORT_MAPPING.md §5.
    """
    from src.evaluation import subgroup_breakdown as sb

    win = sb.hybrid_win_table(split).set_index("subgroup")
    crit = sb.evaluate_subgroup_criteria().set_index("subgroup")

    rows = []
    for key in sb.SUBGROUPS:
        if key not in win.index:
            continue
        w = win.loc[key]
        rows.append(
            {
                "Subgrupo": _subgroup_label(key),
                "n": int(w["n"]),
                "Mejor competidor": w["best_model"],
                "MAE competidor": float(w["best_MAE"]),
                "Mejor híbrido": w["best_hybrid"],
                "MAE híbrido": float(w["best_hybrid_MAE"]),
                "Delta MAE": float(w["delta_MAE"]),
                "Veredicto": crit.loc[key, "verdict"],
            }
        )
    return pd.DataFrame(rows)


def build_subgrupos_inferencia() -> pd.DataFrame:
    """Moving-block bootstrap ΔMAE with 95% CI, raw and BH-adjusted p, for the inferential
    family on the verdict scope (test) only.

    A suppressed CI (usable-resample fraction below the threshold) prints as '--'. The
    verdict column repeats the pre-registered reading so the table is self-contained.
    """
    from src.evaluation import subgroup_breakdown as sb

    boot = sb.bootstrap_delta("test").set_index("subgroup")
    crit = sb.evaluate_subgroup_criteria().set_index("subgroup")

    rows = []
    for key in sb.SUBGROUPS:
        if key not in boot.index:
            continue
        b = boot.loc[key]
        rows.append(
            {
                "Subgrupo": _subgroup_label(key),
                "n": int(b["n"]),
                "Delta MAE": float(b["delta_MAE"]),
                "IC 95% inf.": float(b["ci_low"]),
                "IC 95% sup.": float(b["ci_high"]),
                "p": float(b["p_raw"]),
                "p (BH)": float(b["p_adj_bh"]),
                "Veredicto": crit.loc[key, "verdict"],
            }
        )
    return pd.DataFrame(rows)


def build_paridad_lstm() -> pd.DataFrame:
    """Informational-parity LSTM per scope -- diagnostic control (Phase 8, Q1).

    Source: `data/processed/lstm_parity_control.parquet` (5 seeds per scope, val only).
    The baseline for the gap-closure column is the 5-seed median univariate Stage 1 LSTM
    val MAE (416.324); the best single seed (411.142) is quoted in the section 5.10.2
    prose. Per objection O4 a partial gap-closure is weak evidence.
    """
    parity = pd.read_parquet(_require(PARITY_CONTROL_FILE))
    grouped = parity.groupby("scope", sort=False)
    rows = []
    for scope, block in grouped:
        rows.append(
            {
                "Ámbito": scope,
                "k": int(block["n_exog"].iloc[0]),
                "MAE val (mediana)": float(block["val_MAE"].median()),
                "MAE val (mín)": float(block["val_MAE"].min()),
                "MAE val (máx)": float(block["val_MAE"].max()),
                "Hueco cerrado (%)": float(block["gap_closed_pct"].median()),
                "Fits degenerados": int(block["degenerate"].sum()),
                "best epoch (mediana)": float(block["best_epoch"].median()),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------


def export_all() -> list[Path]:
    """Build and write every table. Returns the paths written, in order."""
    written: list[Path] = []

    def emit(df: pd.DataFrame, filename: str, **kwargs) -> None:
        written.append(write_table(df, filename, **kwargs))

    # 1.4 / 2.4 — direction-mandated additions
    emit(
        build_cronograma_fases(),
        "tabla_cronograma_fases.tex",
        column_spec=(
            r"p{1.7cm}"
            r">{\raggedright\arraybackslash}p{2.7cm}"
            r">{\raggedright\arraybackslash}p{4.3cm}"
            r">{\raggedright\arraybackslash}p{4.3cm}"
            r"r"
        ),
    )
    emit(
        build_comparativa_estado_arte(),
        "tabla_comparativa_estado_arte.tex",
        column_spec="p{2.6cm}p{3.6cm}p{3.0cm}p{4.2cm}p{4.4cm}p{4.6cm}",
    )

    # Chapter 3
    emit(
        build_fuentes_datos(),
        "tabla_fuentes_datos.tex",
        # Four columns now, same 14.5cm declared total as the three-column version: the
        # \resizebox at the \input site scales the declared width, so keeping the total
        # fixed keeps the effective font size where it was. Origen gets 3.6cm because the
        # widest unbreakable run in the three addresses is a host name ('datos.crtm.es' --
        # \url breaks after '/' and '-' but not after '.'), and a p{} narrower than its
        # widest token overflows silently past the margin (incidencia 12b).
        column_spec=(
            r">{\raggedright\arraybackslash}p{3.5cm}"
            r">{\raggedright\arraybackslash}p{4.6cm}"
            r">{\raggedright\arraybackslash}p{2.8cm}"
            r">{\raggedright\arraybackslash}p{3.6cm}"
        ),
        url_columns={"Origen"},
    )
    emit(
        build_esquema_unificado(),
        "tabla_esquema_unificado.tex",
        column_spec=(
            r">{\raggedright\arraybackslash}p{3.0cm}"
            r">{\centering\arraybackslash}p{2.0cm}"
            r">{\raggedright\arraybackslash}p{9.0cm}"
        ),
    )
    emit(
        build_grupos_features(),
        "tabla_grupos_features.tex",
        column_spec=(
            r">{\raggedright\arraybackslash}p{4.3cm}"
            r">{\centering\arraybackslash}p{2.0cm}"
            r">{\raggedright\arraybackslash}p{7.8cm}"
        ),
    )
    emit(build_particiones(), "tabla_particiones.tex", decimals={"Porcentaje": 1})

    # Chapter 4
    emit(build_sarimax_orden(), "tabla_sarimax_orden.tex", decimals={"AIC": 2})
    emit(
        build_ventana_lstm(),
        "tabla_ventana_lstm.tex",
        decimals={"MAE (val)": 0, "RMSE (val)": 0, "MAPE (val, %)": 2, "R² (val)": 4},
        column_spec=(
            r">{\centering\arraybackslash}p{0.8cm}"
            r">{\centering\arraybackslash}p{2.1cm}"
            r">{\centering\arraybackslash}p{2.0cm}"
            r">{\centering\arraybackslash}p{1.7cm}"
            r">{\centering\arraybackslash}p{1.9cm}"
            r">{\centering\arraybackslash}p{1.9cm}"
            r">{\centering\arraybackslash}p{2.0cm}"
            r">{\centering\arraybackslash}p{1.5cm}"
        ),
    )
    emit(
        build_folds_oof(),
        "tabla_folds_oof.tex",
        decimals={"Loss train (final)": 4, "Loss val (final)": 4, "MAE": 0, "RMSE": 0},
        column_spec=(
            r">{\centering\arraybackslash}p{1.0cm}"
            r">{\centering\arraybackslash}p{2.1cm}"
            r">{\centering\arraybackslash}p{1.9cm}"
            r">{\centering\arraybackslash}p{2.2cm}"
            r">{\centering\arraybackslash}p{2.1cm}"
            r">{\centering\arraybackslash}p{1.9cm}"
            r">{\centering\arraybackslash}p{1.9cm}"
        ),
    )
    emit(
        build_hiperparametros(),
        "tabla_hiperparametros.tex",
        decimals={"learning_rate": 2},
        column_spec=(
            r">{\raggedright\arraybackslash}p{3.0cm}"
            r">{\centering\arraybackslash}p{2.3cm}"
            r">{\centering\arraybackslash}p{1.8cm}"
            r">{\centering\arraybackslash}p{2.1cm}"
            r">{\centering\arraybackslash}p{2.0cm}"
            r">{\centering\arraybackslash}p{2.3cm}"
        ),
    )

    # Chapter 5
    comparacion_spec = (
        r">{\raggedright\arraybackslash}p{3.2cm}"
        r">{\raggedright\arraybackslash}p{2.4cm}"
        r">{\centering\arraybackslash}p{0.8cm}"
        r">{\centering\arraybackslash}p{1.7cm}"
        r">{\centering\arraybackslash}p{1.7cm}"
        r">{\centering\arraybackslash}p{1.7cm}"
        r">{\centering\arraybackslash}p{1.4cm}"
    )
    # 5 fixed model columns (see DEFAULT_BREAKDOWN_MODELS): sarimax, xgboost_alone, hybrid,
    # hybrid_weighted, ensemble_equal. Widths sized so the two longest headers
    # ('xgboost_alone', 'hybrid_weighted') wrap on their escaped underscore instead of
    # overflowing an unbreakable 'r' column.
    desglose_spec = (
        r">{\raggedright\arraybackslash}p{2.6cm}"
        r">{\centering\arraybackslash}p{0.7cm}"
        r">{\centering\arraybackslash}p{1.7cm}"
        r">{\centering\arraybackslash}p{1.9cm}"
        r">{\centering\arraybackslash}p{1.6cm}"
        r">{\centering\arraybackslash}p{1.9cm}"
        r">{\centering\arraybackslash}p{1.9cm}"
    )
    for split in ("test", "val"):
        emit(
            build_comparacion(split),
            f"tabla_comparacion_{split}.tex",
            decimals={"MAPE (%)": 2, "R²": 4},
            column_spec=comparacion_spec,
        )
        emit(
            build_desglose_dia(split),
            f"tabla_desglose_dia_{split}.tex",
            column_spec=desglose_spec,
        )

    emit(build_importancia_grupo("xgboost_alone"), "tabla_importancia_grupo_alone.tex",
         decimals={"Ganancia": 4, "Peso (%)": 2})
    emit(build_importancia_grupo("xgboost_residual"), "tabla_importancia_grupo_residual.tex",
         decimals={"Ganancia": 4, "Peso (%)": 2})
    emit(build_top15("xgboost_alone"), "tabla_top15_alone.tex", decimals={"Ganancia": 4})
    emit(build_top15("xgboost_residual"), "tabla_top15_residual.tex", decimals={"Ganancia": 4})

    emit(build_delta_hibrido(), "tabla_delta_hibrido.tex", decimals={"Delta MAE": 0})
    emit(
        build_criterio_experimento_a(),
        "tabla_criterio_experimento_a.tex",
        decimals={"hybrid": 0, "hybrid_weighted": 0},
        column_spec=(
            r">{\raggedright\arraybackslash}p{6.5cm}"
            r">{\centering\arraybackslash}p{1.9cm}"
            r">{\centering\arraybackslash}p{2.4cm}"
            r">{\centering\arraybackslash}p{3.0cm}"
        ),
    )
    emit(build_ensemble_pesos(), "tabla_ensemble_pesos.tex",
         decimals={"Peso SARIMAX": 4, "Peso XGBoost independiente": 4})

    # 5.9 — Stage 1 residual anatomy (Phase 7 diagnostic)
    emit(
        build_autocorrelacion_residuo(),
        "tabla_autocorrelacion_residuo.tex",
        decimals={"ACF": 3, "PACF": 3, "Ljung-Box Q": 2, "p-valor (LB)": 4},
        column_spec=(
            r">{\centering\arraybackslash}p{2.2cm}"
            r">{\centering\arraybackslash}p{1.8cm}"
            r">{\centering\arraybackslash}p{1.8cm}"
            r">{\centering\arraybackslash}p{2.2cm}"
            r">{\centering\arraybackslash}p{2.2cm}"
            r">{\centering\arraybackslash}p{2.2cm}"
        ),
    )
    emit(
        build_residuo_dia_semana(),
        "tabla_residuo_dia_semana.tex",
        decimals={
            "Residuo medio": 0, "Mediana": 0, "Desv. típica": 0,
            "|Residuo| medio": 0, "Media / sigma": 3,
        },
        column_spec=(
            r">{\raggedright\arraybackslash}p{1.9cm}"
            r">{\centering\arraybackslash}p{0.8cm}"
            r">{\centering\arraybackslash}p{2.2cm}"
            r">{\centering\arraybackslash}p{2.2cm}"
            r">{\centering\arraybackslash}p{2.2cm}"
            r">{\centering\arraybackslash}p{2.2cm}"
            r">{\centering\arraybackslash}p{1.8cm}"
        ),
    )
    emit(
        build_residuo_calendario(),
        "tabla_residuo_calendario.tex",
        decimals={"|Residuo| medio": 0, "Error abs. medio (%)": 2},
        column_spec=(
            r">{\raggedright\arraybackslash}p{2.4cm}"
            r">{\raggedright\arraybackslash}p{5.2cm}"
            r">{\centering\arraybackslash}p{0.9cm}"
            r">{\centering\arraybackslash}p{2.6cm}"
            r">{\centering\arraybackslash}p{2.6cm}"
        ),
    )
    emit(
        build_residuo_meteo(),
        "tabla_residuo_meteo.tex",
        decimals={
            "Rho bruto": 3, "Rho parcial": 3, "p parcial": 4, "p ajustado (BH)": 4,
        },
        column_spec=(
            r">{\raggedright\arraybackslash}p{3.6cm}"
            r">{\centering\arraybackslash}p{1.8cm}"
            r">{\centering\arraybackslash}p{1.6cm}"
            r">{\centering\arraybackslash}p{1.6cm}"
            r">{\centering\arraybackslash}p{1.7cm}"
            r">{\centering\arraybackslash}p{1.9cm}"
            # "Significativa" is a single unbreakable word ~2.2cm wide at \small; a
            # narrower p{} column lets it hang outside the cell, and \resizebox scales the
            # DECLARED width, so the overhanging glyphs still cross the margin.
            r">{\centering\arraybackslash}p{2.4cm}"
        ),
    )
    emit(
        build_curva_aprendizaje(),
        "tabla_curva_aprendizaje.tex",
        decimals={
            "MAE val (mediana)": 0, "MAE val (mín)": 0, "MAE val (máx)": 0,
            "R² val (mediana)": 4,
        },
        column_spec=(
            r">{\centering\arraybackslash}p{1.6cm}"
            r">{\centering\arraybackslash}p{1.6cm}"
            r">{\centering\arraybackslash}p{1.7cm}"
            r">{\centering\arraybackslash}p{2.4cm}"
            r">{\centering\arraybackslash}p{2.1cm}"
            r">{\centering\arraybackslash}p{2.1cm}"
            r">{\centering\arraybackslash}p{2.2cm}"
            r">{\centering\arraybackslash}p{2.0cm}"
        ),
    )
    emit(
        build_cadena_error(),
        "tabla_cadena_error.tex",
        decimals={
            "MAE": 0, "RMSE": 0, "MAPE (%)": 2, "R²": 4,
            "Delta MAE": 0, "Delta MAE (%)": 1, "Cuota del hueco (%)": 1,
        },
        column_spec=(
            r">{\centering\arraybackslash}p{1.4cm}"
            r">{\raggedright\arraybackslash}p{2.4cm}"
            r">{\centering\arraybackslash}p{1.6cm}"
            r">{\centering\arraybackslash}p{1.6cm}"
            r">{\centering\arraybackslash}p{1.3cm}"
            r">{\centering\arraybackslash}p{1.3cm}"
            r">{\centering\arraybackslash}p{1.6cm}"
            r">{\centering\arraybackslash}p{1.5cm}"
            r">{\centering\arraybackslash}p{1.7cm}"
        ),
    )

    # 5.10 — Phase 8 diagnostic: informational parity and block ablation
    emit(
        build_ablacion_bloques(),
        "tabla_ablacion_bloques.tex",
        decimals={
            "lr": 2, "MAE val": 0, "MAE test": 0, "Delta MAE val (%)": 1,
        },
        column_spec=(
            r">{\raggedright\arraybackslash}p{2.3cm}"
            r">{\centering\arraybackslash}p{1.4cm}"
            r">{\centering\arraybackslash}p{1.0cm}"
            r">{\centering\arraybackslash}p{1.0cm}"
            r">{\centering\arraybackslash}p{1.0cm}"
            r">{\centering\arraybackslash}p{1.8cm}"
            r">{\centering\arraybackslash}p{1.8cm}"
            r">{\centering\arraybackslash}p{2.0cm}"
        ),
    )
    emit(
        build_paridad_lstm(),
        "tabla_paridad_lstm.tex",
        decimals={
            "MAE val (mediana)": 0, "MAE val (mín)": 0, "MAE val (máx)": 0,
            "Hueco cerrado (%)": 1, "best epoch (mediana)": 1,
        },
        column_spec=(
            r">{\raggedright\arraybackslash}p{1.8cm}"
            r">{\centering\arraybackslash}p{0.8cm}"
            r">{\centering\arraybackslash}p{2.0cm}"
            r">{\centering\arraybackslash}p{1.8cm}"
            r">{\centering\arraybackslash}p{1.8cm}"
            r">{\centering\arraybackslash}p{1.9cm}"
            r">{\centering\arraybackslash}p{1.7cm}"
            r">{\centering\arraybackslash}p{2.0cm}"
        ),
    )

    # 5.11 — Phase 9 diagnostic: subgroup and period breakdown
    subgrupos_spec = (
        r">{\raggedright\arraybackslash}p{2.7cm}"
        r">{\centering\arraybackslash}p{0.7cm}"
        r">{\raggedright\arraybackslash}p{2.3cm}"
        r">{\centering\arraybackslash}p{1.8cm}"
        r">{\raggedright\arraybackslash}p{2.1cm}"
        r">{\centering\arraybackslash}p{1.8cm}"
        r">{\centering\arraybackslash}p{1.8cm}"
        r">{\raggedright\arraybackslash}p{2.9cm}"
    )
    for split in ("test", "val"):
        emit(
            build_subgrupos(split),
            f"tabla_subgrupos_{split}.tex",
            decimals={"MAE competidor": 0, "MAE híbrido": 0, "Delta MAE": 0},
            column_spec=subgrupos_spec,
        )
    emit(
        build_subgrupos_inferencia(),
        "tabla_subgrupos_inferencia.tex",
        decimals={
            "Delta MAE": 0, "IC 95% inf.": 0, "IC 95% sup.": 0, "p": 3, "p (BH)": 3,
        },
        column_spec=(
            r">{\raggedright\arraybackslash}p{2.9cm}"
            r">{\centering\arraybackslash}p{0.7cm}"
            r">{\centering\arraybackslash}p{1.9cm}"
            r">{\centering\arraybackslash}p{1.9cm}"
            r">{\centering\arraybackslash}p{1.9cm}"
            r">{\centering\arraybackslash}p{1.3cm}"
            r">{\centering\arraybackslash}p{1.3cm}"
            r">{\raggedright\arraybackslash}p{2.9cm}"
        ),
    )

    return written


def main() -> None:
    print("=" * 78)
    print("EXPORTING LATEX TABLES — no model is trained or re-predicted")
    print("=" * 78)
    written = export_all()
    for path in written:
        print(f"  exported: {path.name}")
    print("-" * 78)
    print(f"{len(written)} tables written to {TABLES_DIR}")


if __name__ == "__main__":
    main()
