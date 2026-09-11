"""Additional Plotly figure builders needed by the memoria's chapters 3 and 4.

The Phase 6/6b dashboard only ever needed to visualise *predictions* (Phases 3-5b), so
`dashboard_figures.py` has nothing about the raw series, the EDA, the chronological splits
or the OOF walk-forward scheme. Those live here instead of being bolted onto
`dashboard_figures.py`, because their inputs are different (features/OOF artifacts, not
prediction tables) and mixing the two would blur the "no model logic in a figure module"
boundary that already exists.

SAME RULES AS dashboard_figures.py:
  - Pure functions returning `go.Figure`. No I/O side effects beyond reading a parquet.
  - Every builder returns `apply_theme(fig)` — colours and fonts come from theme.py only.
  - NOTHING here imports from `src.models`: these figures describe data and diagnostics
    already computed in Phases 2-5b, not models being re-fit. A test pins this.
"""

import math

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.evaluation import stage1_residual_anatomy as sra
from src.evaluation import subgroup_breakdown as sb
from src.evaluation.dashboard_data import _require, model_comparison_table
from src.evaluation.theme import VIU_COLORS, VIU_COLORWAY, apply_theme
from src.features.build_features import FEATURES_DAILY_FILE
from src.utils.paths import PROCESSED_DIR
from src.utils.splits import chronological_split

OOF_FILE = PROCESSED_DIR / "lstm_oof_predictions.parquet"
OOF_FOLD_LOG_FILE = PROCESSED_DIR / "lstm_oof_fold_log.parquet"
ENSEMBLE_FILE = PROCESSED_DIR / "ensemble_predictions.parquet"
LEARNING_CURVE_FILE = PROCESSED_DIR / "lstm_learning_curve.parquet"
BLOCK_ABLATION_FILE = PROCESSED_DIR / "feature_block_ablation.parquet"
PARITY_CONTROL_FILE = PROCESSED_DIR / "lstm_parity_control.parquet"

# Degeneracy thresholds are owned by lstm_learning_curve.py; re-stated here (not imported)
# so this figure module never transitively pulls in TensorFlow via src.models.
_DEGENERATE_STD_RATIO_MAX = 0.50
_DEGENERATE_CORR_MAX = 0.50

OPERATOR_COLS = ["metro", "emt", "carretera", "cercanias"]
OPERATOR_LABELS = {
    "metro": "Metro de Madrid",
    "emt": "EMT",
    "carretera": "Concesionarias por carretera",
    "cercanias": "Renfe Cercanías",
}

# lunes -> domingo, the natural reading order for a weekly-seasonality chart. The raw
# column is already unaccented ASCII (CLAUDE.md, Phase 1); accents are added only for the
# axis labels, never for matching logic.
WEEKDAY_ORDER = ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"]
WEEKDAY_LABELS = {
    "lunes": "Lunes",
    "martes": "Martes",
    "miercoles": "Miércoles",
    "jueves": "Jueves",
    "viernes": "Viernes",
    "sabado": "Sábado",
    "domingo": "Domingo",
}

MONTH_LABELS = [
    "Ene", "Feb", "Mar", "Abr", "May", "Jun",
    "Jul", "Ago", "Sep", "Oct", "Nov", "Dic",
]

DAY_TYPE_LABELS = {
    "laborable": "Laborable",
    "sabado": "Sábado",
    "domingo": "Domingo",
    "festivo": "Festivo",
    "domingo festivo": "Domingo festivo",
}


def _load_features() -> pd.DataFrame:
    return pd.read_parquet(_require(FEATURES_DAILY_FILE))


def _load_oof() -> pd.DataFrame:
    return pd.read_parquet(_require(OOF_FILE))


def _load_oof_fold_log() -> pd.DataFrame:
    return pd.read_parquet(_require(OOF_FOLD_LOG_FILE))


def _load_ensemble() -> pd.DataFrame:
    return pd.read_parquet(_require(ENSEMBLE_FILE))


# --------------------------------------------------------------------------------------
# 3.3 — Full series and per-operator breakdown
# --------------------------------------------------------------------------------------


def total_series_figure(features: pd.DataFrame | None = None) -> go.Figure:
    """Daily total demand, full 2023-2026 history — establishes scale and weekly texture."""
    df = features if features is not None else _load_features()

    fig = go.Figure(
        go.Scatter(
            x=df["date"],
            y=df["total"],
            mode="lines",
            line=dict(color=VIU_COLORS["near_black"], width=1),
            name="Total",
        )
    )
    fig.update_layout(
        xaxis_title="Fecha",
        yaxis_title="Viajes/día",
        showlegend=False,
    )
    return apply_theme(fig)


def operator_demand_figure(features: pd.DataFrame | None = None) -> go.Figure:
    """Daily demand broken down by the four CRTM operator columns."""
    df = features if features is not None else _load_features()

    fig = go.Figure()
    for col, color in zip(OPERATOR_COLS, VIU_COLORWAY):
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df[col],
                mode="lines",
                name=OPERATOR_LABELS[col],
                line=dict(color=color, width=1),
            )
        )
    fig.update_layout(
        xaxis_title="Fecha",
        yaxis_title="Viajes/día",
        legend=dict(orientation="h", y=1.02, yanchor="bottom"),
    )
    return apply_theme(fig)


# --------------------------------------------------------------------------------------
# 3.4 — EDA: seasonality and weather
# --------------------------------------------------------------------------------------


def weekly_seasonality_figure(features: pd.DataFrame | None = None) -> go.Figure:
    """Box plot of total demand by day of week — the dominant seasonal pattern."""
    df = features if features is not None else _load_features()

    fig = go.Figure()
    for day, color in zip(WEEKDAY_ORDER, VIU_COLORWAY):
        subset = df.loc[df["day_of_week_es"] == day, "total"]
        fig.add_trace(
            go.Box(
                y=subset,
                name=WEEKDAY_LABELS[day],
                marker_color=color,
                boxmean=True,
            )
        )
    fig.update_layout(
        yaxis_title="Viajes/día",
        showlegend=False,
    )
    return apply_theme(fig)


def annual_seasonality_figure(features: pd.DataFrame | None = None) -> go.Figure:
    """Mean total demand by calendar month, averaged across all covered years."""
    df = features if features is not None else _load_features()

    monthly = (
        df.assign(month=df["date"].dt.month)
        .groupby("month")["total"]
        .agg(["mean", "std"])
        .reindex(range(1, 13))
    )

    fig = go.Figure(
        go.Bar(
            x=MONTH_LABELS,
            y=monthly["mean"],
            error_y=dict(type="data", array=monthly["std"], visible=True),
            marker_color=VIU_COLORS["primary"],
        )
    )
    fig.update_layout(
        xaxis_title="Mes",
        yaxis_title="Viajes/día (media ± desv. típica)",
        showlegend=False,
    )
    return apply_theme(fig)


def weather_vs_demand_figure(features: pd.DataFrame | None = None) -> go.Figure:
    """Scatter of same-day mean temperature vs. total demand, coloured by day type.

    Same-day weather is EDA-only here (CLAUDE.md Phase 3: it never enters the model
    matrix unlagged) — this figure motivates *why* weather is a candidate exogenous
    signal, not a claim about what the model actually sees.
    """
    df = features if features is not None else _load_features()

    day_types = [d for d in DAY_TYPE_LABELS if d in set(df["day_type"].astype(str))]
    fig = go.Figure()
    for day_type, color in zip(day_types, VIU_COLORWAY):
        subset = df[df["day_type"].astype(str) == day_type]
        fig.add_trace(
            go.Scatter(
                x=subset["temperature_2m_mean"],
                y=subset["total"],
                mode="markers",
                name=DAY_TYPE_LABELS[day_type],
                marker=dict(color=color, size=6, opacity=0.65),
            )
        )
    fig.update_layout(
        xaxis_title="Temperatura media diaria (°C)",
        yaxis_title="Viajes/día",
        legend=dict(orientation="h", y=1.02, yanchor="bottom"),
    )
    return apply_theme(fig)


# --------------------------------------------------------------------------------------
# 3.6 — Chronological splits
# --------------------------------------------------------------------------------------


def splits_figure(features: pd.DataFrame | None = None) -> go.Figure:
    """Full series with the train/val/test chronological blocks shaded — CLAUDE.md Phase 3."""
    df = features if features is not None else _load_features()
    bounds = chronological_split(df["date"].sort_values())

    fig = go.Figure(
        go.Scatter(
            x=df["date"],
            y=df["total"],
            mode="lines",
            line=dict(color=VIU_COLORS["near_black"], width=1),
            name="Total",
            showlegend=False,
        )
    )

    regions = [
        ("Train", bounds.train_start, bounds.train_end, VIU_COLORS["gray_light"]),
        ("Val", bounds.val_start, bounds.val_end, VIU_COLORS["steel"]),
        ("Test", bounds.test_start, bounds.test_end, VIU_COLORS["primary"]),
    ]
    for label, start, end, color in regions:
        # kaleido's JSON serializer chokes on a raw pandas Timestamp; vrect boundaries
        # must be plain ISO strings, unlike the trace x-values above (which plotly.js
        # handles natively as a datetime axis).
        fig.add_vrect(
            x0=start.isoformat(),
            x1=end.isoformat(),
            fillcolor=color,
            opacity=0.18,
            line_width=0,
            annotation_text=label,
            annotation_position="top left",
        )

    fig.update_layout(
        xaxis_title="Fecha",
        yaxis_title="Viajes/día",
    )
    return apply_theme(fig)


# --------------------------------------------------------------------------------------
# 4.4 — LSTM out-of-fold walk-forward scheme
# --------------------------------------------------------------------------------------


def oof_folds_figure(oof: pd.DataFrame | None = None) -> go.Figure:
    """Actual demand vs. OOF prediction, coloured by walk-forward fold (1-5)."""
    df = oof if oof is not None else _load_oof()
    covered = df[df["has_oof"]].sort_values("date")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=covered["date"],
            y=covered["y_true"],
            mode="lines",
            name="Real",
            line=dict(color=VIU_COLORS["near_black"], width=1.5),
        )
    )
    for fold, color in zip(sorted(covered["fold"].unique()), VIU_COLORWAY):
        subset = covered[covered["fold"] == fold]
        fig.add_trace(
            go.Scatter(
                x=subset["date"],
                y=subset["y_pred_oof"],
                mode="lines",
                name=f"OOF fold {fold}",
                line=dict(color=color, width=1.5, dash="dot"),
            )
        )
    fig.update_layout(
        xaxis_title="Fecha",
        yaxis_title="Viajes/día",
        legend=dict(orientation="h", y=1.02, yanchor="bottom"),
    )
    return apply_theme(fig)


def oof_fold1_degenerate_figure(
    oof: pd.DataFrame | None = None, fold_log: pd.DataFrame | None = None
) -> go.Figure:
    """Side-by-side: fold 1's near-constant OOF prediction vs. a healthy fold (fold 3).

    Fold 1 (CLAUDE.md Phase 4): OOF std ratio 0.199x, correlation 0.300 with actuals — the
    model never escapes predicting close to a constant. Fold 3 is the best-correlated of
    the remaining folds and makes the contrast legible at a glance.
    """
    df = oof if oof is not None else _load_oof()
    covered = df[df["has_oof"]].sort_values("date")

    fold1 = covered[covered["fold"] == 1]
    fold3 = covered[covered["fold"] == 3]

    std_ratio = fold1["y_pred_oof"].std() / fold1["y_true"].std()
    corr = fold1[["y_true", "y_pred_oof"]].corr().iloc[0, 1]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"Fold 1 (degenerado) — ratio std {std_ratio:.3f}, corr {corr:.3f}",
            "Fold 3 (sano, referencia)",
        ),
    )
    for col, subset in ((1, fold1), (2, fold3)):
        fig.add_trace(
            go.Scatter(
                x=subset["date"], y=subset["y_true"], mode="lines",
                name="Real", line=dict(color=VIU_COLORS["near_black"], width=1.5),
                showlegend=(col == 1),
            ),
            row=1, col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=subset["date"], y=subset["y_pred_oof"], mode="lines",
                name="OOF", line=dict(color=VIU_COLORS["primary"], width=1.5, dash="dot"),
                showlegend=(col == 1),
            ),
            row=1, col=col,
        )

    fig.update_layout(
        legend=dict(orientation="h", y=1.12, yanchor="bottom"),
        margin=dict(t=100),
        height=480,
    )
    fig.update_yaxes(title_text="Viajes/día", row=1, col=1)
    return apply_theme(fig)


# --------------------------------------------------------------------------------------
# 5.9 — Why the ensemble wins: error decorrelation
# --------------------------------------------------------------------------------------


def error_dispersion_figure(ensemble: pd.DataFrame | None = None, split: str = "test") -> go.Figure:
    """Scatter of SARIMAX vs. XGBoost-alone residuals — the mechanical evidence for why
    averaging the two beats either alone (Granger 1989's decorrelated-information-sets
    condition, cited in CLAUDE.md Phase 5b).
    """
    df = ensemble if ensemble is not None else _load_ensemble()
    subset = df[df["split"] == split].copy()
    subset["resid_sarimax"] = subset["y_true"] - subset["sarimax"]
    subset["resid_xgboost_alone"] = subset["y_true"] - subset["xgboost_alone"]

    corr = subset[["resid_sarimax", "resid_xgboost_alone"]].corr().iloc[0, 1]

    fig = go.Figure(
        go.Scatter(
            x=subset["resid_sarimax"],
            y=subset["resid_xgboost_alone"],
            mode="markers",
            marker=dict(color=VIU_COLORS["primary"], size=7, opacity=0.7),
            showlegend=False,
        )
    )
    fig.add_hline(y=0, line=dict(color=VIU_COLORS["gray_light"], width=1))
    fig.add_vline(x=0, line=dict(color=VIU_COLORS["gray_light"], width=1))
    fig.add_annotation(
        text=f"corr = {corr:.3f}",
        xref="paper", yref="paper", x=0.02, y=0.98, showarrow=False,
        font=dict(size=13, color=VIU_COLORS["near_black"]),
    )
    fig.update_layout(
        xaxis_title="Residuo SARIMAX (real − predicho)",
        yaxis_title="Residuo XGBoost independiente (real − predicho)",
    )
    return apply_theme(fig)


# --------------------------------------------------------------------------------------
# 5.9 (Fase diagnóstica) — Anatomía del residuo de la etapa 1 LSTM
#
# These read the tables computed in stage1_residual_anatomy.py (never a model). They plot
# the PRIMARY series (final model, 393 contiguous val+test days) unless noted.
# --------------------------------------------------------------------------------------

_STRUCTURE_LAGS = [7, 14, 21, 28]


def stage1_acf_pacf_figure(series: str = sra.PRIMARY_SERIES) -> go.Figure:
    """ACF and PACF of the Stage 1 residual to lag 35, with the Bartlett band shaded.

    A univariate LSTM that never sees the calendar is expected to leave period-7 residual
    structure; this quantifies how much. Weekly lags (7/14/21/28) are marked.
    """
    table = sra.residual_autocorrelation(series)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
        subplot_titles=("ACF", "PACF"),
    )
    for row, col_name in ((1, "acf"), (2, "pacf")):
        fig.add_trace(
            go.Scatter(
                x=list(table["lag"]) + list(table["lag"][::-1]),
                y=list(table["ci_high"]) + list(table["ci_low"][::-1]),
                fill="toself",
                fillcolor=VIU_COLORS["gray_pale"],
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=row, col=1,
        )
        fig.add_trace(
            go.Bar(
                x=table["lag"], y=table[col_name],
                marker_color=VIU_COLORS["steel"], showlegend=False,
            ),
            row=row, col=1,
        )
        for lag in _STRUCTURE_LAGS:
            fig.add_vline(
                x=lag, line=dict(color=VIU_COLORS["gray_light"], width=1, dash="dot"),
                row=row, col=1,
            )
    fig.update_yaxes(title_text="Autocorrelación", row=1, col=1)
    fig.update_yaxes(title_text="Autocorrelación parcial", row=2, col=1)
    fig.update_xaxes(title_text="Retardo (días)", row=2, col=1)
    fig.update_layout(height=560, bargap=0.5)
    return apply_theme(fig)


def stage1_periodogram_figure(series: str = sra.PRIMARY_SERIES) -> go.Figure:
    """Periodogram of the Stage 1 residual — power vs. period in days, log axis.

    The period-7 bin is annotated. Spectral leakage spreads weekly power across adjacent
    bins and into the 7/3-day harmonic, so the single-bin share understates the weekly
    structure that the weekday profile and ACF(7) also show.

    LOG-AXIS COORDINATES. Plotly maps a *shape* placed by `add_vline` onto a log axis for
    you, but not the annotation it creates alongside it: that annotation's `x` is read as
    an already-log10 value, so `annotation_text=` on `add_vline(x=7.0)` puts the label at
    10^7 days and drags autorange out to ~4e7, compressing the whole 2-393 day band into a
    sliver. Hence the label is added separately in log10 coordinates, and the range is
    pinned to the band the periodogram can actually represent. The zero-frequency bin is
    already dropped upstream in `periodogram_frame`; it was never the cause.
    """
    table = sra.residual_periodogram(series)

    fig = go.Figure(
        go.Scatter(
            x=table["period_days"], y=table["power_share"],
            mode="markers",
            marker=dict(color=VIU_COLORS["steel"], size=6, opacity=0.75),
            showlegend=False,
        )
    )
    fig.add_vline(
        x=7.0, line=dict(color=VIU_COLORS["primary"], width=1.5, dash="dash"),
    )
    fig.add_annotation(
        x=math.log10(7.0), y=1.0, yref="y domain",
        text="periodo 7 días", showarrow=False,
        xanchor="left", yanchor="bottom", xshift=4,
        font=dict(color=VIU_COLORS["primary"]),
    )
    # Derived from the table, not hardcoded: the axis cannot disagree with the data. The
    # band runs from the Nyquist bin (~2 days) to the fundamental (N days, 393 on the
    # primary series) — NOT to N/2, which would clip the period-196.5 bin that is the
    # second-highest peak of the whole periodogram.
    pad = 0.03
    fig.update_xaxes(
        title_text="Periodo (días, escala log)", type="log",
        range=[
            math.log10(float(table["period_days"].min())) - pad,
            math.log10(float(table["period_days"].max())) + pad,
        ],
    )
    fig.update_yaxes(title_text="Cuota de potencia espectral")
    fig.update_layout(height=460)
    return apply_theme(fig)


def stage1_weekday_figure(series: str = sra.PRIMARY_SERIES) -> go.Figure:
    """Mean Stage 1 residual by day of week, with ±1 s.e. bars and n under each tick.

    A positive bar means the LSTM under-predicts that weekday on average. The spread of
    these means as a fraction of the residual sigma is the effect size that carries the
    weekly-seasonality claim (the significance tests are anti-conservative here).
    """
    table = sra.weekday_residual_profile(series)
    se = table["sd"] / table["n"] ** 0.5

    fig = go.Figure(
        go.Bar(
            x=[WEEKDAY_LABELS[d] for d in table["day_of_week_es"]],
            y=table["mean_residual"],
            error_y=dict(type="data", array=se, visible=True),
            marker_color=VIU_COLORS["steel"],
            showlegend=False,
        )
    )
    fig.add_hline(y=0, line=dict(color=VIU_COLORS["near_black"], width=1))
    fig.update_layout(
        xaxis=dict(
            tickmode="array",
            tickvals=[WEEKDAY_LABELS[d] for d in table["day_of_week_es"]],
            ticktext=[
                f"{WEEKDAY_LABELS[d]}<br><i>n = {int(n)}</i>"
                for d, n in zip(table["day_of_week_es"], table["n"])
            ],
        ),
        yaxis_title="Residuo medio (real − predicho)",
        height=460,
    )
    return apply_theme(fig)


def stage1_weather_corr_figure(series: str = sra.PRIMARY_SERIES, top_n: int = 15) -> go.Figure:
    """Top-N lagged-weather features by |partial ρ| with the Stage 1 residual.

    Partial ρ controls for the four annual Fourier terms; the raw ρ is shown as a ghost
    marker so the confounding gap (both weather and demand carry the annual cycle) is
    visible. BH-significant features at q=0.10 are highlighted.
    """
    table = sra.residual_weather_correlation(series).head(top_n).iloc[::-1]

    colors = [
        VIU_COLORS["primary"] if sig else VIU_COLORS["steel"]
        for sig in table["significant"]
    ]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=table["spearman_partial"], y=table["feature"], orientation="h",
            marker_color=colors, name="ρ parcial",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=table["spearman_raw"], y=table["feature"], mode="markers",
            marker=dict(color=VIU_COLORS["gray"], size=9, symbol="circle-open"),
            name="ρ bruto",
        )
    )
    fig.add_vline(x=0, line=dict(color=VIU_COLORS["near_black"], width=1))
    fig.update_layout(
        xaxis_title="Correlación de Spearman con el residuo",
        legend=dict(orientation="h", y=1.03, yanchor="bottom"),
        height=560,
        margin=dict(l=280),
    )
    return apply_theme(fig)


def stage1_error_chain_figure(split: str = "test") -> go.Figure:
    """Stepped horizontal bars: lstm_alone → hybrid → xgboost_alone MAE on `split`.

    Deliberately NOT a waterfall with connector risers: these are three separate models,
    not additive contributions to one error. The step between consecutive bars is
    annotated in absolute MAE and as a share of the full lstm_alone → xgboost_alone gap.

    Each step label sits in the whitespace BETWEEN the two bars it compares, at the
    half-integer position of the categorical axis. Anchoring it on one bar and nudging it
    by a pixel offset both collided with the neighbouring bar and put the label on the
    wrong side — a step describes the pair it separates, not the bar it happens to touch.
    """
    table = sra.error_chain_table(split)
    order = list(table["Modelo"])[::-1]  # best at top
    plotted = table.set_index("Modelo").loc[order].reset_index()
    # Categorical axis positions: index in `plotted` is the y coordinate of each bar.
    position = {m: i for i, m in enumerate(plotted["Modelo"])}

    fig = go.Figure(
        go.Bar(
            x=plotted["MAE"], y=plotted["Modelo"], orientation="h",
            marker_color=[
                VIU_COLORS["steel"] if m == "xgboost_alone" else VIU_COLORS["secondary"]
                for m in plotted["Modelo"]
            ],
            text=[f"{v:,.0f}" for v in plotted["MAE"]],
            textposition="outside", cliponaxis=False, showlegend=False,
        )
    )
    previous = None
    for _, row in table.iterrows():
        if row["delta_MAE_abs"] != 0 and previous is not None:
            fig.add_annotation(
                x=row["MAE"],
                y=(position[previous] + position[row["Modelo"]]) / 2.0,
                text=f"Δ {row['delta_MAE_abs']:,.0f}  "
                     f"({row['gap_share_pct']:.0f}% del hueco)",
                showarrow=False, xanchor="right", yanchor="middle", xshift=-4,
                font=dict(size=11, color=VIU_COLORS["primary"]),
                bgcolor="rgba(255,255,255,0.85)",
            )
        previous = row["Modelo"]
    fig.update_layout(
        xaxis_title="MAE (viajes/día)",
        height=460,
        bargap=0.45,
        margin=dict(l=140),
    )
    return apply_theme(fig)


def stage1_fold_size_figure() -> go.Figure:
    """Per-fold OOF error vs. training rows: raw MAE and difficulty-controlled skill ratio.

    Five points, non-monotonic, confounded with which period each fold predicts — this is
    indicative context, not a learning curve. The skill ratio (fold MAE / seasonal_naive
    on the identical block) flattens the confound: fold 1 is marked degenerate.
    """
    table = sra.fold_size_vs_error()
    degenerate = table["fold"] == sra.DEGENERATE_OOF_FOLD
    marker_symbols = ["x" if d else "circle" for d in degenerate]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("MAE OOF por fold", "Ratio de habilidad (MAE / seasonal_naive)"),
        horizontal_spacing=0.13,
    )
    fig.add_trace(
        go.Scatter(
            x=table["train_rows"], y=table["mae"], mode="markers+lines+text",
            marker=dict(color=VIU_COLORS["steel"], size=12, symbol=marker_symbols),
            line=dict(color=VIU_COLORS["gray_light"], width=1),
            text=[f"fold {f}" for f in table["fold"]], textposition="top center",
            showlegend=False,
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=table["train_rows"], y=table["skill_ratio"], mode="markers+lines+text",
            marker=dict(color=VIU_COLORS["primary"], size=12, symbol=marker_symbols),
            line=dict(color=VIU_COLORS["gray_light"], width=1),
            text=[f"fold {f}" for f in table["fold"]], textposition="top center",
            showlegend=False,
        ),
        row=1, col=2,
    )
    fig.add_hline(
        y=1.0, line=dict(color=VIU_COLORS["near_black"], width=1, dash="dot"), row=1, col=2,
    )
    fig.update_xaxes(title_text="Filas de entrenamiento del fold", row=1, col=1)
    fig.update_xaxes(title_text="Filas de entrenamiento del fold", row=1, col=2)
    fig.update_yaxes(title_text="MAE (viajes/día)", row=1, col=1)
    fig.update_yaxes(title_text="Ratio (menor = mejor que naive)", row=1, col=2)
    fig.update_layout(height=460)
    return apply_theme(fig)


def stage1_learning_curve_figure() -> go.Figure:
    """Val MAE of the Stage 1 LSTM vs. training rows: median line, min-max band, 3 seeds.

    Two horizontal references are drawn — SARIMAX and xgboost_alone val MAE — because the
    supervisor's question is specifically why the LSTM path loses to a direct XGBoost, so
    the reader needs to see whether the trend is even heading toward that line. Fractions
    where every seed produced a DEGENERATE fit (near-constant prediction, not merely an
    undertrained one) are drawn with a hollow marker: a collapsed model and an
    undertrained one give different MAEs for different reasons.
    """
    curve = pd.read_parquet(_require(LEARNING_CURVE_FILE))
    grouped = curve.groupby("fraction")
    summary = pd.DataFrame(
        {
            "train_rows": grouped["train_rows"].first(),
            "median": grouped["val_MAE"].median(),
            "lo": grouped["val_MAE"].min(),
            "hi": grouped["val_MAE"].max(),
            "n_degenerate": grouped["degenerate"].sum(),
            "n_seeds": grouped["seed"].size(),
        }
    ).sort_values("train_rows")
    all_degenerate = summary["n_degenerate"] == summary["n_seeds"]

    val_table = model_comparison_table("val").set_index("model")["MAE"]
    sarimax_ref = float(val_table["sarimax"])
    xgb_ref = float(val_table["xgboost_alone"])

    fig = go.Figure()
    # Min-max band across the 3 seeds.
    fig.add_trace(
        go.Scatter(
            x=list(summary["train_rows"]) + list(summary["train_rows"][::-1]),
            y=list(summary["hi"]) + list(summary["lo"][::-1]),
            fill="toself", fillcolor=VIU_COLORS["gray_pale"], line=dict(width=0),
            hoverinfo="skip", showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=summary["train_rows"], y=summary["median"], mode="lines+markers",
            line=dict(color=VIU_COLORS["secondary"], width=2),
            marker=dict(
                color=VIU_COLORS["secondary"], size=13,
                symbol=["circle-open" if d else "circle" for d in all_degenerate],
                line=dict(width=2, color=VIU_COLORS["secondary"]),
            ),
            name="LSTM etapa 1 (mediana de 3 semillas)",
        )
    )
    for ref, label, color in (
        (sarimax_ref, "SARIMAX (val)", VIU_COLORS["steel"]),
        (xgb_ref, "xgboost_alone (val)", VIU_COLORS["primary"]),
    ):
        fig.add_hline(
            y=ref, line=dict(color=color, width=1.5, dash="dash"),
            annotation_text=f"{label}: {ref:,.0f}", annotation_position="right",
        )
    fig.add_annotation(
        text="marcador hueco = todas las semillas degeneradas",
        xref="paper", yref="paper", x=0.02, y=0.06, xanchor="left", showarrow=False,
        font=dict(size=11, color=VIU_COLORS["gray"]),
    )
    fig.update_layout(
        xaxis_title="Filas de entrenamiento (ventana final, ancla en train_end)",
        yaxis_title="MAE de validación (viajes/día)",
        legend=dict(orientation="h", y=1.03, yanchor="bottom"),
        height=480,
        margin=dict(r=180),
    )
    return apply_theme(fig)


# --------------------------------------------------------------------------------------
# Phase 8 — informational parity and block ablation (diagnostic controls)
# --------------------------------------------------------------------------------------

_ABLATION_LABELS = {
    "completo": "completo (161)",
    "solo_temporal": "solo temporal (39)",
    "solo_exogeno": "solo exógeno (122)",
    "sin_meteo": "sin meteo (56)",
    "solo_calendario": "solo calendario (17)",
}


def block_ablation_figure(split: str = "val") -> go.Figure:
    """Val MAE of `xgboost_alone` retrained on restricted feature subsets.

    Diagnostic control (Phase 8, Q2), NOT a candidate model. Horizontal bars, best subset
    at the top, with a dashed reference line at the `completo` MAE and muted test-split
    markers as a stability check. This is a block-removal study, NOT a decomposition: the
    blocks are not information-disjoint (objection O2), so the deltas never sum to anything.
    """
    table = pd.read_parquet(_require(BLOCK_ABLATION_FILE))
    val = table[table["split"] == "val"].set_index("subset")
    test = table[table["split"] == "test"].set_index("subset")
    order = val["MAE"].sort_values(ascending=False).index.tolist()  # worst first -> top-down bar

    completo_ref = float(val.loc["completo", "MAE"])
    labels = [_ABLATION_LABELS.get(s, s) for s in order]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=labels, x=[float(val.loc[s, "MAE"]) for s in order],
            orientation="h",
            marker=dict(color=VIU_COLORS["secondary"]),
            text=[f"{val.loc[s, 'MAE']:,.0f}" for s in order],
            textposition="outside",
            name="MAE validación",
        )
    )
    fig.add_trace(
        go.Scatter(
            y=labels, x=[float(test.loc[s, "MAE"]) for s in order],
            mode="markers", marker=dict(color=VIU_COLORS["gray"], size=10, symbol="diamond"),
            name="MAE test (estabilidad)",
        )
    )
    fig.add_vline(
        x=completo_ref,
        line=dict(color=VIU_COLORS["primary"], width=1.5, dash="dash"),
        annotation_text=f"completo (val): {completo_ref:,.0f}",
        annotation_position="top",
    )
    fig.update_layout(
        xaxis_title="MAE (viajes/día)",
        yaxis_title="",
        legend=dict(orientation="h", y=1.08, yanchor="bottom"),
        height=460,
        margin=dict(l=160, r=120),
    )
    return apply_theme(fig)


def parity_control_figure() -> go.Figure:
    """Val MAE of the informational-parity LSTM per scope: median + min-max band, 5 seeds.

    Diagnostic control (Phase 8, Q1), NOT a candidate model. Three horizontal references:
    the univariate Stage 1 LSTM baseline (5-seed median) and the `xgboost_alone` / SARIMAX
    val MAE. A scope whose every seed is degenerate is drawn with a hollow marker. Per
    objection O4 a partial gap-closure is weak evidence -- the parity LSTM sees strictly
    more information than XGBoost -- so only a clear failure to close the gap is quotable.
    """
    parity = pd.read_parquet(_require(PARITY_CONTROL_FILE))
    grouped = parity.groupby("scope")
    summary = pd.DataFrame(
        {
            "median": grouped["val_MAE"].median(),
            "lo": grouped["val_MAE"].min(),
            "hi": grouped["val_MAE"].max(),
            "n_exog": grouped["n_exog"].first(),
            "n_degenerate": grouped["degenerate"].sum(),
            "n_seeds": grouped["seed"].size(),
        }
    ).sort_values("n_exog")
    all_degenerate = summary["n_degenerate"] == summary["n_seeds"]
    labels = [f"{s} (k={int(summary.loc[s, 'n_exog'])})" for s in summary.index]

    univ = pd.read_parquet(_require(LEARNING_CURVE_FILE))
    univ_ref = float(univ[univ["fraction"] == 1.0]["val_MAE"].median())
    val_table = model_comparison_table("val").set_index("model")["MAE"]
    xgb_ref = float(val_table["xgboost_alone"])
    sarimax_ref = float(val_table["sarimax"])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=labels, y=summary["median"],
            error_y=dict(
                type="data", symmetric=False,
                array=summary["hi"] - summary["median"],
                arrayminus=summary["median"] - summary["lo"],
            ),
            mode="markers",
            marker=dict(
                color=VIU_COLORS["secondary"], size=16,
                symbol=["circle-open" if d else "circle" for d in all_degenerate],
                line=dict(width=2, color=VIU_COLORS["secondary"]),
            ),
            name="LSTM paridad (mediana de 5 semillas)",
        )
    )
    for ref, label, color in (
        (univ_ref, "LSTM univariante (val, mediana 5 semillas)", VIU_COLORS["near_black"]),
        (sarimax_ref, "SARIMAX (val)", VIU_COLORS["steel"]),
        (xgb_ref, "xgboost_alone (val)", VIU_COLORS["primary"]),
    ):
        fig.add_hline(
            y=ref, line=dict(color=color, width=1.5, dash="dash"),
            annotation_text=f"{label}: {ref:,.0f}", annotation_position="right",
        )
    fig.update_layout(
        xaxis_title="",
        yaxis_title="MAE de validación (viajes/día)",
        legend=dict(orientation="h", y=1.05, yanchor="bottom"),
        height=480,
        margin=dict(r=280),
    )
    return apply_theme(fig)


# --------------------------------------------------------------------------------------
# Phase 9 — subgroup and period breakdown (§5.11). Diagnostic re-analysis: these read the
# tables computed in subgroup_breakdown.py (never a model).
# --------------------------------------------------------------------------------------

# Reduced model set for the subgroup charts, matching dashboard_figures.DEFAULT_BREAKDOWN
# _MODELS so §5.11 shows the same competitors as §5.4.
SUBGROUP_FIGURE_MODELS = [
    "sarimax", "xgboost_alone", "hybrid", "hybrid_weighted", "ensemble_equal",
]

SUBGROUP_LABELS = {
    "laborable_ordinario": "Laborable ordinario",
    "laborable_puente": "Laborable puente",
    "sabado": "Sábado",
    "domingo": "Domingo",
    "festivo": "Festivo",
    "fin_de_semana": "Fin de semana",
    "entre_semana": "Entre semana",
    "calendario_irregular": "Calendario irregular",
    "calendario_ordinario": "Calendario ordinario",
    "verano_jul_ago": "Verano (jul-ago)",
    "resto_del_anio": "Resto del año",
    "periodo_navidad": "Periodo Navidad",
    "periodo_semana_santa": "Periodo Semana Santa",
    "demanda_anomala": "Demanda anómala",
    "demanda_regular": "Demanda regular",
}


def subgroup_mae_figure(scope: str = "test") -> go.Figure:
    """Grouped horizontal bars: MAE by subgroup for a reduced model set.

    Non-inferential subgroups (n < MIN_N_INFERENTIAL) are drawn with a hatched, translucent
    bar so a small-n MAE is never read with the same weight as a 133-day group. The verdict
    lives in the LaTeX table, not here.
    """
    metrics = sb.subgroup_metrics(scope)
    win = sb.hybrid_win_table(scope).set_index("subgroup")
    order = [s for s in sb.SUBGROUPS if s in set(metrics["subgroup"])]
    inferential = {s: bool(win.loc[s, "inferential"]) for s in order}

    fig = go.Figure()
    for model, color in zip(SUBGROUP_FIGURE_MODELS, VIU_COLORWAY):
        sub = metrics[metrics["model"] == model].set_index("subgroup")
        fig.add_trace(
            go.Bar(
                y=[SUBGROUP_LABELS[s] for s in order],
                x=[float(sub.loc[s, "MAE"]) if s in sub.index else None for s in order],
                orientation="h",
                name=model,
                marker=dict(
                    color=color,
                    opacity=[1.0 if inferential[s] else 0.4 for s in order],
                    pattern=dict(
                        shape=["" if inferential[s] else "/" for s in order]
                    ),
                ),
            )
        )
    fig.update_layout(
        barmode="group",
        xaxis_title="MAE (viajes/día)",
        yaxis=dict(autorange="reversed"),
        legend=dict(orientation="h", y=1.02, yanchor="bottom"),
        height=760,
        margin=dict(l=190),
    )
    return apply_theme(fig)


def subgroup_delta_ci_figure(scope: str = "test") -> go.Figure:
    """Forest plot of the hybrid ΔMAE (best hybrid − best competitor) per subgroup.

    Positive Δ = the hybrid is worse. Inferential subgroups carry a moving-block bootstrap
    95% CI; non-inferential ones are drawn as hollow markers with no interval (a point Δ
    only). The zero line is the reference: a CI entirely to its right is significant
    evidence that the hybrid loses in that subgroup.
    """
    boot = sb.bootstrap_delta(scope).set_index("subgroup")
    win = sb.hybrid_win_table(scope).set_index("subgroup")
    order = [
        s for s in sb.SUBGROUPS
        if s not in sb.EXCLUDED_SUBGROUPS and s in win.index
    ][::-1]

    inf_y, inf_x, err_plus, err_minus = [], [], [], []
    non_y, non_x = [], []
    for s in order:
        delta = float(win.loc[s, "delta_MAE"])
        if s in boot.index and bool(win.loc[s, "inferential"]) and pd.notna(
            boot.loc[s, "ci_low"]
        ):
            inf_y.append(SUBGROUP_LABELS[s])
            inf_x.append(delta)
            err_plus.append(float(boot.loc[s, "ci_high"]) - delta)
            err_minus.append(delta - float(boot.loc[s, "ci_low"]))
        else:
            non_y.append(SUBGROUP_LABELS[s])
            non_x.append(delta)

    fig = go.Figure()
    if inf_x:
        fig.add_trace(
            go.Scatter(
                x=inf_x, y=inf_y, mode="markers",
                error_x=dict(
                    type="data", symmetric=False, array=err_plus, arrayminus=err_minus
                ),
                marker=dict(color=VIU_COLORS["primary"], size=10),
                name="inferencial (IC bootstrap 95%)",
            )
        )
    if non_x:
        fig.add_trace(
            go.Scatter(
                x=non_x, y=non_y, mode="markers",
                marker=dict(
                    color=VIU_COLORS["gray"], size=10, symbol="circle-open",
                    line=dict(width=2, color=VIU_COLORS["gray"]),
                ),
                name="no inferencial (sin IC)",
            )
        )
    fig.add_vline(x=0, line=dict(color=VIU_COLORS["near_black"], width=1))
    fig.update_layout(
        xaxis_title="Δ MAE = MAE(mejor híbrido) − MAE(mejor competidor)  (viajes/día)",
        legend=dict(orientation="h", y=1.03, yanchor="bottom"),
        height=620,
        margin=dict(l=190),
    )
    return apply_theme(fig)


def period_rolling_mae_figure() -> go.Figure:
    """Two panels over the continuous val+test span: (a) 28-day rolling MAE per model;
    (b) rolling Δ = min(hybrids) − min(competitors) with a zero line.

    Descriptive only — consecutive 28-day windows overlap by 27 days, so no test is run on
    this figure (objection O7). The val/test boundary is drawn as a vertical rule; windows
    straddling it mix the two splits.
    """
    roll = sb.rolling_mae()
    bounds = chronological_split(_load_features()["date"].sort_values())
    boundary = bounds.test_start.isoformat()

    wide = roll.pivot(index="date", columns="model", values="MAE")
    hyb = wide[[m for m in sb.HYBRID_MODELS if m in wide.columns]].min(axis=1)
    comp = wide[[m for m in sb.COMPETITOR_MODELS if m in wide.columns]].min(axis=1)
    delta = hyb - comp

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09,
        subplot_titles=("MAE móvil (ventana 28 días)", "Δ móvil (mejor híbrido − mejor competidor)"),
    )
    for model, color in zip(SUBGROUP_FIGURE_MODELS, VIU_COLORWAY):
        if model not in wide.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=wide.index, y=wide[model], mode="lines", name=model,
                line=dict(color=color, width=1.5),
            ),
            row=1, col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=delta.index, y=delta, mode="lines", showlegend=False,
            line=dict(color=VIU_COLORS["primary"], width=1.5),
        ),
        row=2, col=1,
    )
    fig.add_hline(y=0, line=dict(color=VIU_COLORS["near_black"], width=1), row=2, col=1)
    for row in (1, 2):
        fig.add_vline(
            x=boundary, line=dict(color=VIU_COLORS["gray"], width=1.5, dash="dot"),
            row=row, col=1,
        )
    fig.add_annotation(
        x=boundary, y=1.0, yref="paper", yanchor="bottom", showarrow=False,
        text="val | test", font=dict(size=11, color=VIU_COLORS["gray"]),
    )
    fig.update_yaxes(title_text="MAE (viajes/día)", row=1, col=1)
    fig.update_yaxes(title_text="Δ MAE (viajes/día)", row=2, col=1)
    fig.update_xaxes(title_text="Fecha", row=2, col=1)
    fig.update_layout(
        legend=dict(orientation="h", y=1.06, yanchor="bottom"), height=640,
    )
    return apply_theme(fig)
