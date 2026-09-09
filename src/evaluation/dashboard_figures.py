"""Plotly figure builders for the Phase 6 dashboard.

Pure functions: each takes plain data (or fetches it via dashboard_data) and returns a
`go.Figure`. No I/O, no display side effects, no `write_image` -- rendering is the
notebook's job and exporting is export_figures.py's job. Keeping the builders here rather
than in notebook cells follows the project convention that `notebooks/` contains no
business logic, and it means the same figure definition backs both the interactive view and
the static PNGs in the memoria, so the two cannot drift.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.evaluation.dashboard_data import (
    MODEL_FAMILY,
    actual_vs_predicted,
    error_by_day_type,
    feature_importance_by_group,
    model_comparison_table,
    residual_distribution,
)

# Colours and template come from theme.py exclusively -- nothing here defines a colour.
from src.evaluation.theme import (
    FAMILY_COLOR,
    GROUP_COLOR,
    VIU_COLORS,
    VIU_TEMPLATE,
    apply_theme,
)

METRICS = ["MAE", "RMSE", "MAPE", "R2"]

DAY_TYPE_ORDER = [
    "laborable, ordinary",
    "laborable",
    "sabado",
    "domingo",
    "festivo",
    "laborable, bridge day",
]

DEFAULT_BREAKDOWN_MODELS = [
    "sarimax",
    "xgboost_alone",
    "hybrid",
    "hybrid_weighted",
    "ensemble_equal",
]



def demand_figure(
    model: str, split: str, predictions: pd.DataFrame | None = None
) -> go.Figure:
    """Observed vs. predicted daily demand as overlaid lines."""
    data = actual_vs_predicted(model, split, predictions)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data["date"],
            y=data["y_true"],
            name="Demanda real",
            mode="lines",
            line=dict(color=VIU_COLORS["near_black"], width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data["date"],
            y=data["y_pred"],
            name=f"Predicción · {model}",
            mode="lines",
            line=dict(color=FAMILY_COLOR[MODEL_FAMILY[model]], width=2, dash="dash"),
        )
    )

    # The train split is in-sample for every learned model; label it on the figure itself
    # so a screenshot cannot be mistaken for an out-of-sample result.
    # The train split is in-sample for every learned model; label it via an in-plot
    # annotation (not the removed top title) so a screenshot cannot be mistaken for an
    # out-of-sample result. The descriptive title itself lives only in the LaTeX caption.
    if split == "train":
        fig.add_annotation(
            text="⚠ IN-SAMPLE — no es una estimación honesta",
            xref="paper",
            yref="paper",
            x=0.5,
            y=1.10,
            showarrow=False,
            font=dict(size=12, color=VIU_COLORS["primary"]),
        )
    fig.update_layout(
        xaxis_title="Fecha",
        yaxis_title="Viajes/día",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, yanchor="bottom"),
    )
    return apply_theme(fig)


def comparison_figure(split: str, predictions: pd.DataFrame | None = None) -> go.Figure:
    """Four-panel grouped bars of MAE/RMSE/MAPE/R2, coloured by model family."""
    table = model_comparison_table(split, predictions)

    titles = [f"{m} ({'mayor mejor' if m == 'R2' else 'menor mejor'})" for m in METRICS]
    fig = make_subplots(
        rows=2, cols=2, subplot_titles=titles, vertical_spacing=0.17, horizontal_spacing=0.10
    )

    for i, metric in enumerate(METRICS):
        row, col = divmod(i, 2)
        labels = [
            f"{v:,.0f}" if metric in ("MAE", "RMSE") else f"{v:.3f}" for v in table[metric]
        ]
        fig.add_trace(
            go.Bar(
                x=table["model"],
                y=table[metric],
                marker_color=[FAMILY_COLOR[f] for f in table["family"]],
                text=labels,
                textposition="outside",
                cliponaxis=False,
                showlegend=False,
            ),
            row=row + 1,
            col=col + 1,
        )

    # Invisible traces purely to produce a family legend.
    for family, color in FAMILY_COLOR.items():
        fig.add_trace(
            go.Bar(x=[None], y=[None], name=family, marker_color=color, showlegend=True),
            row=1,
            col=1,
        )

    fig.update_layout(
        height=820,
        barmode="group",
        margin=dict(t=110),
        legend=dict(orientation="h", y=1.05, yanchor="bottom"),
    )
    fig.update_xaxes(tickangle=-40)
    return apply_theme(fig)


def residual_figure(
    model: str, split: str = "test", predictions: pd.DataFrame | None = None
) -> go.Figure:
    """Residual histogram, plus a train-vs-split box comparison when train exists."""
    residuals = residual_distribution(model, split, predictions)
    has_train = "train" in residuals

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"Distribución de residuos — {split}",
            "Train vs. test (brecha de sobreajuste)"
            if has_train
            else "Sin cobertura de train para este modelo",
        ),
    )
    fig.add_trace(
        go.Histogram(x=residuals[split], name=split, nbinsx=40, marker_color=VIU_COLORS["steel"]),
        row=1,
        col=1,
    )

    if has_train:
        fig.add_trace(
            go.Box(y=residuals["train"], name="train", marker_color=VIU_COLORS["gray"], boxmean="sd"),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Box(y=residuals[split], name=split, marker_color=VIU_COLORS["steel"], boxmean="sd"),
            row=1,
            col=2,
        )
        gap = residuals[split].abs().mean() / residuals["train"].abs().mean()
        fig.add_annotation(
            text=f"MAE {split} / MAE train = {gap:.2f}x",
            xref="paper",
            yref="paper",
            x=0.99,
            y=1.12,
            showarrow=False,
            font=dict(size=13, color=VIU_COLORS["primary"]),
        )

    fig.update_xaxes(title_text="Residuo (real − predicho)", row=1, col=1)
    fig.update_yaxes(title_text="Frecuencia", row=1, col=1)
    fig.update_yaxes(title_text="Residuo", row=1, col=2)
    fig.update_layout(
        showlegend=False,
        height=520,
    )
    return apply_theme(fig)


def day_type_figure(
    split: str = "test",
    models: list[str] | None = None,
    predictions: pd.DataFrame | None = None,
) -> go.Figure:
    """Grouped MAE bars per day type, with n annotated and small samples de-emphasised."""
    table = error_by_day_type(split, predictions)
    models = models or DEFAULT_BREAKDOWN_MODELS
    table = table[table["model"].isin(models)]

    groups = [g for g in DAY_TYPE_ORDER if g in set(table["group"])]

    fig = go.Figure()
    for model in models:
        rows = table[table["model"] == model].set_index("group").reindex(groups)
        fig.add_trace(
            go.Bar(
                name=model,
                x=groups,
                y=rows["MAE"],
                marker_color=FAMILY_COLOR[MODEL_FAMILY[model]],
                # Reduced opacity is the visual flag for n < 10, per the CLAUDE.md caveat
                # that festivo (n=5) and bridge (n=3) on test are indicative only.
                marker_opacity=[0.45 if bool(s) else 1.0 for s in rows["small_sample"]],
                text=[f"{v:,.0f}" for v in rows["MAE"]],
                textposition="outside",
                cliponaxis=False,
            )
        )

    counts = table.drop_duplicates("group").set_index("group").reindex(groups)["n"]
    fig.update_layout(
        xaxis=dict(
            tickmode="array",
            tickvals=groups,
            ticktext=[f"{g}<br><i>n = {int(counts[g])}</i>" for g in groups],
        ),
        yaxis_title="MAE (viajes/día)",
        barmode="group",
        height=620,
        legend=dict(orientation="h", y=1.02, yanchor="bottom"),
    )
    return apply_theme(fig)


def importance_group_figure(model: str) -> go.Figure:
    """Horizontal bars of gain aggregated by feature group."""
    by_group = feature_importance_by_group(model)["by_group"].sort_values("importance")

    fig = go.Figure(
        go.Bar(
            x=by_group["importance"],
            y=by_group["group"],
            orientation="h",
            marker_color=[GROUP_COLOR.get(g, VIU_COLORS["gray"]) for g in by_group["group"]],
            text=[
                f"{s:.1%}  ({int(n)} vars)"
                for s, n in zip(by_group["share"], by_group["n_features"])
            ],
            textposition="outside",
            cliponaxis=False,
        )
    )
    fig.update_layout(
        xaxis_title="Ganancia agregada",
        height=460,
        margin=dict(l=150, r=170),
    )
    return apply_theme(fig)


def importance_top_figure(model: str, top_n: int = 15) -> go.Figure:
    """Horizontal bars of the top-N individual features by gain."""
    top = feature_importance_by_group(model, top_n=top_n)["top_features"].sort_values(
        "importance"
    )

    fig = go.Figure(
        go.Bar(
            x=top["importance"],
            y=top["feature"],
            orientation="h",
            marker_color=[GROUP_COLOR.get(g, VIU_COLORS["gray"]) for g in top["group"]],
            text=[f"{v:.3f}" for v in top["importance"]],
            textposition="outside",
            cliponaxis=False,
        )
    )
    fig.update_layout(
        xaxis_title="Ganancia",
        height=620,
        margin=dict(l=360, r=110),
    )
    return apply_theme(fig)


# --------------------------------------------------------------------------------------
# Figure captions, ready to paste into the memoria
# --------------------------------------------------------------------------------------

# Numbering is fixed here rather than generated, so a caption keeps its number across
# re-runs and the memoria's cross-references stay valid.
#
# The descriptions deliberately REUSE the wording already present in the notebook's
# markdown sections rather than introducing new phrasing: caption and surrounding prose
# must say the same thing, and two independently-worded descriptions of one figure is how
# a results chapter starts contradicting itself.
FIGURE_NUMBERS: dict[str, int] = {
    "demand": 1,
    "comparison": 2,
    "residual": 3,
    "day_type": 4,
    "importance_group": 5,
    "importance_top": 6,
}

_CAPTION_TEMPLATES: dict[str, str] = {
    "demand": (
        "Demanda diaria real y predicha por el modelo {model} en la partición {split}. "
        "Serie observada y predicha superpuestas."
    ),
    "comparison": (
        "Comparación de modelos en la partición {split}. Barras agrupadas de MAE, RMSE, "
        "MAPE y R², ordenadas de mejor a peor por MAE. El color codifica la familia del "
        "modelo."
    ),
    "residual": (
        "Distribución del error de predicción del modelo {model} en la partición {split}. "
        "Si la distribución de residuos en train es mucho más estrecha que en test, el "
        "modelo ha memorizado el periodo de entrenamiento en lugar de generalizar."
    ),
    "day_type": (
        "MAE por tipo de día en la partición {split}. Los grupos con n < 10 se dibujan con "
        "opacidad reducida y su n anotado bajo el eje: son indicativos, no concluyentes."
    ),
    "importance_group": (
        "Importancia agregada por grupo de variables del modelo {model}, medida como "
        "ganancia."
    ),
    "importance_top": (
        "Top-{top_n} variables individuales por ganancia del modelo {model}."
    ),
}


def caption(fig_id: str, **kwargs) -> str:
    """Return a ready-to-paste 'Figura N. …' caption for one figure type.

    Args:
        fig_id: One of FIGURE_NUMBERS ('demand', 'comparison', 'residual', 'day_type',
            'importance_group', 'importance_top').
        **kwargs: Substituted into the template — typically `model`, `split`, `top_n`.

    Returns:
        A single line, e.g. "Figura 2. Comparación de modelos en la partición test. …"

    Raises:
        ValueError: for an unknown fig_id, or if a template placeholder is unfilled.
    """
    if fig_id not in FIGURE_NUMBERS:
        raise ValueError(
            f"caption: unknown fig_id {fig_id!r}; expected one of "
            f"{sorted(FIGURE_NUMBERS)}"
        )

    defaults = {"top_n": 15}
    values = {**defaults, **kwargs}

    try:
        body = _CAPTION_TEMPLATES[fig_id].format(**values)
    except KeyError as exc:
        raise ValueError(
            f"caption: template for {fig_id!r} needs {exc} — pass it as a keyword argument"
        ) from exc

    return f"Figura {FIGURE_NUMBERS[fig_id]}. {body}"
