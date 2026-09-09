"""Presentation-ready tables for the memoria — Spanish headers, rounded, sorted.

These wrap `dashboard_data.py` outputs for display; they compute nothing. Metric values
come from the same code path as every table already in CLAUDE.md, so a number shown here
cannot disagree with one reported there.

RETURNS PLAIN DataFrames, NOT `.style` OBJECTS — deliberately. A pandas Styler renders as
CSS-heavy HTML that Google Docs pastes as an image or as mangled markup; a plain DataFrame
renders as a native HTML `<table>` that pastes as a real, editable Google Docs table with
selectable text. Gradients and bar charts inside cells are not worth losing that.

ROUNDING: whole passengers for MAE/RMSE, two decimals for MAPE, **four for R²**.

R² gets four decimals rather than two because at two the top test values
(0.9755 / 0.9690 / 0.9672) all collapse to 0.98 / 0.97 / 0.97, and the table stops
separating the ensemble from XGBoost-alone and SARIMAX on that column -- the exact
distinction the results chapter needs. MAPE stays at two decimals: at this project's scale
(3.13% / 3.30% / 3.85%) those values are already well separated and do not suffer the same
collapse. `r2_decimals` remains a parameter for a table that deliberately wants coarser
figures.
"""

import pandas as pd

from src.evaluation.dashboard_data import (
    error_by_day_type,
    model_comparison_table,
)

MAE_RMSE_DECIMALS = 0
MAPE_DECIMALS = 2
# Four, not two: see the rounding note in the module docstring. At two decimals the three
# leading models are indistinguishable on this column.
R2_DECIMALS = 4

FAMILY_ES = {
    "naive baseline": "Baseline ingenuo",
    "single-stage": "Una etapa",
    "LSTM family": "Familia LSTM",
    "ensemble": "Ensemble",
}

DAY_TYPE_ES = {
    "laborable": "Laborable",
    "laborable, ordinary": "Laborable ordinario",
    "laborable, bridge day": "Laborable, puente",
    "sabado": "Sábado",
    "domingo": "Domingo",
    "festivo": "Festivo",
    "domingo festivo": "Domingo festivo",
}


def comparison_table(
    split: str = "test",
    predictions: pd.DataFrame | None = None,
    r2_decimals: int = R2_DECIMALS,
) -> pd.DataFrame:
    """Model comparison for one split, ready to paste.

    Returns:
        DataFrame [Modelo, Familia, n, MAE, RMSE, MAPE (%), R²], sorted best-to-worst by
        MAE — the same ordering `comparison_figure` uses, so table and chart read together.
    """
    table = model_comparison_table(split, predictions).copy()

    table["family"] = table["family"].map(FAMILY_ES)
    table["MAE"] = table["MAE"].round(MAE_RMSE_DECIMALS).astype("int64")
    table["RMSE"] = table["RMSE"].round(MAE_RMSE_DECIMALS).astype("int64")
    table["MAPE"] = table["MAPE"].round(MAPE_DECIMALS)
    table["R2"] = table["R2"].round(r2_decimals)

    return table.rename(
        columns={
            "model": "Modelo",
            "family": "Familia",
            "n": "n",
            "MAPE": "MAPE (%)",
            "R2": "R²",
        }
    )[["Modelo", "Familia", "n", "MAE", "RMSE", "MAPE (%)", "R²"]].reset_index(drop=True)


def day_type_table(
    split: str = "test",
    models: list[str] | None = None,
    predictions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per-day-type MAE by model, wide format, ready to paste.

    Returns:
        DataFrame with [Tipo de día, n] followed by one MAE column per model. The `n`
        column is retained so the small-sample caveat (festivo n=5, puente n=3 on test)
        travels with the table into the memoria.
    """
    from src.evaluation.dashboard_figures import DAY_TYPE_ORDER, DEFAULT_BREAKDOWN_MODELS

    long = error_by_day_type(split, predictions)
    models = models or DEFAULT_BREAKDOWN_MODELS
    long = long[long["model"].isin(models)]

    wide = long.pivot(index="group", columns="model", values="MAE")
    counts = long.drop_duplicates("group").set_index("group")["n"]

    order = [g for g in DAY_TYPE_ORDER if g in wide.index]
    wide = wide.reindex(order)

    out = pd.DataFrame({"Tipo de día": [DAY_TYPE_ES.get(g, g) for g in wide.index]})
    out["n"] = [int(counts[g]) for g in wide.index]
    for model in models:
        if model in wide.columns:
            out[model] = wide[model].round(MAE_RMSE_DECIMALS).astype("int64").to_numpy()

    return out.reset_index(drop=True)


def hyperparameters_table() -> pd.DataFrame:
    """Selected hyperparameters of both XGBoost models, as reported in Phases 5 and 5b.

    Read from the persisted model artifacts rather than retyped, so the table cannot drift
    from what was actually fitted.

    Returns:
        DataFrame [Modelo, Filas entrenamiento, max_depth, n_estimators, learning_rate,
        min_child_weight].
    """
    import joblib

    from src.evaluation.dashboard_data import XGB_ALONE_MODEL, XGB_RESIDUAL_MODEL, _require

    rows = []
    for label, path, n_train in (
        ("xgboost_residual", XGB_RESIDUAL_MODEL, 552),
        ("xgboost_alone", XGB_ALONE_MODEL, 889),
    ):
        params = joblib.load(_require(path))["params"]
        rows.append(
            {
                "Modelo": label,
                "Filas entrenamiento": n_train,
                "max_depth": params["max_depth"],
                "n_estimators": params["n_estimators"],
                "learning_rate": params["learning_rate"],
                "min_child_weight": params["min_child_weight"],
            }
        )

    return pd.DataFrame(rows)


def feature_importance_table(model: str, top_n: int = 15) -> pd.DataFrame:
    """Top-N features by gain for one XGBoost model, ready to paste.

    Returns:
        DataFrame [#, Variable, Grupo, Ganancia] with gain to four decimals -- importances
        are small fractions, so two decimals would round most of them to 0.00.
    """
    from src.evaluation.dashboard_data import feature_importance_by_group

    top = feature_importance_by_group(model, top_n=top_n)["top_features"].copy()
    top.insert(0, "#", range(1, len(top) + 1))
    top["importance"] = top["importance"].round(4)

    return top.rename(
        columns={"feature": "Variable", "group": "Grupo", "importance": "Ganancia"}
    ).reset_index(drop=True)


def group_importance_table(model: str) -> pd.DataFrame:
    """Feature-group importance for one XGBoost model, ready to paste.

    Returns:
        DataFrame [Grupo, N.º variables, Ganancia, Peso (%)] sorted by descending gain.
    """
    from src.evaluation.dashboard_data import feature_importance_by_group

    by_group = feature_importance_by_group(model)["by_group"].copy()
    by_group["importance"] = by_group["importance"].round(4)
    by_group["share"] = (by_group["share"] * 100).round(MAPE_DECIMALS)

    return by_group.rename(
        columns={
            "group": "Grupo",
            "n_features": "N.º variables",
            "importance": "Ganancia",
            "share": "Peso (%)",
        }
    )[["Grupo", "N.º variables", "Ganancia", "Peso (%)"]].reset_index(drop=True)
