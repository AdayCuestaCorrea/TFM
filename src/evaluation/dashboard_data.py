"""Data access layer for the Phase 6 results dashboard.

Everything the notebook needs, as pure functions over artifacts already written by Phases
3-5b. NOTHING here trains, refits, or recomputes a prediction: the dashboard must show the
numbers that are already in CLAUDE.md, and a dashboard that quietly recomputes them is a
dashboard that can quietly disagree with the thesis it illustrates.

COVERAGE IS NOT UNIFORM, and callers must not assume it is. The baselines, xgboost_alone
and the ensembles cover all three splits; the LSTM-derived models (lstm_alone, hybrid,
hybrid_weighted) exist only on val and test, because Phase 4 saved stage-1 predictions for
those splits alone. Functions here return what exists rather than padding with NaN rows,
and `available_models(split)` reports it explicitly.
"""

import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.evaluation.metrics import all_metrics
from src.features.build_features import FEATURES_DAILY_FILE
from src.features.weather_features import WEATHER_COLS
from src.utils.paths import MODELS_DIR, PROCESSED_DIR
from src.utils.splits import chronological_split, split_masks

# --------------------------------------------------------------------------------------
# Artifact registry: path -> the phase that produces it, for actionable error messages
# --------------------------------------------------------------------------------------

BASELINE_FILE = PROCESSED_DIR / "baseline_predictions.parquet"
LSTM_FILE = PROCESSED_DIR / "lstm_val_test_predictions.parquet"
HYBRID_FILE = PROCESSED_DIR / "hybrid_predictions.parquet"
HYBRID_WEIGHTED_FILE = PROCESSED_DIR / "hybrid_weighted_predictions.parquet"
ENSEMBLE_FILE = PROCESSED_DIR / "ensemble_predictions.parquet"

XGB_RESIDUAL_MODEL = MODELS_DIR / "xgboost_residual.joblib"
XGB_ALONE_MODEL = MODELS_DIR / "xgboost_alone.joblib"

_PRODUCED_BY = {
    BASELINE_FILE: "Phase 3 — python -m src.models.baselines.run_baselines",
    LSTM_FILE: "Phase 4 — python -m src.models.lstm.final_model",
    HYBRID_FILE: "Phase 5 — python -m src.models.hybrid_residual.combine",
    HYBRID_WEIGHTED_FILE: (
        "Phase 5b — python -m src.models.hybrid_residual.xgboost_residual_weighted"
    ),
    ENSEMBLE_FILE: "Phase 5b — python -m src.evaluation.ensemble_baseline",
    XGB_RESIDUAL_MODEL: "Phase 5 — python -m src.models.hybrid_residual.xgboost_residual",
    XGB_ALONE_MODEL: "Phase 5 — python -m src.models.hybrid_residual.xgboost_alone",
    FEATURES_DAILY_FILE: "Phase 2/3 — python -m src.features.build_features",
}

MODEL_NAMES = [
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
]

# Model families, used by the dashboard to make "the best model uses no LSTM" legible.
MODEL_FAMILY = {
    "persistence": "naive baseline",
    "seasonal_naive": "naive baseline",
    "moving_average_7": "naive baseline",
    "moving_average_28": "naive baseline",
    "sarimax": "single-stage",
    "xgboost_alone": "single-stage",
    "lstm_alone": "LSTM family",
    "hybrid": "LSTM family",
    "hybrid_weighted": "LSTM family",
    "ensemble_equal": "ensemble",
    "ensemble_inverse_mae": "ensemble",
}

SPLITS = ["train", "val", "test"]


def _require(path: Path) -> Path:
    """Raise an actionable FileNotFoundError naming the phase that produces `path`."""
    if not Path(path).exists():
        producer = _PRODUCED_BY.get(path, "an earlier phase")
        raise FileNotFoundError(
            f"Missing artifact: {path}\n"
            f"  Produced by: {producer}\n"
            "  Phase 6 never retrains anything, so run that phase first."
        )
    return Path(path)


# --------------------------------------------------------------------------------------
# Predictions
# --------------------------------------------------------------------------------------


def load_all_predictions() -> pd.DataFrame:
    """Every model's predictions in tidy long format.

    Returns:
        DataFrame with columns [date, split, model, y_true, y_pred], one row per
        (date, model). `split` is always one of {train, val, test} -- rows that fall
        outside the Phase 3 boundaries are an error, not a category. `model` takes values
        from MODEL_NAMES. Coverage differs by model (see the module docstring): the
        LSTM-derived models appear only on val and test.

    Raises:
        FileNotFoundError: if any required artifact is missing.
        ValueError: if any row cannot be assigned to a split, or a (date, model) pair is
            duplicated.
    """
    baselines = pd.read_parquet(_require(BASELINE_FILE))
    ensembles = pd.read_parquet(_require(ENSEMBLE_FILE))
    lstm = pd.read_parquet(_require(LSTM_FILE))
    hybrid = pd.read_parquet(_require(HYBRID_FILE))
    hybrid_w = pd.read_parquet(_require(HYBRID_WEIGHTED_FILE))

    frames: list[pd.DataFrame] = []

    def add(dates, y_true, y_pred, model: str) -> None:
        frames.append(
            pd.DataFrame(
                {
                    "date": pd.to_datetime(dates).to_numpy(),
                    "model": model,
                    "y_true": np.asarray(y_true, dtype="float64"),
                    "y_pred": np.asarray(y_pred, dtype="float64"),
                }
            )
        )

    for model in ("persistence", "seasonal_naive", "moving_average_7", "moving_average_28", "sarimax"):
        add(baselines["date"], baselines["total"], baselines[model], model)

    for model in ("xgboost_alone", "ensemble_equal", "ensemble_inverse_mae"):
        add(ensembles["date"], ensembles["y_true"], ensembles[model], model)

    add(lstm["date"], lstm["y_true"], lstm["y_pred_lstm"], "lstm_alone")
    add(hybrid["date"], hybrid["y_true"], hybrid["y_pred_hybrid"], "hybrid")
    add(
        hybrid_w["date"],
        hybrid_w["y_true"],
        hybrid_w["y_pred_hybrid_weighted"],
        "hybrid_weighted",
    )

    tidy = pd.concat(frames, ignore_index=True)

    # Assign splits from the single source of truth rather than trusting the per-artifact
    # 'split' columns, so a disagreement between artifacts surfaces here.
    feats = pd.read_parquet(_require(FEATURES_DAILY_FILE))
    bounds = chronological_split(feats["date"].sort_values())
    masks = split_masks(tidy, bounds)
    tidy["split"] = np.select(
        [masks["train"], masks["val"], masks["test"]], SPLITS, default="unassigned"
    )

    unassigned = tidy[tidy["split"] == "unassigned"]
    if not unassigned.empty:
        raise ValueError(
            f"{len(unassigned)} row(s) fall outside the Phase 3 split boundaries, "
            f"e.g. {unassigned['date'].head(3).dt.strftime('%Y-%m-%d').tolist()}"
        )

    duplicated = tidy.duplicated(subset=["date", "model"])
    if duplicated.any():
        raise ValueError(f"{int(duplicated.sum())} duplicated (date, model) pair(s)")

    # Drop rows where a model has no prediction (warm-up lags in the naive baselines).
    tidy = tidy.dropna(subset=["y_pred"]).reset_index(drop=True)

    return tidy[["date", "split", "model", "y_true", "y_pred"]].sort_values(
        ["model", "date"]
    ).reset_index(drop=True)


def available_models(split: str, predictions: pd.DataFrame | None = None) -> list[str]:
    """Models that actually have predictions on `split`, in MODEL_NAMES order."""
    df = predictions if predictions is not None else load_all_predictions()
    present = set(df.loc[df["split"] == split, "model"].unique())
    return [m for m in MODEL_NAMES if m in present]


def model_comparison_table(
    split: str, predictions: pd.DataFrame | None = None
) -> pd.DataFrame:
    """MAE/RMSE/MAPE/R2 per model for one split, sorted best-to-worst by MAE.

    Metrics come from src/evaluation/metrics.py unchanged, so these numbers are identical
    to the tables already recorded in CLAUDE.md.

    Returns:
        DataFrame [model, family, n, MAE, RMSE, MAPE, R2].
    """
    if split not in SPLITS:
        raise ValueError(f"model_comparison_table: split must be one of {SPLITS}")

    df = predictions if predictions is not None else load_all_predictions()
    subset = df[df["split"] == split]

    rows = []
    for model in available_models(split, df):
        pair = subset[subset["model"] == model]
        scores = all_metrics(pair["y_true"], pair["y_pred"])
        rows.append(
            {
                "model": model,
                "family": MODEL_FAMILY[model],
                "n": int(scores["n"]),
                "MAE": scores["MAE"],
                "RMSE": scores["RMSE"],
                "MAPE": scores["MAPE"],
                "R2": scores["R2"],
            }
        )

    return pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Day-type breakdown
# --------------------------------------------------------------------------------------

SMALL_SAMPLE_THRESHOLD = 10


def error_by_day_type(
    split: str, predictions: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Per-day-type MAE/MAPE for EVERY model with coverage on `split`.

    Extends Phase 5's four-model breakdown to all models. The `n` column is included and
    `small_sample` flags groups below SMALL_SAMPLE_THRESHOLD, so the festivo (n=5) and
    bridge-day (n=3) groups on test can be marked in the figure rather than presented with
    the same visual weight as the 133-row laborable group.

    Returns:
        DataFrame [model, group, n, MAE, MAPE, small_sample].
    """
    if split not in SPLITS:
        raise ValueError(f"error_by_day_type: split must be one of {SPLITS}")

    from src.evaluation.day_type_breakdown import BRIDGE_COL, DAY_TYPE_COL, breakdown

    df = predictions if predictions is not None else load_all_predictions()
    feats = pd.read_parquet(_require(FEATURES_DAILY_FILE))[
        ["date", DAY_TYPE_COL, BRIDGE_COL]
    ]

    subset = df[df["split"] == split].merge(feats, on="date", how="left")

    rows = []
    for model in available_models(split, df):
        pair = subset[subset["model"] == model].reset_index(drop=True)
        table = breakdown(pair["y_true"], pair["y_pred"], pair)
        table.insert(0, "model", model)
        rows.append(table)

    out = pd.concat(rows, ignore_index=True)
    out["small_sample"] = out["n"] < SMALL_SAMPLE_THRESHOLD
    return out


# --------------------------------------------------------------------------------------
# Residuals
# --------------------------------------------------------------------------------------


def residual_distribution(
    model: str, split: str, predictions: pd.DataFrame | None = None
) -> dict[str, pd.Series]:
    """Residuals (y_true - y_pred) for `model`, plus its train residuals when they exist.

    Returns:
        {split: Series of residuals, 'train': Series} -- the requested split always, and a
        'train' entry only when the model has training-split coverage. The pairing is the
        point: xgboost_alone's train R2 of 0.9929 against a 156,121 test MAE is visible as
        a much narrower train residual distribution, which is what an overfit gap looks
        like. The LSTM-derived models have no train coverage, so their dict has one key.

    Raises:
        ValueError: for an unknown model or split, or a model absent from `split`.
    """
    if split not in SPLITS:
        raise ValueError(f"residual_distribution: split must be one of {SPLITS}")
    if model not in MODEL_NAMES:
        raise ValueError(f"residual_distribution: unknown model {model!r}")

    df = predictions if predictions is not None else load_all_predictions()
    rows = df[df["model"] == model]

    requested = rows[rows["split"] == split]
    if requested.empty:
        raise ValueError(
            f"residual_distribution: model {model!r} has no predictions on split {split!r}"
        )

    out = {split: (requested["y_true"] - requested["y_pred"]).reset_index(drop=True)}

    train_rows = rows[rows["split"] == "train"]
    if split != "train" and not train_rows.empty:
        out["train"] = (train_rows["y_true"] - train_rows["y_pred"]).reset_index(drop=True)

    return out


# --------------------------------------------------------------------------------------
# Feature importance
# --------------------------------------------------------------------------------------

# Explicit, ordered mapping from column name to feature group. Ordered because several
# patterns would otherwise overlap -- weather columns also carry _lag_ and _roll_mean_
# suffixes, so the target-specific patterns must be tested first. Defined once as a
# constant rather than inferred per call, so the grouping cannot drift between figures.
_WEATHER_ALTERNATION = "|".join(re.escape(c) for c in WEATHER_COLS)

FEATURE_GROUP_PATTERNS: list[tuple[str, str]] = [
    ("fourier_weekly", r"^feat_fourier_weekly_"),
    ("fourier_annual", r"^feat_fourier_annual_"),
    ("calendar", r"^feat_(day_type_|holiday_type_|is_weekend$|is_bridge_day$)"),
    ("lag_total", r"^feat_total_lag_\d+$"),
    ("rolling", r"^feat_total_roll_(mean|std)_\d+$"),
    ("lag_operator", r"^feat_(metro|emt|carretera|cercanias)_lag_\d+$"),
    ("weather", rf"^feat_({_WEATHER_ALTERNATION})_(lag_\d+|roll_mean_\d+)$"),
]

FEATURE_GROUPS = [name for name, _ in FEATURE_GROUP_PATTERNS]

MODEL_ARTIFACTS = {
    "xgboost_residual": XGB_RESIDUAL_MODEL,
    "xgboost_alone": XGB_ALONE_MODEL,
}


def classify_feature(name: str) -> str:
    """Map one feature column to its group.

    Raises:
        ValueError: if the name matches no pattern. Failing loudly matters here -- a
            silently unmatched feature would make the group-level totals disagree with the
            model's actual importances, which is exactly what the dashboard is meant to
            show faithfully.
    """
    for group, pattern in FEATURE_GROUP_PATTERNS:
        if re.match(pattern, name):
            return group
    raise ValueError(
        f"classify_feature: {name!r} matches no group in FEATURE_GROUP_PATTERNS. "
        "Extend the mapping deliberately rather than letting the feature vanish from the "
        "group totals."
    )


def feature_importance_by_group(
    model: str, top_n: int = 15
) -> dict[str, pd.DataFrame]:
    """Gain-based importance for an XGBoost model, grouped and per-feature.

    Args:
        model: 'xgboost_residual' or 'xgboost_alone'.
        top_n: How many individual features to return.

    Returns:
        {'by_group': DataFrame [group, importance, n_features, share],
         'top_features': DataFrame [feature, group, importance]}.
        Group importances sum to the model's total raw importance -- every feature is
        assigned to exactly one group, verified by test.

    Raises:
        ValueError: for an unknown model name.
        FileNotFoundError: if the model artifact is missing.
    """
    if model not in MODEL_ARTIFACTS:
        raise ValueError(
            f"feature_importance_by_group: unknown model {model!r}; "
            f"expected one of {sorted(MODEL_ARTIFACTS)}"
        )

    payload = joblib.load(_require(MODEL_ARTIFACTS[model]))
    estimator, features = payload["model"], payload["features"]

    importances = np.asarray(estimator.feature_importances_, dtype="float64")
    if len(importances) != len(features):
        raise ValueError(
            f"feature_importance_by_group: {model} reports {len(importances)} importances "
            f"for {len(features)} stored feature names."
        )

    table = pd.DataFrame({"feature": features, "importance": importances})
    table["group"] = table["feature"].map(classify_feature)

    by_group = (
        table.groupby("group", as_index=False)
        .agg(importance=("importance", "sum"), n_features=("feature", "size"))
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    total = by_group["importance"].sum()
    by_group["share"] = by_group["importance"] / total if total else 0.0

    top_features = (
        table.sort_values("importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)[["feature", "group", "importance"]]
    )

    return {"by_group": by_group, "top_features": top_features}


def actual_vs_predicted(
    model: str, split: str, predictions: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Date-indexed actual and predicted series for one model and split, for line charts.

    Returns:
        DataFrame [date, y_true, y_pred] sorted by date.
    """
    df = predictions if predictions is not None else load_all_predictions()
    rows = df[(df["model"] == model) & (df["split"] == split)]
    if rows.empty:
        raise ValueError(f"actual_vs_predicted: no rows for {model!r} on {split!r}")
    return rows[["date", "y_true", "y_pred"]].sort_values("date").reset_index(drop=True)
