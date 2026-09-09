"""Master comparison table across every model built in Phases 3-5.

This is the results-chapter table for the memoria. It scores every model with the same
metrics.py used since Phase 3, on the same split boundaries from src/utils/splits.py, so
the numbers are directly comparable across phases.

READ THE ROW COUNTS. Not every model covers the same rows on every split:
  - the LSTM and the hybrid need a 28-day window, so they cannot predict the first 28 rows
  - the OOF-derived models are defined only over the training period
Within val and test, however, all models cover the identical rows, which is what the
headline comparison rests on. The `n` column is printed for exactly this reason.
"""

from pathlib import Path

import pandas as pd

from src.evaluation.metrics import comparison_table, format_table
from src.features.build_features import FEATURES_DAILY_FILE, feature_columns
from src.models.baselines.run_baselines import BASELINE_PREDICTIONS_FILE
from src.models.hybrid_residual.combine import HYBRID_PREDICTIONS_FILE
from src.models.hybrid_residual.xgboost_alone import ALONE_MODEL_FILE
from src.utils.paths import PROCESSED_DIR
from src.utils.splits import apply_warmup_policy, chronological_split, split_masks

FULL_COMPARISON_FILE: Path = PROCESSED_DIR / "full_comparison.parquet"

DESCRIPTIONS = {
    "persistence": "yhat_t = y_{t-1}",
    "seasonal_naive": "yhat_t = y_{t-7}",
    "moving_average_7": "trailing 7-day mean",
    "moving_average_28": "trailing 28-day mean",
    "sarimax": "SARIMAX(2,1,2)x(0,1,2,7) + calendar exog",
    "lstm_alone": "Stage 1 only: univariate LSTM(32), W=28",
    "xgboost_alone": "XGBoost on total, full feature matrix",
    "hybrid": "Stage 1 LSTM + Stage 2 XGBoost residual",
}

ORDER = [
    "hybrid",
    "xgboost_alone",
    "sarimax",
    "lstm_alone",
    "seasonal_naive",
    "moving_average_7",
    "moving_average_28",
    "persistence",
]


def assemble_predictions(
    baselines: pd.DataFrame | None = None,
    hybrid: pd.DataFrame | None = None,
    alone_payload: dict | None = None,
    features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join every model's predictions onto one date-indexed frame.

    Every argument defaults to reading the persisted artifact, so calling this with no
    arguments is byte-identical to the original behaviour and remains what `main()` does.

    The arguments exist so that a caller which has just RECOMPUTED these pieces in memory --
    `notebooks/07_flujo_completo.ipynb` -- can score them through this exact code path
    instead of reimplementing the join. That matters more than convenience: a notebook that
    rebuilt the join itself could diverge from the one that produced the published table, and
    the comparison would then measure the divergence rather than the pipeline.

    Args:
        baselines: Frame with [date, total, persistence, seasonal_naive, moving_average_7,
            moving_average_28, sarimax]. Defaults to `baseline_predictions.parquet`.
        hybrid: Frame with [date, y_pred_lstm, y_pred_hybrid]. Defaults to
            `hybrid_predictions.parquet`.
        alone_payload: `{"model": ..., "features": [...]}` for XGBoost-alone. Defaults to
            `models/xgboost_alone.joblib`.
        features: The full feature table. Defaults to `features_daily.parquet`.

    Returns:
        Wide frame: [date, y_true, one column per model, split].
    """
    import joblib

    feats = (
        features if features is not None else pd.read_parquet(FEATURES_DAILY_FILE)
    ).sort_values("date").reset_index(drop=True)
    bounds = chronological_split(feats["date"])
    trimmed, _ = apply_warmup_policy(feats, bounds)
    trimmed = trimmed.reset_index(drop=True)

    if baselines is None:
        baselines = pd.read_parquet(BASELINE_PREDICTIONS_FILE)
    if hybrid is None:
        hybrid = pd.read_parquet(HYBRID_PREDICTIONS_FILE)

    df = baselines.rename(columns={"total": "y_true"}).copy()

    # XGBoost-alone predicts every row from its stored feature list.
    payload = alone_payload if alone_payload is not None else joblib.load(ALONE_MODEL_FILE)
    model, feat_cols = payload["model"], payload["features"]
    alone = pd.DataFrame(
        {
            "date": trimmed["date"],
            "xgboost_alone": model.predict(trimmed[feat_cols]),
        }
    )
    df = df.merge(alone, on="date", how="left")

    df = df.merge(
        hybrid[["date", "y_pred_lstm", "y_pred_hybrid"]].rename(
            columns={"y_pred_lstm": "lstm_alone", "y_pred_hybrid": "hybrid"}
        ),
        on="date",
        how="left",
    )

    masks = split_masks(df, bounds)
    import numpy as np

    df["split"] = np.select(
        [masks["train"], masks["val"], masks["test"]], ["train", "val", "test"], "unassigned"
    )
    return df


def build_tables(df: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """One comparison table per split."""
    df = df if df is not None else assemble_predictions()
    models = [m for m in ORDER if m in df.columns]

    tables: dict[str, pd.DataFrame] = {}
    for split in ("train", "val", "test"):
        subset = df[df["split"] == split]
        results = {}
        for name in models:
            pair = subset[["y_true", name]].dropna()
            if pair.empty:
                continue
            results[name] = (pair["y_true"], pair[name])
        if results:
            tables[split] = comparison_table(results, DESCRIPTIONS)
    return tables


METRIC_COLS = ["RMSE", "MAE", "MAPE", "R2"]

# Pass bar. Measured, not guessed: on the reference machine every one of the 88 metric cells
# reproduces at a relative deviation of exactly 0.0 (XGBoost refits bit-for-bit, and the
# baselines and SARIMAX reproduce their parquet exactly). The tolerance is not absorbing known
# drift -- it exists solely for float reassociation in XGBoost's multithreaded `hist`
# histogram, which sums in a different order when the core count changes. 1e-6 relative on a
# test MAE of ~156,000 is 0.16 passengers/day, six orders of magnitude below anything the
# memoria reports.
DEFAULT_RTOL = 1e-6

# Interpretation boundary, NOT a second pass bar. A deviation between DEFAULT_RTOL and this
# value is too large for a single machine but far too small to be a pipeline change; it is the
# signature of a different execution environment. Above it, the arithmetic itself has changed.
# 1e-3 relative on a test MAE of ~156,000 is ~156 passengers/day -- still invisible at the
# precision the memoria reports, while a real change (a dropped shift, a moved split boundary,
# a different feature set) moves the metrics by thousands.
ENV_DRIFT_CEILING = 1e-3


def load_published(path: Path | str = FULL_COMPARISON_FILE) -> pd.DataFrame:
    """The published master table that a recomputed run is verified against."""
    path = Path(path)
    if not path.exists():
        raise ValueError(
            f"load_published: missing {path} -- run `python -m src.evaluation.full_comparison`"
        )
    return pd.read_parquet(path)


def verify_against_published(
    recomputed: pd.DataFrame, published: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Contrast recomputed metrics against the published master table, cell by cell.

    This function NEVER raises on a numeric mismatch -- it reports one. Separating the
    contrast from the verdict is deliberate: the caller must be able to display the full
    table BEFORE anything blows up, so a reader who does hit a failure sees which cell
    drifted and by how much instead of an exception with the evidence still unprinted.
    `assert_within_tolerance` renders the verdict.

    Args:
        recomputed: A table in `full_comparison` layout, with at least
            [Model, split, RMSE, MAE, MAPE, R2, n].
        published: The reference table; defaults to `full_comparison.parquet` on disk.

    Returns:
        Long DataFrame, one row per (Model, split, metric): published value, recomputed
        value, absolute and relative deviation.

    Raises:
        ValueError: if the two tables do not cover the identical set of (Model, split) keys.
            A missing or extra row is a structural defect, not a numeric one, and no
            tolerance can express it -- so it fails here rather than being scored away.
    """
    published = published if published is not None else load_published()

    key = ["Model", "split"]
    published_keys = set(map(tuple, published[key].itertuples(index=False, name=None)))
    recomputed_keys = set(map(tuple, recomputed[key].itertuples(index=False, name=None)))
    if published_keys != recomputed_keys:
        missing = sorted(published_keys - recomputed_keys)
        extra = sorted(recomputed_keys - published_keys)
        raise ValueError(
            "verify_against_published: the recomputed table does not cover the same "
            f"(Model, split) rows as the published one. Missing: {missing}. Extra: {extra}."
        )

    merged = published.merge(recomputed, on=key, suffixes=("_pub", "_new"))

    rows = []
    for _, row in merged.iterrows():
        for metric in METRIC_COLS:
            pub, new = float(row[f"{metric}_pub"]), float(row[f"{metric}_new"])
            abs_dev = abs(new - pub)
            # R2 is the one metric that can legitimately approach zero, where a relative
            # deviation would explode on a meaningless absolute difference. Guard the
            # denominator rather than special-casing the metric.
            rel_dev = abs_dev / max(abs(pub), 1e-12)
            rows.append(
                {
                    "Model": row["Model"],
                    "split": row["split"],
                    "metrica": metric,
                    "publicado": pub,
                    "recomputado": new,
                    "desviacion_abs": abs_dev,
                    "desviacion_rel": rel_dev,
                }
            )

    # `n` is a row count, not a measurement: compared for exact equality, because a difference
    # means the two tables scored different observations, which invalidates every metric above
    # it. An inequality is reported as an infinite relative deviation so no rtol can pass it.
    for _, row in merged.iterrows():
        n_pub, n_new = int(row["n_pub"]), int(row["n_new"])
        rows.append(
            {
                "Model": row["Model"],
                "split": row["split"],
                "metrica": "n",
                "publicado": float(n_pub),
                "recomputado": float(n_new),
                "desviacion_abs": float(abs(n_new - n_pub)),
                "desviacion_rel": 0.0 if n_pub == n_new else float("inf"),
            }
        )

    out = pd.DataFrame(rows)
    return out.sort_values(["split", "Model", "metrica"]).reset_index(drop=True)


def assert_within_tolerance(comparison: pd.DataFrame, rtol: float = DEFAULT_RTOL) -> str:
    """Render the verdict on a `verify_against_published` table.

    Returns:
        A one-line success message, for the caller to print.

    Raises:
        AssertionError: if any cell exceeds `rtol`. The message is written for the person
            reading the notebook, not for a stack trace: it names the worst offending cell,
            quotes both values and the deviation, and INTERPRETS the magnitude -- because the
            first conclusion a reader draws from a red cell is "this work does not reproduce",
            and at float-reassociation scale that conclusion is wrong.
    """
    n_cells = len(comparison)
    offenders = comparison[comparison["desviacion_rel"] > rtol]

    if offenders.empty:
        return (
            f"OK - {n_cells}/{n_cells} valores coinciden con full_comparison.parquet "
            f"(rtol={rtol:g}; desviacion relativa maxima observada: "
            f"{comparison['desviacion_rel'].max():.3e})"
        )

    worst = offenders.loc[offenders["desviacion_rel"].idxmax()]
    worst_rel = float(worst["desviacion_rel"])

    if worst_rel <= ENV_DRIFT_CEILING:
        interpretacion = (
            f"INTERPRETACION: una desviacion de esta magnitud ({worst_rel:.3e} relativo) NO "
            "indica que el pipeline haya cambiado. Es la firma de una diferencia de ENTORNO "
            "DE EJECUCION: XGBoost usa tree_method='hist' con n_jobs=-1, y su histograma "
            "multihilo suma en coma flotante en un orden distinto segun el numero de nucleos "
            "de la maquina. Los resultados publicados en la memoria se mantienen: a la "
            "precision con la que se reportan (viajeros enteros, MAPE a dos decimales) esta "
            "diferencia es invisible.\n"
            "  Una desviacion REAL del pipeline -- un shift perdido, una frontera de "
            "particion desplazada, un conjunto de variables distinto -- mueve las metricas en "
            "MILES de viajeros, varios ordenes de magnitud por encima de lo observado aqui."
        )
    else:
        interpretacion = (
            f"INTERPRETACION: una desviacion de esta magnitud ({worst_rel:.3e} relativo, "
            f"{float(worst['desviacion_abs']):,.1f} en valor absoluto) SI indica un cambio "
            "real en el pipeline, no una diferencia de entorno. La reasociacion en coma "
            f"flotante entre maquinas se queda por debajo de {ENV_DRIFT_CEILING:g} relativo. "
            "Revisar, por este orden: la matriz de variables (161 columnas feat_), las "
            "fronteras de la particion cronologica, la politica de warm-up de 28 filas, y las "
            "versiones fijadas en requirements.txt."
        )

    detalle = "\n".join(
        f"    {r['Model']:<18} {r['split']:<6} {r['metrica']:<5} "
        f"publicado={r['publicado']:>16,.6f}  recomputado={r['recomputado']:>16,.6f}  "
        f"desv.rel={r['desviacion_rel']:.3e}"
        for _, r in offenders.sort_values("desviacion_rel", ascending=False)
        .head(10)
        .iterrows()
    )

    raise AssertionError(
        "\nVERIFICACION FALLIDA contra data/processed/full_comparison.parquet\n"
        "  QUE SE COMPARA : las metricas RMSE/MAE/MAPE/R2 y el recuento n de cada modelo en "
        "cada particion, recomputadas en vivo en este cuaderno, frente a los valores "
        "publicados en la memoria.\n"
        f"  RESULTADO      : {len(offenders)} de {n_cells} valores fuera de tolerancia.\n"
        f"  TOLERANCIA     : rtol={rtol:g} (desviacion relativa).\n"
        f"  PEOR CELDA     : {worst['Model']} / {worst['split']} / {worst['metrica']}: "
        f"publicado={float(worst['publicado']):,.6f}, "
        f"recomputado={float(worst['recomputado']):,.6f}, "
        f"desviacion relativa={worst_rel:.3e}.\n"
        f"  {interpretacion}\n"
        f"  CELDAS AFECTADAS (hasta 10, peores primero):\n{detalle}\n"
    )


def build_comparison_frame(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Stack the per-split tables into the long layout of `full_comparison.parquet`."""
    return pd.concat(
        [t.assign(split=split) for split, t in tables.items()], ignore_index=True
    )


def main() -> None:
    df = assemble_predictions()
    tables = build_tables(df)

    for split in ("train", "val", "test"):
        if split not in tables:
            continue
        print("\n" + "=" * 96)
        title = f"MASTER COMPARISON — {split.upper()} SPLIT"
        if split == "train":
            title += "   (in-sample for LSTM/XGBoost; reference only)"
        print(title)
        print("=" * 96)
        print(format_table(tables[split]))

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    build_comparison_frame(tables).to_parquet(FULL_COMPARISON_FILE, index=False)
    print(f"\nsaved: {FULL_COMPARISON_FILE}")


if __name__ == "__main__":
    main()
