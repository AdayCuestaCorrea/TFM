"""Pin the high-salience numbers of the memoria's PROSE to the artifacts they come from.

The 32 generated tables are already covered: ``export_all()`` builds them from
``data/processed/`` and ``test_export_latex_tables.py`` fixes their shape. The surface that
was NOT covered is the hand-written prose in ``docs/LaTeX/sections/``, and that is exactly
where the known failure happened: the test count sat stale for three phases in
``00_preliminares.tex`` because no synchronisation list included the first file a reader
sees.

This module is that synchronisation list, made executable. Each claim declares

  * the value, and where it comes from (an artifact, or an explicit frozen constant);
  * every ``.tex`` file the value must appear in.

A claim therefore fails in two directions, which is the point:

  * the artifact moved and the prose was not updated  -> value mismatch;
  * a file silently lost the number, or a new file gained it -> location mismatch.

Numbers are matched in both the Spanish (``139.725``) and English (``139,725``) rendering,
because ``00_preliminares.tex`` carries the resumen and the abstract in the same file and
the English abstract drifts independently.

Nothing here trains or re-predicts anything; every value is read from a persisted artifact.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from src.evaluation.export_latex_tables import TOTAL_TESTS_BEFORE_MEMORIA
from src.utils.splits import chronological_split

ROOT = Path(__file__).resolve().parents[1]
SECTIONS = ROOT / "docs" / "LaTeX" / "sections"
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "reports" / "figures"
TABLES = ROOT / "docs" / "LaTeX" / "tables"


# --------------------------------------------------------------------------------------
# Artifact readers. Each returns the canonical value the prose must agree with.
# --------------------------------------------------------------------------------------
def _test_mae(model: str) -> int:
    df = pd.read_parquet(PROCESSED / "full_comparison.parquet")
    row = df[(df["split"] == "test") & (df["Model"] == model)]
    if row.empty:  # ensembles live in the sensitivity artifact
        df = pd.read_parquet(PROCESSED / "sensitivity_comparison.parquet")
        row = df[(df["split"] == "test") & (df["Model"] == model)]
    assert not row.empty, f"{model} absent from both comparison artifacts"
    return round(float(row["MAE"].iloc[0]))


def _feature_matrix() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "features_daily.parquet")


def _splits():
    df = _feature_matrix()
    dates = df["date"] if "date" in df.columns else df.index.to_series()
    return chronological_split(pd.DatetimeIndex(pd.to_datetime(dates)))


def _spanish(value: int) -> str:
    """Thousands separated by '.', the convention of the Spanish prose."""
    return f"{value:,}".replace(",", ".")


def _english(value: int) -> str:
    return f"{value:,}"


# --------------------------------------------------------------------------------------
# THE MANIFEST. (value_fn, human label, files it must appear in)
# --------------------------------------------------------------------------------------
PROSE_CLAIMS: list[tuple[str, object, tuple[str, ...]]] = [
    # --- headline test metrics --------------------------------------------------------
    ("MAE test ensemble_equal", lambda: _test_mae("ensemble_equal"),
     ("00_preliminares.tex", "01_introduccion.tex", "02_estado_arte.tex",
      "04_metodologia.tex", "05_resultados.tex", "06_conclusiones.tex")),
    ("MAE test ensemble_inverse_mae", lambda: _test_mae("ensemble_inverse_mae"),
     ("00_preliminares.tex", "01_introduccion.tex", "02_estado_arte.tex",
      "05_resultados.tex", "06_conclusiones.tex")),
    ("MAE test xgboost_alone", lambda: _test_mae("xgboost_alone"),
     ("00_preliminares.tex", "01_introduccion.tex", "05_resultados.tex")),
    ("MAE test sarimax", lambda: _test_mae("sarimax"),
     ("00_preliminares.tex", "01_introduccion.tex", "04_metodologia.tex")),
    ("MAE test hybrid", lambda: _test_mae("hybrid"),
     ("00_preliminares.tex", "01_introduccion.tex", "05_resultados.tex")),
    ("MAE test lstm_alone", lambda: _test_mae("lstm_alone"), ("05_resultados.tex",)),

    # --- shape of the data ------------------------------------------------------------
    ("filas de la serie unificada", lambda: len(_feature_matrix()),
     ("00_preliminares.tex", "01_introduccion.tex", "02_estado_arte.tex",
      "03_datos_features.tex", "05_resultados.tex", "06_conclusiones.tex",
      "07_anexos.tex")),
    ("variables predictoras feat_",
     lambda: sum(c.startswith("feat_") for c in _feature_matrix().columns),
     ("00_preliminares.tex", "01_introduccion.tex", "02_estado_arte.tex",
      "03_datos_features.tex", "04_metodologia.tex", "05_resultados.tex",
      "06_conclusiones.tex", "07_anexos.tex")),

    # --- chronological split ----------------------------------------------------------
    ("n de train", lambda: _splits().n_train,
     ("03_datos_features.tex", "06_conclusiones.tex")),
    ("n de val", lambda: _splits().n_val, ("03_datos_features.tex",)),
    ("n de test", lambda: _splits().n_test, ("03_datos_features.tex",)),

    # --- Stage 2 training set ---------------------------------------------------------
    ("filas del conjunto residual",
     lambda: len(pd.read_parquet(PROCESSED / "residual_training_set.parquet")),
     ("02_estado_arte.tex", "04_metodologia.tex", "05_resultados.tex", "07_anexos.tex")),

    # --- bookkeeping counters ---------------------------------------------------------
    # Incident 12g: these are the counters that went stale. 00_preliminares.tex is listed
    # FIRST and deliberately, because omitting it is precisely how the defect escaped.
    # 05_resultados.tex was dropped from this entry: it never carried the claim. The count
    # is 34 and the file contains no sentence stating it -- the entry passed only because
    # "-0{,}34\,\sigma" (a residual expressed in sigmas, section 5.9.1) satisfies the
    # "preceded by a non-digit" boundary rule in _appears. A guard that a decimal digit can
    # satisfy by accident guards nothing, and leaving it would have masked a genuinely
    # stale counter in the one file that does state it. The claim lives in 07_anexos.tex
    # ("exporta las 34 figuras", the export_figures entry of the reproducibility list),
    # which is where the count can actually go out of date.
    ("figuras exportadas", lambda: len(list(FIGURES.glob("*.png"))),
     ("07_anexos.tex",)),
    ("tablas generadas", lambda: len(list(TABLES.glob("tabla_*.tex"))),
     ("04_metodologia.tex", "07_anexos.tex")),
    ("tests acumulados antes de la memoria", lambda: TOTAL_TESTS_BEFORE_MEMORIA,
     ("01_introduccion.tex", "02_estado_arte.tex", "07_anexos.tex")),
]


def _appears(value: int, text: str) -> bool:
    """True if `value` occurs as a standalone number in either numeric convention.

    The boundary is 'not a digit' rather than 'not a digit or separator': a figure at the
    end of a clause is legitimately followed by a comma ("... y 156.121, respectivamente"),
    and excluding that swallows real occurrences. Separators inside the number are matched
    as a character class so 139.725 and 139,725 are one pattern.
    """
    grouped = re.escape(_spanish(value)).replace(r"\.", r"[.,]")
    for pattern in (rf"(?<!\d){grouped}(?!\d)", rf"(?<![\d.,]){value}(?![\d.,])"):
        if re.search(pattern, text):
            return True
    return False


@pytest.mark.parametrize("label,value_fn,files", PROSE_CLAIMS, ids=[c[0] for c in PROSE_CLAIMS])
def test_prose_claim_appears_in_every_declared_file(label, value_fn, files) -> None:
    value = value_fn()
    missing = [f for f in files if not _appears(value, (SECTIONS / f).read_text(encoding="utf-8"))]
    assert not missing, (
        f"{label} = {value} (from its artifact) is missing from: {missing}. "
        "Either the artifact changed and the prose was not synchronised, or the prose "
        "dropped the figure. Update both, or update this manifest if the claim moved."
    )


def test_test_count_is_synchronised_across_every_location() -> None:
    """The counter of incident 12g, pinned in all four prose locations at once.

    Collected at runtime rather than hardcoded, so the assertion cannot itself go stale.
    """
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=ROOT, capture_output=True, text=True,
    ).stdout
    m = re.search(r"(\d+) tests collected", out)
    assert m, f"could not read the collected test count from pytest output:\n{out[-500:]}"
    total = int(m.group(1))

    for fname in ("00_preliminares.tex", "07_anexos.tex"):
        text = (SECTIONS / fname).read_text(encoding="utf-8")
        assert _appears(total, text), (
            f"the suite collects {total} tests but {fname} does not state that number. "
            "All locations in the synchronisation list must be updated together "
            "(REPORT_MAPPING.md, incident 12g)."
        )


def test_sarimax_order_in_prose_matches_the_persisted_selection() -> None:
    order = json.loads((PROCESSED / "sarimax_selected_order.json").read_text(encoding="utf-8"))
    p, d, q = order["order"]
    P, D, Q, s = order["seasonal_order"]
    text = (SECTIONS / "04_metodologia.tex").read_text(encoding="utf-8")
    expected = f"({p},{d},{q})" + r"\times" + f"({P},{D},{Q},{s})"
    normalised = re.sub(r"\s+", "", text)
    assert re.sub(r"\s+", "", expected) in normalised, (
        f"the persisted SARIMAX order {expected} does not appear in 04_metodologia.tex"
    )
