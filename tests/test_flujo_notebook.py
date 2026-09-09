"""Tests for the main workflow notebook and the two functions it leans on.

`notebooks/07_flujo_completo.ipynb` recomputes the deterministic half of the pipeline and
contrasts every metric against `full_comparison.parquet`, raising on any drift. That makes the
notebook itself a reproducibility check -- but only while it actually runs, and only while its
safeguard actually fires. Both halves are pinned here.

The negative test is the load-bearing one. A safeguard that never fires is indistinguishable
from no safeguard at all, and this project has already been bitten twice by exactly that shape
of defect (the stale test count that sat wrong across three phases, and the "figuras
exportadas" prose claim that passed by accident on a boundary rule). So
`assert_within_tolerance` is tested for what it does when a metric is WRONG, not only when it
is right.
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.evaluation.full_comparison import (
    DEFAULT_RTOL,
    ENV_DRIFT_CEILING,
    assert_within_tolerance,
    load_published,
    verify_against_published,
)
from src.ingestion.unify import integrity_report, unify

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "07_flujo_completo.ipynb"


def test_integrity_report_confirms_every_invariant_on_the_real_dataset():
    """The four invariants must hold on the actual unified data, not just on a fixture."""
    report = integrity_report(unify(save=False))

    assert list(report.columns) == ["comprobacion", "esperado", "observado", "veredicto"]
    assert (report["veredicto"] == "OK").all(), report.to_string(index=False)

    # The operator-sum identity is the one invariant with no guard of its own upstream, and
    # it is the reason same-day operator columns are perfect leakage. Pin it by name so a
    # future refactor cannot quietly drop the row and still show an all-OK report.
    checks = set(report["comprobacion"])
    assert "suma de operadores == total" in checks
    assert {"huecos de calendario", "fechas duplicadas", "valores nulos"} <= checks


def test_verification_passes_when_the_published_table_is_compared_with_itself():
    """The identity case: published vs published must be exactly zero drift, not merely close."""
    published = load_published()
    comparison = verify_against_published(published.copy(), published)

    # 22 published rows x (4 metrics + n).
    assert len(comparison) == len(published) * 5
    assert comparison["desviacion_rel"].max() == 0.0

    message = assert_within_tolerance(comparison, rtol=DEFAULT_RTOL)
    assert message.startswith("OK")
    assert str(len(comparison)) in message


def test_verification_raises_and_interprets_the_magnitude_of_a_perturbed_metric():
    """The safeguard must fire, and must tell the reader which KIND of failure this is.

    A reader who hits a red cell on their own machine concludes "this work does not
    reproduce". For a deviation at float-reassociation scale that conclusion is wrong, so the
    message is required to say so -- and to say the opposite when the deviation is large
    enough to be a genuine pipeline change.
    """
    published = load_published()

    # Below ENV_DRIFT_CEILING: over tolerance, but the signature of a different machine.
    entorno = published.copy()
    entorno.loc[entorno.index[0], "MAE"] *= 1 + (ENV_DRIFT_CEILING / 100)
    with pytest.raises(AssertionError) as env_exc:
        assert_within_tolerance(verify_against_published(entorno, published))
    env_message = str(env_exc.value)
    assert "ENTORNO DE EJECUCION" in env_message
    assert "NO indica que el pipeline haya cambiado" in env_message
    # The evidence must be in the message, not only the verdict.
    assert "PEOR CELDA" in env_message and "MAE" in env_message

    # Above ENV_DRIFT_CEILING: a real change, and it must NOT be excused as an environment
    # difference. This is the assertion that stops the interpretation from being a blanket
    # reassurance that swallows genuine breakage.
    real = published.copy()
    real.loc[real.index[0], "MAE"] *= 1.4
    with pytest.raises(AssertionError) as real_exc:
        assert_within_tolerance(verify_against_published(real, published))
    real_message = str(real_exc.value)
    assert "SI indica un cambio" in real_message
    assert "ENTORNO DE EJECUCION" not in real_message

    # A row that is missing entirely is a structural defect no tolerance can express.
    with pytest.raises(ValueError, match="does not cover the same"):
        verify_against_published(published.iloc[1:].copy(), published)


@pytest.mark.slow
def test_notebook_executes_clean():
    """Execute the notebook end to end and require zero cell errors.

    Because the notebook raises on any metric that drifts from the published table, "it ran"
    IS the reproducibility check -- this test is what keeps that check from rotting silently.

    Executed into a temporary output so the committed notebook keeps the outputs it was
    published with; nbconvert is invoked without --inplace for exactly that reason.
    """
    assert NOTEBOOK.exists(), f"missing {NOTEBOOK}"

    result = subprocess.run(
        [
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "notebook", "--execute",
            "--ExecutePreprocessor.timeout=900",
            "--output-dir", str(ROOT / "reports" / ".nbconvert_check"),
            str(NOTEBOOK),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "el cuaderno no se ejecuto limpio:\n"
        f"{result.stderr[-4000:]}"
    )


def test_committed_notebook_carries_its_outputs():
    """The supervisor will READ this notebook on GitHub, not run it.

    A notebook committed with cleared outputs renders as code with no results, which defeats
    its purpose as a deliverable. This pins the outputs into the artifact contract.
    """
    nbformat = pytest.importorskip("nbformat")
    nb = nbformat.read(NOTEBOOK, as_version=4)

    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    assert code_cells, "el cuaderno no tiene celdas de codigo"

    sin_salida = [i for i, c in enumerate(code_cells) if not c.get("outputs")]
    assert not sin_salida, f"celdas de codigo sin salida guardada: {sin_salida}"

    errores = [
        o for c in code_cells for o in c.get("outputs", []) if o.get("output_type") == "error"
    ]
    assert not errores, f"el cuaderno guardado contiene {len(errores)} salida(s) de error"

    # The verification line is the notebook's whole point; if it is not in the saved output,
    # the reader has no evidence the run actually verified anything.
    texto = "\n".join(
        o.get("text", "")
        for c in code_cells
        for o in c.get("outputs", [])
        if o.get("output_type") == "stream"
    )
    assert "valores coinciden con full_comparison.parquet" in texto

    figuras = sum(
        1
        for c in code_cells
        for o in c.get("outputs", [])
        if "application/vnd.plotly.v1+json" in (o.get("data") or {})
    )
    assert figuras >= 10, f"solo {figuras} figuras embebidas; se esperaban al menos 10"
