"""Traslada los once cuadros diagnosticos anchos de las secciones 5.9-5.11 a un Anexo E.

Motivo de cumplimiento, no solo de extension. La guia docente (14MBID, ed. 2025-26,
seccion 4) dice: "No se aconseja la utilizacion de tablas extensas en el cuerpo principal.
En caso de ser necesarias se pueden incorporar como anexo." Los once cuadros que mueve este
script tienen 8 o 9 columnas y solo caben en el cuerpo escalados con \\resizebox, a 9,3-10,9
pt de fuente efectiva, por debajo del cuerpo del documento. Son exactamente el caso que la
guia desaconseja. Que ademas liberen paginas del cuerpo es una consecuencia, no el motivo.

Cada cuadro se lleva su \\label, de modo que todo \\ref del capitulo 5 sigue resolviendo:
apunta al anexo en vez de a la pagina de enfrente. Ninguna cifra queda sin citar y no se
toca una sola linea de prosa; las frases remiten a los cuadros por numero, nunca por
posicion ("como muestra el cuadro 5.13"), que es la razon por la que el traslado es seguro.

    python scripts/prepare_anexo_e.py            # simulacro: dice que movera
    python scripts/prepare_anexo_e.py --apply    # escribe el cambio

Despues, siempre:
    cd docs/LaTeX && latexmk -pdf -interaction=nonstopmode -halt-on-error TFT.tex
    grep -ac "LaTeX Warning: Reference" TFT.log       # debe ser 0
    python scripts/audit_guia_docente.py              # 0 infracciones de margen
    python -m pytest                                  # 470/470
"""
from __future__ import annotations

import io
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SECTIONS = ROOT / "docs" / "LaTeX" / "sections"
BODY = SECTIONS / "05_resultados.tex"
ANNEX = SECTIONS / "07_anexos.tex"

# El orden de esta lista es el orden en que apareceran en el anexo.
LABELS = [
    "tab:autocorrelacion_residuo",
    "tab:residuo_dia_semana",
    "tab:residuo_calendario",
    "tab:residuo_meteo",
    "tab:cadena_error",
    "tab:curva_aprendizaje",
    "tab:ablacion_bloques",
    "tab:paridad_lstm",
    "tab:subgrupos_test",
    "tab:subgrupos_val",
    "tab:subgrupos_inferencia",
]

# cap:anexo5, no cap:anexo4: el Anexo D (declaracion de uso de IA) ya ocupa cap:anexo4 en
# 07_anexos.tex. El borrador de este script declaraba cap:anexo4 y habria producido una
# etiqueta duplicada, que LaTeX resuelve en silencio a favor de la ultima y habria hecho
# que las referencias al Anexo D apuntasen aqui.
ANNEX_HEAD = r"""

\chapter{Cuadros diagnósticos extendidos}\label{cap:anexo5}

Este anexo recoge los cuadros de detalle de las secciones \ref{sec:anatomia_residuo_etapa1},
\ref{sec:paridad_informativa} y \ref{sec:desglose_subgrupos}. El argumento, las cifras
titulares y los veredictos preregistrados permanecen íntegros en el cuerpo del capítulo
\ref{cap:resultados}; aquí solo se traslada el detalle tabular, al que el texto sigue
remitiendo por número de cuadro.
"""


def extract(text: str, label: str) -> tuple[str | None, str]:
    """Devuelve (bloque, texto_sin_bloque) del flotante table que lleva `label`."""
    for m in re.finditer(r"\\begin\{table\}.*?\\end\{table\}\n?", text, re.S):
        if "\\label{%s}" % label in m.group(0):
            return m.group(0), text[: m.start()] + text[m.end():]
    return None, text


def main() -> int:
    apply = "--apply" in sys.argv
    body = io.open(BODY, encoding="utf-8").read()

    blocks, missing = [], []
    for lab in LABELS:
        blk, body = extract(body, lab)
        if blk is None:
            missing.append(lab)
        else:
            blocks.append((lab, blk))
    if missing:
        print("NO ENCONTRADOS (abortando):", ", ".join(missing))
        return 1

    annex_src = io.open(ANNEX, encoding="utf-8").read()
    if "cap:anexo5" in annex_src:
        print("07_anexos.tex ya contiene cap:anexo5; el traslado ya se aplico. Nada que hacer.")
        return 0

    body = re.sub(r"\n{3,}", "\n\n", body)
    annex = annex_src.rstrip() + "\n" + ANNEX_HEAD + "\n" + "\n".join(b for _, b in blocks)

    print("cuadros trasladados: %d" % len(blocks))
    for lab, blk in blocks:
        print("  %-32s %4d lineas" % (lab, blk.count("\n")))
    if not apply:
        print("\nsimulacro -- no se ha escrito nada. Reejecuta con --apply.")
        return 0

    io.open(BODY, "w", encoding="utf-8").write(body)
    io.open(ANNEX, "w", encoding="utf-8").write(annex)
    print("\nescrito. Recompila y comprueba: 0 referencias indefinidas, 470/470 tests.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
