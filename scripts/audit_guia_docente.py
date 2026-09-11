"""Mide la memoria compilada contra la guia docente 14MBID (ed. octubre 2025-2026).

Instrumento de verificacion, no de generacion: no escribe nada en el proyecto. Existe
porque los requisitos formales de la guia (extension, recuento de palabras, tipografia,
margenes) solo son comprobables sobre el PDF compuesto, y porque la incidencia 12f de
docs/LaTeX/REPORT_MAPPING.md ya establecio que el contador de `Overfull \\hbox` de LaTeX no
acredita el cumplimiento de margenes: una celda p{} cuyo contenido excede su ancho
declarado desborda en silencio. El criterio vigente es la medicion directa del bbox de
cada linea contra la caja de texto, y aqui vive.

    python scripts/audit_guia_docente.py            # informe completo
    python scripts/audit_guia_docente.py --strict   # ademas, codigo de salida != 0 si
                                                    # incumple un umbral bloqueante

Los umbrales bloqueantes son los que dependen solo de nosotros: tipografia, margenes y
extension del cuerpo. El recuento de palabras se REPORTA bajo las cinco convenciones
posibles pero no bloquea: cual de ellas rige es una decision de direccion (ver
docs/CONSULTA_DIRECCION.md), no un hecho medible.
"""
from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
LATEX = ROOT / "docs" / "LaTeX"
SECTIONS = LATEX / "sections"
PDF = LATEX / "TFT.pdf"

CM = 28.3464567  # pt por cm

# --- exigencias de la guia docente (seccion 4, "Estructura formal") -------------------
REQ_FONT_PT = 11.0
REQ_LINESPREAD = 1.15
REQ_MARGIN_CM = {"top": 4.4, "bottom": 2.19, "left": 3.0, "right": 3.0}
REQ_PAGES = (40, 80)
REQ_WORDS = 30_000
REQ_ABSTRACT_WORDS = (400, 600)
REQ_KEYWORDS = (4, 6)

# La cota de fuente se compara con tolerancia: la opcion de clase 11pt de LaTeX compone a
# 10.95pt, no a 11.00pt exactos, y esa diferencia de 0.45 % no es un incumplimiento.
FONT_TOL = 0.1
SPREAD_TOL = 0.02
MARGIN_TOL_CM = 0.05


# =====================================================================================
# Geometria declarada
# =====================================================================================
def declared_geometry() -> dict[str, float]:
    """Margenes en cm, leidos de la llamada a geometry del preambulo.

    Se lee la fuente en lugar de inferirla del PDF porque el PDF solo revela donde cae el
    contenido, no donde esta la caja: una pagina cuyo texto no llega al margen inferior es
    indistinguible de una con el margen mal puesto.
    """
    src = (LATEX / "preamble.tex").read_text(encoding="utf-8")
    m = re.search(r"\\usepackage\[([^\]]*)\]\{geometry\}", src, re.S)
    if not m:
        raise SystemExit("no encuentro la llamada a geometry en preamble.tex")
    out = {}
    for key, val in re.findall(r"(\w+)\s*=\s*([\d.]+)\s*cm", m.group(1)):
        out[key] = float(val)
    return out


def declared_class_size() -> float:
    src = (LATEX / "TFT.tex").read_text(encoding="utf-8")
    m = re.search(r"\\documentclass\[([^\]]*)\]", src)
    opts = m.group(1) if m else ""
    m2 = re.search(r"(\d+)pt", opts)
    return float(m2.group(1)) if m2 else 10.0


def declared_linespread() -> float | None:
    src = (LATEX / "preamble.tex").read_text(encoding="utf-8")
    for pat in (r"\\setstretch\{([\d.]+)\}", r"\\linespread\{([\d.]+)\}"):
        m = re.search(pat, src)
        if m:
            return float(m.group(1))
    return None


# =====================================================================================
# Estructura del PDF
# =====================================================================================
def open_pdf() -> pymupdf.Document:
    if not PDF.exists():
        raise SystemExit(f"no existe {PDF}; compila primero con latexmk")
    return pymupdf.open(PDF)


def find_structure(doc: pymupdf.Document) -> dict[str, range]:
    """Localiza los tramos del documento por su primera linea de texto.

    Los limites se detectan por contenido y no se codifican como numeros de pagina: el
    proposito de este script es medir despues de un reflujo, momento en el que cualquier
    numero de pagina fijado a mano seria falso.
    """
    firsts = []
    for i in range(doc.page_count):
        words = doc[i].get_text().split()
        firsts.append(" ".join(words[:6]))

    def first_page_matching(pattern: str, start: int = 0) -> int | None:
        rx = re.compile(pattern, re.I)
        for i in range(start, doc.page_count):
            if rx.match(firsts[i]):
                return i
        return None

    toc = first_page_matching(r"[ÍI]ndice general")
    resumen = first_page_matching(r"Resumen\b")
    body = first_page_matching(r"1\.?\s")
    if body is None:  # el titulo del capitulo 1 puede partirse; busca el encabezado
        body = first_page_matching(r"1\.\s*Introducci")
    annex = first_page_matching(r"A\.\s")
    # "Referencias" desde la pasada APA 7; "Bibliografia" se conserva para PDFs anteriores.
    biblio = first_page_matching(r"(Referencias|Bibliograf[ií]a)")

    if None in (resumen, body, annex, biblio):
        raise SystemExit(
            "no consigo delimitar la estructura del PDF "
            f"(resumen={resumen} cuerpo={body} anexos={annex} biblio={biblio})"
        )
    return {
        "portada": range(0, toc if toc is not None else resumen),
        "indices": range(toc, resumen) if toc is not None else range(0, 0),
        "preliminares": range(resumen, body),
        "cuerpo": range(body, annex),
        "anexos": range(annex, biblio),
        "bibliografia": range(biblio, doc.page_count),
    }


def footer_y(doc: pymupdf.Document, body: range) -> float:
    """Umbral por encima del cual empieza el pie de pagina repetido.

    Se deduce del propio pie ("N de M | <titulo>"), no de una constante: con
    bottom=2.19cm el pie sube respecto a la geometria anterior y una constante fijada a
    mano dejaria de separarlo del ultimo parrafo.
    """
    tops = []
    for i in body:
        for blk in doc[i].get_text("blocks"):
            if re.search(r"\b\d+\s+de\s+\d+\s*\|", " ".join(blk[4].split())):
                tops.append(blk[1])
    if not tops:
        return doc[body.start].rect.height  # sin pie detectable: no se recorta nada
    return min(tops) - 2.0


# =====================================================================================
# Recuento de palabras
# =====================================================================================
def words_in_range(doc, rng: range, fy: float) -> int:
    total = 0
    for i in rng:
        chunks = [b[4] for b in doc[i].get_text("blocks") if b[1] < fy]
        total += len(" ".join(chunks).replace("-\n", "").split())
    return total


def size_split(doc, rng: range, fy: float, cut: float) -> tuple[int, int]:
    """(palabras a >= cut pt, palabras a < cut pt), clasificando BLOQUE a bloque.

    Separa prosa de contenido tabular y pies sin reconstruir los flotantes: en este
    documento los cuadros se componen a 9.3-11.1 pt y los pies a \\small, ambos por debajo
    del cuerpo.

    La clasificacion es por bloque, no por span, y eso importa: contar palabras span a span
    parte una palabra en dos cuando cambia el estilo a mitad, e infla el recuento ~3.7 %.
    Como esta cifra se RESTA de un total medido por bloque (convencion C4), mezclar las dos
    granularidades restaria de mas.
    """
    big = small = 0
    for i in rng:
        for blk in doc[i].get_text("dict")["blocks"]:
            if blk.get("type") != 0 or blk["bbox"][1] >= fy:
                continue
            sizes = Counter()
            words = 0
            for line in blk["lines"]:
                for sp in line["spans"]:
                    sizes[round(sp["size"], 1)] += len(sp["text"])
                words += len("".join(sp["text"] for sp in line["spans"]).split())
            dom = sizes.most_common(1)[0][0] if sizes else 0
            if dom >= cut:
                big += words
            else:
                small += words
    return big, small


# Sin "description": la convencion C5 se define por reproducibilidad contra la pasada de
# LanguageTool, y aquel detex no lo descartaba. Anadirlo mejoraria el filtro pero cambiaria
# la cifra que se ha comunicado a direccion, que es justo lo que C5 sirve para fijar.
DROP_ENVS = (
    "tabular", "table", "figure", "equation", "align", "ganttchart",
    "itemize", "enumerate",
)
BS = "\\"
DROP_WITH_ARG = re.compile(
    BS + BS + r"(?:texttt|label|ref|autoref|pageref|cite|textcite|parencite|includegraphics|input|resizebox|"
    r"setlength|arraystretch|vspace|hspace|rowcolor|definecolor|caption|keywords|"
    r"url|href|ganttbar|colorbox)\s*(?:\[[^\]]*\])?\{(?:[^{}]|\{[^{}]*\})*\}"
)


def _strip(text: str) -> str:
    text = re.sub(r"(?<!" + BS + BS + r")%.*$", "", text)
    text = re.sub(r"\$[^$]*\$", " NUM ", text)
    text = DROP_WITH_ARG.sub(" ", text)
    for _ in range(6):
        new = re.sub(BS + BS + r"(?:textbf|textit|emph|textsl|underline)\{([^{}]*)\}", r"\1", text)
        if new == text:
            break
        text = new
    text = re.sub(BS + BS + r"begin\{[^}]*\}(?:\[[^\]]*\])?", " ", text)
    text = re.sub(BS + BS + r"end\{[^}]*\}", " ", text)
    text = re.sub(BS + BS + r"[a-zA-Z@]+\s*(?:\[[^\]]*\])?", " ", text)
    text = re.sub(r"[{}&" + BS + BS + r"~^_]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def detex_words() -> int:
    """Convencion de la pasada de LanguageTool: prosa corrida de sections/*.tex.

    Reimplementada aqui, y no importada, porque el detex original vivia en un scratchpad
    de sesion. Verificado contra su salida: 28.642 palabras sobre el estado del 09/09/2026.
    """
    rx_begin = re.compile(BS + BS + r"begin\{(" + "|".join(DROP_ENVS) + r")\}")
    rx_end = re.compile(BS + BS + r"end\{(" + "|".join(DROP_ENVS) + r")\}")
    total = 0
    for path in sorted(SECTIONS.glob("*.tex")):
        depth = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if rx_begin.search(line):
                depth += 1
            inside = depth > 0
            if rx_end.search(line):
                depth = max(0, depth - 1)
            if inside:
                continue
            s = _strip(line)
            if len(s) >= 3:
                total += len(s.split())
    return total


# =====================================================================================
# Reparto vertical y margenes
# =====================================================================================
def _union_len(intervals: list[tuple[float, float]]) -> float:
    merged: list[list[float]] = []
    for a, b in sorted(intervals):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return sum(b - a for a, b in merged)


def vertical_split(doc, body: range, box: tuple[float, float], fy: float, cut: float):
    """Paginas-equivalente de figura, cuadro y prosa en el cuerpo.

    Por union de intervalos y no por suma de alturas: las columnas de un mismo cuadro son
    bloques distintos con rangos y solapados, y sumarlas triplica el resultado.
    """
    top, bot = box
    img_t = small_t = prose_t = 0.0
    for i in body:
        img, small, prose = [], [], []
        for blk in doc[i].get_text("dict")["blocks"]:
            y0, y1 = max(blk["bbox"][1], top), min(blk["bbox"][3], bot)
            if blk["bbox"][1] >= fy or y1 <= y0:
                continue
            if blk.get("type") == 1:
                img.append((y0, y1))
                continue
            sizes = Counter()
            for line in blk["lines"]:
                for sp in line["spans"]:
                    sizes[round(sp["size"], 1)] += len(sp["text"])
            dom = sizes.most_common(1)[0][0] if sizes else 0
            (small if dom < cut else prose).append((y0, y1))
        a = _union_len(img)
        b = _union_len(img + small)
        c = _union_len(img + small + prose)
        img_t += a
        small_t += b - a
        prose_t += c - b
    return img_t, small_t, prose_t


def margin_violations(doc, box: tuple[float, float, float, float], skip: range):
    """Lineas cuyo bbox sale de la caja de texto. Criterio vigente (incidencia 12f)."""
    x0, x1, y0, y1 = box
    bad = []
    for i in range(doc.page_count):
        if i in skip:
            continue
        for blk in doc[i].get_text("dict")["blocks"]:
            if blk.get("type") != 0:
                continue
            for line in blk["lines"]:
                lx0, ly0, lx1, ly1 = line["bbox"]
                over = max(lx0 and (x0 - lx0) or 0.0, lx1 - x1)
                if over > 0.5:
                    txt = "".join(sp["text"] for sp in line["spans"])
                    bad.append((i + 1, round(over, 1), txt.strip()[:70]))
    return bad


def measured_typography(doc, body: range, fy: float):
    """Cuerpo de letra, familia e interlineado tal como estan compuestos."""
    sizes, fonts = Counter(), Counter()
    ys = defaultdict(list)
    for i in body:
        for blk in doc[i].get_text("dict")["blocks"]:
            if blk.get("type") != 0 or blk["bbox"][1] >= fy:
                continue
            for line in blk["lines"]:
                for sp in line["spans"]:
                    n = len(sp["text"])
                    sizes[round(sp["size"], 2)] += n
                    fonts[sp["font"]] += n
                ys[i].append(round(line["bbox"][1], 2))
    body_pt = sizes.most_common(1)[0][0]
    deltas = Counter()
    for v in ys.values():
        col = sorted(set(v))
        for a, b in zip(col, col[1:]):
            d = round(b - a, 1)
            if body_pt * 0.9 < d < body_pt * 1.8:
                deltas[d] += 1
    baseline = deltas.most_common(1)[0][0] if deltas else 0.0
    return body_pt, fonts.most_common(1)[0][0], baseline


# Linea base "sencilla" que la clase da a cada opcion de cuerpo. El interlineado que pide
# la guia se mide contra ESTO, no contra el cuerpo de letra: "interlineado 1,15" significa
# 1,15 veces el sencillo, y el sencillo ya vale ~1,2 veces el cuerpo en cualquier
# compositor. Dividir la linea base entre el cuerpo daria 1,43 y haria fallar una
# composicion correcta.
NATURAL_BASELINE = {10.0: 12.0, 11.0: 13.6, 12.0: 14.5}


# =====================================================================================
# Comprobaciones sobre la fuente
# =====================================================================================
def source_checks() -> dict[str, object]:
    joined = "\n".join(p.read_text(encoding="utf-8") for p in SECTIONS.glob("*.tex"))
    anexos = (SECTIONS / "07_anexos.tex").read_text(encoding="utf-8")
    return {
        "ods": len(re.findall(r"\bODS\b|Desarrollo Sostenible|Agenda 2030", joined)),
        "github": len(re.findall(r"github\.com", anexos)),
    }


def abstract_metrics(doc, prelim: range, fy: float) -> dict[str, tuple[int, int, int]]:
    """(palabras, palabras clave, paginas) del Resumen y del Abstract.

    Las paginas importan: la guia pide "no mas de una pagina" para cada uno, y ese es el
    requisito que primero se rompe al ampliar el texto hasta el minimo de 400 palabras.
    """
    starts = {}
    for i in prelim:
        head = " ".join(doc[i].get_text().split()[:1])
        if head in ("Resumen", "Abstract", "Agradecimientos"):
            starts[i] = head
    span = {}
    keys = sorted(starts)
    for n, i in enumerate(keys):
        end = keys[n + 1] if n + 1 < len(keys) else prelim.stop
        span[starts[i]] = end - i

    out = {}
    for i in prelim:
        text = " ".join(b[4] for b in doc[i].get_text("blocks") if b[1] < fy)
        head = "Resumen" if text.lstrip().startswith("Resumen") else (
            "Abstract" if text.lstrip().startswith("Abstract") else None)
        if head is None:
            continue
        body_txt = re.sub(r"^\s*(Resumen|Abstract)\s*", "", text)
        parts = re.split(r"Palabras clave:|Keywords:", body_txt)
        nkw = len([x for x in parts[1].split(",") if x.strip()]) if len(parts) > 1 else 0
        out[head] = (len(parts[0].split()), nkw, span.get(head, 1))
    return out


# =====================================================================================
def main() -> int:
    strict = "--strict" in sys.argv
    doc = open_pdf()
    st = find_structure(doc)
    body = st["cuerpo"]
    fy = footer_y(doc, body)

    geom = declared_geometry()
    page_w, page_h = doc[body.start].rect.width, doc[body.start].rect.height
    box = (
        geom["left"] * CM,
        page_w - geom["right"] * CM,
        geom["top"] * CM,
        page_h - geom["bottom"] * CM,
    )
    body_pt, font, baseline = measured_typography(doc, body, fy)
    # El corte prosa/cuadro se DERIVA del cuerpo medido; fijarlo a mano (11,5pt valia con
    # la clase de 12pt) lo invalida en cuanto cambia el cuerpo, y entonces el clasificador
    # declara toda la prosa como tabla sin protestar.
    cut = round(body_pt * 0.96, 2)

    natural = NATURAL_BASELINE.get(declared_class_size(), body_pt * 1.2)
    spread = baseline / natural if natural else 0.0

    print("=" * 78)
    print(f"AUDITORIA GUIA DOCENTE 14MBID  --  {PDF.relative_to(ROOT)}, {doc.page_count} paginas")
    print("=" * 78)

    print("\n[C] FORMATO FORMAL")
    ok_font = abs(body_pt - REQ_FONT_PT) <= REQ_FONT_PT * 0.02
    ok_family = any(k in font.lower() for k in ("helvetica", "arial", "nimbus"))
    ok_spread = abs(spread - REQ_LINESPREAD) <= SPREAD_TOL
    print(f"  familia        exigida Arial/Helvetica | medida {font:<28} {'OK' if ok_family else 'FALLA'}")
    print(f"  cuerpo         exigido {REQ_FONT_PT:5.2f} pt      | medido {body_pt:5.2f} pt "
          f"(clase {declared_class_size():.0f}pt){'':7} {'OK' if ok_font else 'FALLA'}")
    print(f"  interlineado   exigido {REQ_LINESPREAD:5.2f}         | medido {spread:5.3f} "
          f"= {baseline:5.2f} pt / {natural:4.1f} pt sencillo  {'OK' if ok_spread else 'FALLA'}")
    print(f"  (corte prosa/cuadro derivado del cuerpo medido: {cut} pt)")
    ok_margins = True
    for key, req in REQ_MARGIN_CM.items():
        got = geom.get(key, float("nan"))
        good = abs(got - req) <= MARGIN_TOL_CM
        ok_margins &= good
        print(f"  margen {key:<8}exigido {req:5.2f} cm      | declarado {got:5.2f} cm{'':17} "
              f"{'OK' if good else 'FALLA'}")

    viol = margin_violations(doc, box, st["portada"])
    print(f"\n  infracciones de margen (bbox de linea vs caja de texto, "
          f"criterio incidencia 12f): {len(viol)}")
    for pg, over, txt in viol[:12]:
        print(f"    p.{pg:<4} +{over:5.1f} pt  {txt}")
    if len(viol) > 12:
        print(f"    ... y {len(viol) - 12} mas")

    print("\n[A] EXTENSION DEL CUERPO")
    n_body = len(body)
    top, bot = box[2], box[3]
    img, small, prose = vertical_split(doc, body, (top, bot), fy, cut)
    ph = bot - top
    print(f"  paginas fisicas del cuerpo (cap. 1-6): {n_body}   "
          f"{'OK' if REQ_PAGES[0] <= n_body <= REQ_PAGES[1] else 'FALLA'}  "
          f"(limite {REQ_PAGES[0]}-{REQ_PAGES[1]})")
    print(f"    figuras                {img / ph:6.2f} pag")
    print(f"    cuadros y pies         {small / ph:6.2f} pag")
    print(f"    prosa y titulos        {prose / ph:6.2f} pag")
    print(f"    blanco de composicion  {(n_body * ph - img - small - prose) / ph:6.2f} pag")
    print(f"  neto de figuras y cuadros: {n_body - (img + small) / ph:6.2f} pag")
    print(f"  solo prosa:                {prose / ph:6.2f} pag   "
          f"(suelo de {REQ_PAGES[0]}: {'OK' if prose / ph >= REQ_PAGES[0] else 'FALLA'})")

    print("\n[B] RECUENTO DE PALABRAS  (informativo: la convencion la fija direccion)")
    w = {k: words_in_range(doc, r, fy) for k, r in st.items()}
    for k in ("portada", "indices", "preliminares", "cuerpo", "anexos", "bibliografia"):
        print(f"    {k:<15}{w[k]:7d}")
    c1 = sum(w.values())
    c2 = c1 - w["indices"]
    c3 = c2 - w["bibliografia"]
    tab = sum(size_split(doc, st[k], fy, cut)[1] for k in ("preliminares", "cuerpo", "anexos"))
    c4 = c3 - tab
    c5 = detex_words()
    for name, val, desc in (
        ("C1", c1, "todo lo compuesto, sin el pie repetido"),
        ("C2", c2, "C1 sin indices auto-generados"),
        ("C3", c3, "C2 sin bibliografia"),
        ("C4", c4, "C3 sin interior de cuadros ni pies"),
        ("C5", c5, "prosa corrida de sections/*.tex (convencion detex)"),
    ):
        print(f"    {name}  {val:7d}  {'OK   ' if val <= REQ_WORDS else 'EXCEDE'}  "
              f"{val - REQ_WORDS:+7d}   {desc}")

    print("\n[D] RESUMEN Y ABSTRACT")
    ok_abs = True
    for name, (nw, nkw, npg) in abstract_metrics(doc, st["preliminares"], fy).items():
        good_w = REQ_ABSTRACT_WORDS[0] <= nw <= REQ_ABSTRACT_WORDS[1]
        good_k = REQ_KEYWORDS[0] <= nkw <= REQ_KEYWORDS[1]
        good_p = npg <= 1
        ok_abs &= good_w and good_k and good_p
        print(f"    {name:<10}{nw:4d} palabras {'OK' if good_w else 'FALLA'}   "
              f"{nkw} palabras clave {'OK' if good_k else 'FALLA'}   "
              f"{npg} pagina {'OK' if good_p else 'FALLA'}")

    print("\n[G] [F] ELEMENTOS REQUERIDOS EN LA FUENTE")
    src = source_checks()
    print(f"    menciones a los ODS en sections/     {src['ods']:3d}  "
          f"{'OK' if src['ods'] else 'FALLA'}")
    print(f"    URL de github en 07_anexos.tex       {src['github']:3d}  "
          f"{'OK' if src['github'] else 'FALLA'}")

    blocking = {
        "tipografia": ok_font and ok_family and ok_spread,
        "margenes declarados": ok_margins,
        "margenes medidos": not viol,
        "extension del cuerpo": REQ_PAGES[0] <= n_body <= REQ_PAGES[1],
        "resumen/abstract": ok_abs,
        "ODS presente": bool(src["ods"]),
        "URL del repositorio": bool(src["github"]),
    }
    print("\n" + "=" * 78)
    failed = [k for k, v in blocking.items() if not v]
    print("BLOQUEANTES: " + ("todos OK" if not failed else "FALLAN -> " + ", ".join(failed)))
    print("=" * 78)
    return 1 if (strict and failed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
