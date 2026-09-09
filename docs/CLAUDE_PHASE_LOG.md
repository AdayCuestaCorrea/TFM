# TFM phase log (read on demand)

Extracted from `CLAUDE.md` so Claude Code does not load this at session start.
Do not `@`-import this file from `CLAUDE.md`.
Read it only when a task needs forensic history of a completed phase.

The spec for the current phase is the plan the user attached — not this log.

---

## Current Status

**Phase 0 — Scaffolding: complete.** Folder structure, raw files copied, venv + pinned
dependencies installed and import-verified, this document written.

**Phase 1 — Ingestion: complete.** 30/30 tests passing.

**Phase 2 — Feature engineering: complete.** 66/66 tests passing (36 new).

**Phase 3 — Splits, weather promotion, baselines: complete.** 111/111 tests passing
(45 new).

**Phase 4 — LSTM Stage 1: complete.** 138/138 tests passing (27 new).

**Phase 5 — Hybrid residual model: complete.** 160/160 tests passing (22 new). All models
built and compared. **HEADLINE RESULT: the hybrid does NOT beat SARIMAX or XGBoost-alone
on either val or test.** See the Phase 5 entry — this is the thesis's central empirical
finding and it is negative for the hybrid architecture.

**Phase 5b — Sensitivity analysis: complete.** 177/177 tests passing (17 new). Two
bounded, pre-registered experiments. **Phase 5's headline conclusion STILL STANDS.** A
simple SARIMAX+XGBoost ensemble is now the best model overall (test MAE 139,681).

**Phase 6 — Results dashboard: complete.** 217/217 tests passing (40 new). Plotly +
ipywidgets notebook, 13 static figures exported. **No model was trained or re-predicted;
every figure is drawn from Phase 3-5b artifacts.**

**Phase 6b — Visual refinement: complete.** 257/257 tests passing (40 new). VIU
brand-approximate theme shared by the interactive and static paths, plus copy-pasteable
tables and captions for the memoria. **Presentation only — no data, metric, or model logic
was touched.**

**Phase 7 — LaTeX modularization (scaffolding): complete.** `docs/LaTeX/TFT.tex` split into
`preamble.tex` + eight `sections/*.tex` chapter files (`00_preliminares` ...
`07_anexos`) plus a `tables/` directory, all `\input`-ed from a thin `TFT.tex`
orchestrator. `latexmk -pdf -halt-on-error` compiles clean. No content was added yet in
this phase — pure restructuring of the existing skeleton so Phases 8-11 have a stable
target to write into.

**Phase 8 — Memoria figures: complete.** 21/21 new tests passing (`test_memoria_figures.py`).
`src/evaluation/memoria_figures.py` adds 9 builders — `total_series_figure`,
`operator_demand_figure`, `weekly_seasonality_figure`, `annual_seasonality_figure`,
`weather_vs_demand_figure`, `splits_figure`, `oof_folds_figure`,
`oof_fold1_degenerate_figure`, `error_dispersion_figure` — covering raw-data EDA, the
chronological split boundaries, and the Phase 4 OOF diagnostics (including the fold-1
degeneracy) that the Phase 6/6b dashboard never needed to plot. All nine reuse
`VIU_TEMPLATE` from `theme.py` and read only from `data/processed/` — no model retrained,
no new number computed outside what Phases 3-5b already persisted.
`export_figures.py` now writes **22 PNGs** total (13 inherited from Phase 6b + 9 new) to
`reports/figures/`, still as a standalone script (kaleido 1.x deadlocks inside a Jupyter
kernel — Phase 6 finding, still true here).

**Phase 9 — Automatic LaTeX tables: complete.** 27/27 new tests passing
(`test_export_latex_tables.py`). `src/evaluation/export_latex_tables.py` renders 21
`booktabs`-styled `tabular` blocks straight to `docs/LaTeX/tables/*.tex`, sourced from
`report_tables.py` (Phase 6b) plus two memoria-only tables built from static project
metadata: the phase-by-phase cronograma (`tabla_cronograma_fases`, one row per `CLAUDE.md`
phase) and the state-of-the-art comparison (`tabla_comparativa_estado_arte`). Spanish
number formatting (period thousands, comma decimals; MAPE 2 decimals, R² 4 — same
asymmetry as Phase 6b, same reason) and LaTeX escaping are centralized in
`format_number`/`escape_latex` so no chapter file ever formats a number by hand.

**Phase 10 — Chapter skeletons and bibliography: complete.** All eight `sections/*.tex`
files populated with prose outlines wired to the Phase 8/9 artifacts: a `pgfgantt` Gantt
chart plus the cronograma table in the introduction, a `pdflscape` landscape table for the
state-of-the-art comparison, and every results table/figure from Phase 5-5b embedded in
`05_resultados.tex`. `Bibliografia_TFT.bib` updated with technical references; a few
project-specific citations are flagged `PENDIENTE DE VERIFICAR` pending primary-source
confirmation. Two real LaTeX defects were found and fixed here, not worked around: the
`Δ MAE` column (Unicode `U+0394`, unsupported by the document's encoding) renamed to
`Delta MAE` in both the table builder and the surrounding caption, and the Gantt chart
overflowing the page margin (`x unit` reduced, wrapped in `\resizebox`). See
`docs/LaTeX/REPORT_MAPPING.md` §5 for the full incident log.

**Phase 11 — Mapping and closure: complete.** `docs/LaTeX/REPORT_MAPPING.md` documents,
for every one of the 22 figures and 21 tables, the exact chain
source phase → Python builder → `data/processed/` artifact → `.tex` file → chapter — so a
number changing upstream can be traced to the one command that regenerates it, without
re-reading the codebase. **308/308 tests passing overall** (51 new across Phases 8-9).
Final compilation verified: `latexmk -pdf -interaction=nonstopmode -halt-on-error TFT.tex`
from `docs/LaTeX/` produces a 40-page `TFT.pdf`, **0 errors**. Remaining warnings (empty
bibliography until `\cite{}` calls are added during prose writing, a handful of
`Overfull`/`Underfull \hbox`, and an undefined `LastPage` reference resolved by
`latexmk`'s own second pass) are cosmetic and documented, not outstanding defects.

**Post-Phase-11 visual fixes (table overflow, figure titles, landscape page): complete.**
Three rounds of visual QA on the compiled PDF, each fixed at the source rather than by
hand-patching the PDF:

1. **Table overflows (Cuadros 1.1, 3.1, 3.2, 3.3, 4.2, 4.3, 4.4, 5.10).** Fixed in
   `export_latex_tables.py`: explicit `p{}`-width `column_spec` per table plus
   `\allowbreak{}` inserted after `/` and `\_` in `escape_latex`, so long technical
   identifiers and paths wrap instead of overflowing `\textwidth`. `\small` added to the
   surrounding `table` environments in `03_datos_features.tex` / `04_metodologia.tex`.
2. **Embedded figure titles colliding with horizontal legends** (worst on
   `model_comparison_bar_*`, `fig_oof_folds`, `fig_oof_fold1_degenerado`, where the
   top-level Plotly `title` sat directly on top of a `legend(orientation="h", y>1)`).
   Removed the top-level `title=` from every builder in `dashboard_figures.py` and
   `memoria_figures.py` — the descriptive title now lives **only** in the LaTeX
   `\caption`, never duplicated in the PNG. Subplot-level `subplot_titles` (e.g. the
   fold-1-vs-fold-3 panel labels, the four-metric panel labels) are informative content,
   not a duplicate headline, and were kept. The one runtime label that mattered
   (`⚠ IN-SAMPLE`) was moved from the title string to an in-plot annotation on
   `demand_figure` so the warning survives without a competing title. `n_train`/`n_val`/
   `n_test` on `splits_figure` was dropped from the (now-removed) title — `tabla_particiones`
   already carries the exact counts, so nothing was lost; `test_splits_figure_matches_the_
   official_split_boundaries` was rewritten to check the vrect annotations instead.
3. **Empty landscape page before Cuadro 2.1** (`02_estado_arte.tex`). Root cause was
   `\begin{table}[H]` inside `\begin{landscape}`: `[H]` forces exact placement, but
   `\begin{landscape}` already forces its own page break, and when the table doesn't fit
   the float defers to the *next* page anyway — producing a blank rotated page in between
   (a documented `pdflscape`/`float` interaction, not a bug in this project's macros;
   confirmed against community reports before touching the file). Fix: `[H]` → `[p]`.
   Adding a manual `\clearpage` before `\begin{landscape}` was tried first and made it
   worse (42 pages, two blank pages); reverted. Net effect: **41 pages, 0 blank pages**
   (verified page-by-page via a temporary `pymupdf` text-extraction pass, not by eye).

`python -m src.evaluation.export_figures` re-exported all 22 PNGs; `tests/
test_memoria_figures.py` (22, one new) and the full suite (**310/310 passing**, 2 new)
both green. `latexmk -pdf -interaction=nonstopmode -halt-on-error TFT.tex` — **0 errors**,
41-page `TFT.pdf`.

**Phase 12 — Visual QA pass on the compiled PDF: complete.** Table overflows (Cuadros
1.1, 3.1, 3.2, 3.3, 4.2, 4.3, 4.4, 5.10), colliding figure titles/legends, and an empty
landscape page before Cuadro 2.1 were all fixed at the source — see "Post-Phase-11 visual
fixes" above. 310/310 tests pass, `latexmk` compiles clean at 41 pages. No prose was
written in this phase; it validated the skeleton before drafting began.

**Chapter 3 (`03_datos_features.tex`) formal prose: complete.** The `%%` bullet-point
skeleton was replaced with full academic prose across all five sections — data sources
(CRTM, Open-Meteo, working calendar), unification and integrity validation, exploratory
analysis of all six chapter figures (annual/weekly seasonality, weather-vs-demand
dispersion), the mathematical design of the 161-variable feature matrix (lags, shift-first
rolling stats, Fourier harmonics, calendar dummies, the same-day-operator and
same-day-weather leakage guards), and the chronological 70/15/15 split with its
floating-point and warm-up safeguards. Every `\input{tables/...}` call, `\label`, and
`\ref` from the Phase 10 skeleton was preserved unchanged; only the surrounding narrative
was written. One rendering defect was found and fixed: straight double quotes (`"..."`)
inside Spanish-babel prose trigger the `"c`/`"z`-style shorthand ligatures and corrupt
adjacent text (observed as `çategoría basez`) — replaced with `` `` / '' pairs
throughout. `latexmk -pdf -interaction=nonstopmode -halt-on-error TFT.tex` compiles clean
(0 errors, 47 pages); 310/310 tests still pass, since no Python code was touched.
Chapters 1, 2, and 4-7 remain at the Phase 10 bullet-point skeleton stage.

**Chapter 4 (`04_metodologia.tex`) formal prose: complete.** The `%%` bullet-point
skeleton was replaced with full academic prose: the chapter-level hybrid formulation
($\hat y^{\text{final}} = \hat y^{\text{LSTM}} + \hat e^{\text{XGBoost}}$) contrasted
explicitly against Zhang (2003)'s ARIMA-neural hybrid; naive/seasonal-naive baselines and
the SARIMAX(2,1,2)x(0,1,2,7) grid selection (§4.1); the univariate LSTM, its $W=28$
window evidence, and its early-stopping-on-a-training-tail safeguard (§4.2); the
five-fold expanding walk-forward OOF scheme with a full account of the fold-1 degeneracy
(std ratio 0.199, correlation 0.300) and its coverage-checked exclusion — 137 rows, 889 to
552 (§4.3); the residual XGBoost stage, its hyperparameters, and the known OOF
distribution-shift limitation stated up front rather than in the discussion (§4.4); and
the `xgboost_alone` control plus the `hybrid_weighted` sensitivity variant, both justified
methodologically without anticipating their Chapter 5 results (§4.5). This chapter also
introduces the project's first real `\cite{}` calls (`zhang2003hybrid`, `box1970time`,
`hochreiter1997lstm`, `chen2016xgboost`, `bergmeir2012cv`, `cerqueira2020evaluating`,
`wolpert1992stacking`), starting to resolve the empty-bibliography warning noted since
Phase 10. `latexmk -pdf -interaction=nonstopmode -halt-on-error TFT.tex` compiles clean
(0 errors, 53 pages, stable across repeated clean rebuilds); 310/310 tests still pass, no
Python code touched.

**Chapter 5 (`05_resultados.tex`) formal prose: complete — the results chapter.** The
`%%` skeleton was replaced with full academic prose across nine sections: the 11-model
master comparison on val/test (§5.1) with the ensemble leading (test MAE 139,681,
R² 0.9755) ahead of `xgboost_alone` and `sarimax`, both far ahead of the LSTM family and
the collapsed naive baselines; temporal fit and residual distributions (§5.2-5.3)
contrasting `xgboost_alone`'s train/test overfit gap (R² train 0.9929 vs. MAE test
156,121) against the ensemble's narrower, better-centred error; the day-type breakdown
(§5.4) explaining the SARIMAX/XGBoost weekday-weekend complementarity that mechanically
motivates the ensemble (148k vs 169k on ordinary weekdays; 100k vs 169k on Sundays), with
the mandatory small-*n* caveat (festivo n=5, puente n=3 on test); feature importance
(§5.5) contrasting `xgboost_alone`'s calendar dominance (58.5%) against
`xgboost_residual`'s weather dominance (51.6%) while `feat_day_type_festivo` (gain 0.125)
still tops the residual model's individual ranking; the central contrast (§5.6: hybrid
worse than SARIMAX by +70,597 and XGBoost-alone by +78,103 MAE on test) read through
Granger (1989)'s shared-information-set condition and the M4/M5 benchmarks (§5.6.1);
sensitivity analysis (§5.7) documenting Experiment A's unmet pre-registered criterion and
the Experiment B ensemble weights; the error-decorrelation mechanism (§5.8) deriving the
Bates & Granger (1969) variance-reduction identity
$\operatorname{Var}((\hat e_1+\hat e_2)/2)=\sigma^2(1+\rho)/2$ from the measured
$\rho=0.576$ residual correlation, alongside Perrone & Cooper (1993); and explicit,
evidence-cited answers to all 5 research questions from §1.2 (§5.9). New citations
(`granger1989combining`, `bates1969combination`, `perrone1993networks`,
`makridakis2018m4`) bring the bibliography to 11 resolved entries — no longer empty.
`latexmk -pdf -interaction=nonstopmode -halt-on-error TFT.tex` compiles clean (0 errors,
61 pages) after one transient `latexmk`-internal failure self-resolved on an immediate
retry (no source defect — brace balance and quote usage were verified clean); 310/310
tests still pass, no Python code touched.

**Chapter 2 (`02_estado_arte.tex`) formal prose: complete.** The `%%` skeleton was
replaced with full academic prose in an inverted-triangle structure across four sections:
urban mobility and CRTM demand drivers (§2.1, contrasting this project's system-wide,
four-operator aggregation against the single-line/single-station granularity of
`monje2022deep` and `cardozo2012spatial`); classical statistical and tabular ML modelling
(§2.2, `box1970time`'s ARIMA/SARIMAX lineage, `chen2016xgboost`'s gradient boosting, and
`bergmeir2012cv`/`cerqueira2020evaluating`'s empirical case against random CV on time
series as the theoretical basis for this project's walk-forward OOF scheme); hybrid and
ensemble architectures (§2.3, the core theoretical section) — `zhang2003hybrid` as the
direct precedent of the two-stage residual formula, with its three explicit differences
from this TFM spelled out (inverted stage order, exogenous vs. self-lag residual
modelling, and OOF vs. in-sample residuals) rather than diluted into a vague similarity
claim; `wolpert1992stacking` and `granger1989combining`'s information-set condition as
the theoretical basis for the winning ensemble; `bates1969combination`'s variance formula
and `perrone1993networks`; `makridakis2018m4`'s M4/M5 evidence that simple combinations
often beat complex individual architectures; and `surribas_sayago_diffuse` cited strictly
as feature-engineering inspiration, never as a replicated architecture. Section 2.4
walks through all five axes of the landscape `tabla_comparativa_estado_arte` (scope,
horizon, exogenous matrix, validation protocol, empirical contribution) and closes with
an explicit research-gap statement: not the absence of hybrid architectures in the
literature, but the absence of a rigorously OOF-validated audit of *when and why* they
fail against simpler alternatives. One cross-reference bug was caught before final
compilation: a `\ref{sec:features_lag_rolling}` label that does not exist in
`03_datos_features.tex` (the real label is `sec:ingenieria_features`) — fixed before the
undefined-reference warning could reach a final build. `latexmk -pdf -interaction=nonstopmode
-halt-on-error TFT.tex` compiles clean (0 errors, 0 undefined references, 68 pages) after
one transient `latexmk`-internal failure self-resolved on retry, and the landscape table
page was visually verified to have no blank page before or after it. 310/310 tests still
pass, no Python code touched.

**Bibliography cleanup: `monje2022deep`, `toque2017short`, `cardozo2012spatial`
verified.** Their `note = {PENDIENTE DE VERIFICAR ...}` fields were removed and replaced
with clean, complete metadata cross-checked against `docs/references/Summary.md` (the
project's original source brief): full author lists, journal/conference name, volume,
pages, and DOI. `surribas_sayago_diffuse` is the only entry still carrying the pending
note — it stays a placeholder because it is cited exclusively as feature-engineering
inspiration (lags, Fourier terms), never as a replicated architecture, and its full
citation was never part of the project brief. The hardcoded `"[ficha sin verificar]"`
suffix on the Cardozo row of `export_latex_tables.LITERATURE_COMPARISON` was removed
accordingly, `tabla_comparativa_estado_arte.tex` was regenerated, and
`test_literature_pending_verification_studies_are_flagged` in
`tests/test_export_latex_tables.py` was updated to assert Cardozo is **no longer**
flagged while Surribas-Sayago still is.

**Chapter 1 (`01_introduccion.tex`) formal prose: complete.** The `%%` skeleton was
replaced with full academic prose across an unheaded lead-in (context and honest
upfront statement of the central negative result, per the "guía Olivas" convention
already fixed in the skeleton — no `\section` header for this opening block) followed by
four sections: general and specific objectives (§1.1, OE1-OE5, each one mapped
one-to-one to a phase of the pipeline and to one of the five research questions);
research questions (§1.2, PI1-PI5 transcribed with **identical wording** to their
answers in §5.9, verified by direct text comparison — no paraphrasing drift between the
question as posed and as answered); project timeline and life cycle (§1.3, narrating the
three-block KDD-incremental structure — data preparation / modelling / exploitation and
documentation — behind the pre-existing `pgfgantt` Gantt chart and
`tabla_cronograma_fases` table, both left untouched, with the relative-axis honesty
caveat preserved verbatim); and document structure (§1.4, one-line summaries of chapters
2-6 and the three appendices, cross-referencing every `cap:*` label). `latexmk -pdf
-interaction=nonstopmode -halt-on-error TFT.tex` compiled clean on the **first** attempt
(0 errors, 0 undefined references, 73 pages) — no transient `latexmk` retry needed this
time. The Gantt chart and Cuadro 1.1 were visually verified to render fully inside their
page margins with no overflow. 310/310 tests still pass, no Python code touched.
Chapters 6 and 7 remain at the Phase 10 skeleton stage.

**Chapter 6 (`06_conclusiones.tex`) and Annexes (`07_anexos.tex`) formal prose:
complete — closes the formal-writing phase.** §6.1 synthesizes OE1-OE5 fulfilment
through the PI1-PI5 answers, framing the negative hybrid result as a Granger
(1989)/M4-M5-consistent scientific contribution and the ensemble's win as measured
error decorrelation (Bates & Granger 1969, Perrone & Cooper 1993). §6.2 states five
real limitations without euphemism: network-level aggregation with no
spatial/station breakdown, tiny test-split festivo/bridge-day samples (n=5, n=3), the
28-day warm-up cut, the CRTM source's own provisional-data note, and the deliberate
same-day-weather exclusion. §6.3 proposes three concrete, unexecuted extensions:
per-operator disaggregated modelling, real operational weather-forecast integration,
and patch/transformer-based temporal architectures. Annex A repeats the six-group,
161-variable `tabla_grupos_features` with prose taxonomy; Annex B documents all 310
tests and formally enumerates ten anti-leakage guarantees (shift-first
lags/rolling, chronological-only splits, train-only scaling, the Fourier-determinism
exception, OOF-only stage-1 predictions feeding stage 2, the operator-sum and
same-day-weather guards, `xgboost_alone`'s full-889-row training set, the
code-evaluated Experiment A verdict, and the shared-theme-object guarantee), each
tied to its concrete test name; Annex C documents the pinned environment
(`requirements.txt`), the `GLOBAL_SEED=42` convention in `src/utils/seed.py`
including TensorFlow op-determinism, and a 12-command end-to-end reproduction
sequence from raw data to compiled PDF.

Several `\allowbreak`-augmented long identifiers (test names, module paths) still
overflowed narrow columns in the new Annex text; wrapping the affected paragraphs in
`sloppypar` (plus `\small` itemize/enumerate blocks for the dependency-pin and
reproduction-command lists) resolved every Annex-specific overfull `\hbox`, leaving
only the already-documented, pre-existing minor overfulls from chapters 2-5. One
`-gg` clean rebuild hit the same transient `latexmk`-internal failure seen when
first compiling earlier chapters ("Extra }, or forgotten \endgroup" with a balanced
brace count verified independently); a direct `pdflatex` pass followed by a normal
`latexmk` run confirmed it was not a real source defect. **Final state: 80-page
`TFT.pdf`, 0 errors, 0 undefined references, 0 undefined citations, 310/310 tests
passing.** The entire memoria — Preliminares through Anexo C — now carries formal
academic prose end to end.

**Editorial closure pass (portada, preliminares, bibliografía, anexos, maquetación):
complete.** The audit's remaining open points were closed:

- **Preliminares written.** `00_preliminares.tex` was still a `%% PENDIENTE` stub
  despite the note above; it now carries the Spanish Resumen, an equivalent English
  Abstract, `\keywords`/Keywords (6 normalized terms each), and a sober institutional
  Agradecimientos. Renders on PDF pages 9-11.
- **Portada.** `TFT.tex` titlepage `tabularx` rebuilt into a clean 3-column block:
  Titulación = "Máster Universitario en Big Data y Ciencia de Datos", Alumno = "Aday
  Cuesta Correa", Convocatoria = "Primera Convocatoria (2025-2026)", Curso académico
  2025-2026. Director left as `XXXX` (not supplied).
- **Ch. 1 OE↔PI.** `01_introduccion.tex` §1.1/§1.2 already stated it, left intact:
  OE2/OE4/OE5 are answered by PI1-PI5; OE1 (ingestión + matriz anti-fuga) and OE3
  (esquema OOF) are design/integrity objectives with no PI, verified via the Anexo B
  test suite.
- **Ch. 4 ensemble + Phase 4 diagnostic.** `04_metodologia.tex` already had
  `\section{Ensamble por combinación de pronósticos}` (`sec:metodologia_ensemble`,
  Bates & Granger 1969 / Granger 1989 / Perrone & Cooper 1993, `ensemble_equal` vs
  `ensemble_inverse_mae`) and `\section{Diagnóstico previo}` (`sec:diagnostico_previo`,
  the SARIMAX residual diagnostic over the 1.282 post-warm-up rows: 48 festivos, 54
  puentes, 2,70x error on bridge days). A one-sentence forward reference to the two
  ensembles' test MAE (139.725 / 139.681) was added at the end of §4.6.
- **Bibliography** (`Bibliografia_TFT.bib`): `makridakis2018m4` `@inproceedings`→
  `@article` (IJF 34(4):802-808); new `makridakis2022m5` (IJF 38(4):1325-1336) cited
  alongside M4 in ch. 2, 5 and 6.1; `bates1969combination` journal `OR`→`Operational
  Research Quarterly`; `surribas_sayago_diffuse` author `... and et al.`→`... and
  others`, now `@misc`, note de-flagged. `06_conclusiones.tex` §6.3 patch/transformer
  mention decoupled from `makridakis2018m4` (anachronism).
  ⚠️ A `@misc` token *inside a `%%` header comment* aborted BibTeX (no line comments
  in `.bib`); reworded to "entrada de tipo misc".
- **Anexo C** (`07_anexos.tex`): reproduction command list corrected — was skipping
  `src.models.lstm.oof`, `src.models.lstm.window_comparison`,
  `src.models.hybrid_residual.xgboost_residual_weighted` (and now also lists
  `src.evaluation.sensitivity_report`); 15 steps. The "single literal orange" claim
  was corrected to acknowledge **two** definitions — `theme.py` for the Python/figure
  layer, `preamble.tex` (`\definecolor` of `naranja`/`slcolor`) for the LaTeX/table
  layer — one per layer, no third.
- **Maquetación.** Cuadros 1.1, 2.1 (apaisada; also cleared its `Float too large by
  42pt`), 4.2, 4.3, 4.4 wrapped in `\resizebox{\textwidth}{!}{...}` + `\tabcolsep`
  4pt at the section level (no table regeneration). A few prose overfulls
  (`CNN+LSTM+MLP`, the `916,9999...` float literal, `test_*` filenames) reduced via
  `sloppypar` / abbreviation.

**Final state:** `python -m pytest` → **310/310**. `latexmk -pdf
-interaction=nonstopmode -halt-on-error TFT.tex` → **84-page `TFT.pdf`, 0 errors,
0 undefined citations, 0 undefined references**, portada + preliminares complete.
16 minor `Overfull \hbox` remain (<20pt: compound-word section titles, LoF/LoT
entries, 3-11pt table floats), all cosmetic. `REPORT_MAPPING.md` §5-§6 synced.

**Diagnostic phase 7 — Stage 1 residual anatomy (§5.9): complete.** 368/368 tests
passing (58 new: `test_stage1_residual_anatomy.py` 22, `test_learning_curve.py` 8,
plus extended parametrise lists in `test_memoria_figures.py` /
`test_export_latex_tables.py`). Chapter 5 §5.6.1 *asserted* Stage 1 was the
bottleneck; this phase *measures* it. **DIAGNOSTIC CONTROL, NOT a candidate model** —
nothing here enters the master comparison. `test_phase7_adds_no_model_to_the_master_
comparison` pins that `dashboard_data.MODEL_NAMES`, `full_comparison.parquet` and
`sensitivity_comparison.parquet` model sets are all unchanged.

New: `src/evaluation/stage1_residual_anatomy.py` (analyses 1-3a, no retraining, no
`src.models` import) and `src/evaluation/lstm_learning_curve.py` (analysis 3b, the one
retraining step — 18 val-only fits, in-memory, **no artifact under `models/`**).
`src/models/lstm/final_model.py` gained an optional `max_train_rows` (trailing window
anchored at `train_end`; `None` is byte-identical to the old path, pinned by test).

**Analysis 1 — residual structure**, four acceptance criteria fixed as module
constants *before* any result:
- **Calendar concentration — STRUCTURE PERSISTS.** Mean |residual| on festivo is
  **6.52x** laborable-ordinary (puente 2.96x). Matches §5.5 (`feat_day_type_festivo`
  tops the residual model's gain) and the Phase 4 step-0 diagnostic. This is the clean
  finding: Stage 1 leaves unexplained exactly the irregular-calendar effect it cannot
  see.
- **Weather signal — NEAR WHITE NOISE.** 0/105 lagged-weather features keep a
  BH-significant partial Spearman ρ (q=0.10) once the four annual Fourier terms are
  partialled out; raw ρ≈0.28 for humidity/temperature collapse to ≈0.10. Per objection
  O7: this does **not** vindicate the hybrid — Stage 2 received those features
  (weather = 51.6% of its gain, §5.5) and still lost; the residual's remaining
  structure is calendar, not weather.
- **Temporal memory — WEAK/AMBIGUOUS.** max |ACF| = 0.166 (lag 28), between the 0.20
  structure and 0.10 white-noise thresholds; 5/35 lags outside the Bartlett band.
- **Weekly seasonality — WEAK/AMBIGUOUS.** ACF(7)=0.162; the single period-7 spectral
  bin is only 1.4x the mean bin share (spectral leakage into 6.4/6.7-day bins and the
  7/3-day harmonic); but the weekday-mean spread is 0.60σ, above the 0.30σ structure
  threshold. Ljung-Box rejects at all lags (p<0.01) but is anti-conservative here (O4)
  — effect size, not the p-value, carries the claim.

The two ambiguous verdicts get a sixth limitation paragraph in
`06_conclusiones.tex` §6.2, per the plan. `07_anexos.tex` Anexo B test count 310 →
368 (with the phase-7 breakdown) and Anexo C reproduction sequence gains the
`lstm_learning_curve` and `stage1_residual_anatomy` steps.

**Analysis 2 — the error chain** (`error_chain_table`, NOT a decomposition, O6). Test
split: `lstm_alone` 280,952 → `hybrid` 234,223 (Δ −46,729, **37.4%** of the 124,831
`lstm_alone`→`xgboost_alone` gap) → `xgboost_alone` 156,121. **62.6% (78,103 MAE)
still separates the corrected hybrid from a direct XGBoost.** Val: Stage 2 closes
78.4% of a wider gap, but the residual 35,895 still favours `xgboost_alone`.

**Analysis 3 — sample size.**
- *3a (per-fold, no retraining):* raw fold MAE is non-monotonic (1.46M → 282k → back to
  ~500k) and confounded with period difficulty. The skill ratio vs `seasonal_naive` on
  the identical block partly flattens it (folds 3, 5 reach parity; fold 1 degenerate
  at 2.67). 5 points — indicative, not a learning curve.
- *3b (learning curve — the retraining step):* trailing training windows at fractions
  0.25-1.00 of the 889-row block, 3 seeds, val-only. **Deliberately conservative**:
  small fractions train on the rows closest to val (recency advantage), biasing the
  experiment *against* a sample-size effect — stated in the module docstring and §5.9
  prose. Result is a **partial** sample-size story: 222 rows **collapses on all 3
  seeds** (degenerate: std ratio 0.30-0.43, corr ~0.45, same failure mode as fold 1);
  median val MAE then falls 1,167k → 516k (356 rows) → and **plateaus ~411-416k from
  756 rows on**, ~166k *above* `xgboost_alone` val MAE (244,843) and ~138k above
  SARIMAX (273,467), with no trend toward either line. Reading: sample size explains
  why Stage 1 is *very* weak (the sub-350-row collapse, the steep initial gain), but
  the univariate architecture's ceiling on this series sits well above the single-stage
  models even with all the data and a window design that favours it. Persisted:
  `data/processed/lstm_learning_curve.parquet` (per-fit rows incl. `pred_std_ratio`,
  `pred_actual_corr`, `degenerate`); no model artifact.

**Figures/tables/prose.** `memoria_figures.py` +7 builders (ACF/PACF, periodogram,
weekday profile, residual-vs-weather, error chain, fold-size twin panel, learning
curve — the last with **two reference lines** SARIMAX/`xgboost_alone` and **hollow
markers for all-seeds-degenerate fractions**). `export_figures.py` now writes **29
PNGs**; `export_latex_tables.py` **27 `.tex`** (+6). New §5.9
`sec:anatomia_residuo_etapa1` in `05_resultados.tex` (after `sec:mecanismo_ensemble`,
before `sec:respuesta_preguntas`), roadmap paragraph updated in lockstep.
`tabla_curva_aprendizaje` (8 cols) needed `\resizebox{\textwidth}{!}{...}` +
`\tabcolsep` 4pt (REPORT_MAPPING §5 incident 9). `latexmk` →
**96-page `TFT.pdf`, 0 errors, 0 undefined refs/citations, 0 `Overfull \hbox` ≥ 10pt**.
⚠️ **Superseded acceptance criterion** (post-phase-9 correction pass, REPORT_MAPPING
incident 12f): the `Overfull \hbox` count above is left as measured but **does not
establish margin compliance** — it cannot see a `p{}` cell wider than its declared
column. `tabla_cadena_error`, added in *this* phase, was 58pt outside the right margin
while that counter read zero. Margin compliance is now verified by direct `pymupdf`
measurement of rendered content against the text block.
`requirements.txt` pins `scipy==1.18.0` (imported directly for `signal.periodogram` /
`stats`).

**Diagnostic phase 8 — Informational parity and ablation (§5.10): complete.** 402/402
tests passing (34 new: `test_feature_block_ablation.py` 10, `test_lstm_parity_control.py`
15, `test_learning_curve.py` +1, plus extended parametrise lists in
`test_memoria_figures.py` / `test_export_latex_tables.py`). Answers two supervisor questions Phase 7 left open —
**Q1**: is Stage 1 losing because it is an *LSTM* or because it is *univariate*? **Q2**:
does direct XGBoost already integrate temporal + exogenous effects simultaneously, making
the two-stage split redundant? **HARD CONSTRAINT, same as Phase 7**: every model here is a
DIAGNOSTIC CONTROL, NOT a candidate. `test_phase8_adds_no_model_to_the_master_comparison`
pins that `dashboard_data.MODEL_NAMES`, `full_comparison.parquet` and
`sensitivity_comparison.parquet` are unchanged; `git status` shows nothing under
`models/`.

New: `src/evaluation/feature_block_ablation.py` (retrains `xgboost_alone` on restricted
feature subsets; no `models/` artifact) and `src/evaluation/lstm_parity_control.py` (the
retraining module — 10 val-only fits, in-memory, no artifact). Parametrised (byte-identical
defaults, each pinned by test): `xgboost_alone.train(feature_subset=...)`; `LSTMConfig.n_exog`
(0 → today's `Sequential`; >0 → two-input functional model, target window + day-*t* exog
vector concatenated after the LSTM branch); `sequence_builder.build_exog_matrix` (gathers
row *t*, NOT the window that excludes it); `scaling.FeatureScaler` (multi-column MinMax,
train-only, not persisted); `final_model.train_final_model(exog_columns=...)`.

**Analysis 1 — feature-block ablation (Q2).** Pre-registered subsets, materiality floor
`BLOCK_MATERIAL_PCT = 5.0` as a module constant, verdict on val (test = stability column).
Block-removal study, **NOT a decomposition** (objection O2: `fourier_annual`/lagged weather
both carry the annual cycle; `lag_total` 7/14/21/28 carries the weekly calendar — deltas
never sum to anything).

| Subset | n | val MAE | Δ val (%) | verdict |
|---|---:|---:|---:|---|
| completo (ref, not retrained) | 161 | 244,843 | 0.0 | referencia |
| solo_temporal | 39 | 461,854 | **+88.6** | aporta |
| solo_exogeno | 122 | 320,520 | **+30.9** | aporta |
| sin_meteo | 56 | 237,040 | **−3.2** | solo informativo |
| solo_calendario | 17 | 300,734 | +22.8 | solo informativo |

**Both mandatory subsets degrade past the 5% floor → `xgboost_alone` demonstrably
integrates BOTH information types internally** — mechanical corroboration (not new proof;
Phase 5 already proved the predictive half) that the two-stage split does across stages
what one tabular model does in one. Secondary: `sin_meteo` is 3.2% *better* than completo —
the 105-column weather block is a net non-contributor (matches Phase 7's near-white-noise
weather finding); `solo_calendario` at +22.8% is NOT near completo, so the series is not
calendar-determined — §6.1 conditional edit NOT triggered.

**Analysis 2 — informational-parity LSTM (Q1).** `Input(W,1)→LSTM→concat(exog Input(k))→
Dropout→Dense(1)` — minimal change for true parity (a plain `(W,k)` window still hides
row *t*'s calendar). **5 seeds [42,1,2,3,4]** (raised from 3 by the pre-registered §2.3/O6
rule after the timing probe showed headroom — slowest fit 9.7s vs ~6min ceiling — decided
before the run; provisional seed-42 numbers were visible and did NOT drive the choice, and
this is stated in the module docstring, §5.10.2 prose and here). **Baseline recomputed as
the 5-seed fraction-1.00 median of `lstm_learning_curve.parquet`.** The extra two seeds are
built into `lstm_learning_curve.main()` via `assemble_full_curve` (`SEEDS = [42,1,2]` stays
the curve's main set; `EXTRA_FRACTION1_SEEDS = [3,4]` runs at fraction 1.00 only,
de-duplicated on `(fraction, seed)`), so one run of the documented regen command reproduces
the whole artifact. The five values are 411,142 / 416,324 / 431,415 / 433,829 / 379,074 → median
**416,324** (unchanged from the 3-seed median; seed 1 is central either way).
`UNIVARIATE_VAL_MAE_BEST_SEED = 411_142` kept separately for the prose. `_GAP` = 416,324 −
244,843 = **171,481**. Thresholds: `PARITY_GAP_CLOSED_HIGH = 60.0` (median val MAE ≤
313,435 → `asimetría informativa`), `PARITY_GAP_CLOSED_LOW = 25.0` (≥ 373,454 →
`limitación arquitectónica`), between → `mixto/ambiguo`; degeneracy evaluated first,
min–max band straddling a threshold auto-downgrades to `mixto/ambiguo` (O6). Val-only
(deliberate asymmetry with the ablation — a new architecture's test number would be a
headline). Degeneracy thresholds imported from `lstm_learning_curve`, not re-declared. O4
stated before results: the parity LSTM sees strictly MORE than XGBoost (raw 28-step window
+ row *t*), so a loss is strong evidence and a partial win is weak. O8: a good result is an
argument *against* the hybrid (Stage 1 that sees the exog features leaves Stage 2 nothing
to correct).

| Scope | k | val MAE median | min–max band | gap closed (median) | verdict |
|---|---:|---:|---|---:|---|
| calendario | 17 | 388,625 | 382,695–395,877 | **16.2%** | **`limitación arquitectónica`** |
| completo | 161 | 363,133 | 354,235–391,630 | **31.0%** | **`mixto/ambiguo`** (band straddles the 25% threshold, O6) |

No seed degenerate, none undertrained (`best_epoch` median 75 / 53, well above
`UNDERTRAINED_EPOCH_MAX = 20`). **Q1 answer: Stage 1 loses primarily by being *recurrent on
this series at this sample size*, not by being univariate** — informational parity does not
rescue it. The sharpest test (`calendario`, exactly the features where Phase 7 located the
residual structure) closes only ~1/6 of the gap. `completo` does better (31%) but is
ambiguous, and by O4 a partial close is weak anyway.

**Conditional edits triggered:** `completo` verdict `mixto/ambiguo` → **seventh limitation
paragraph in `06_conclusiones.tex` §6.2** (Phase 7 sixth-paragraph shape). `solo_calendario`
NOT near completo → §6.1 sentence NOT added.

**Figures/tables/prose.** `memoria_figures.py` +2 builders (`block_ablation_figure`,
`parity_control_figure` — no layout title, hollow markers for all-seeds-degenerate,
reference lines via `add_hline`/`add_vline`); read the two new parquets directly, no
`src.models` import (pinned). `export_figures.py` now writes **31 PNGs**;
`export_latex_tables.py` **29 `.tex`** (+`tabla_ablacion_bloques`, `tabla_paridad_lstm`;
the builders emit a bare `tabular` with ASCII-only headers — the
`\resizebox{\textwidth}{!}{...}` + `\tabcolsep` 4pt wrap is applied where the tables are
`\input`-ed in `05_resultados.tex`, the incident 7/9 idiom). New §5.10
`sec:paridad_informativa` in `05_resultados.tex` (5.10.1 ablación / 5.10.2 paridad LSTM /
5.10.3 lectura conjunta) after `sec:anatomia_residuo_etapa1`, before
`sec:respuesta_preguntas` (label unchanged, auto-renumbered to §5.11); roadmap paragraph
and a §5.11 linking paragraph updated in lockstep. One real LaTeX defect fixed at source:
`$\geq 60\,\%$` in math mode → `! Incompatible glue units` (babel-spanish `\%`); moved the
percent out of math (`$\geq$ 60\,\%`). `REPORT_MAPPING.md` §5 incident 10. `latexmk` →
**102-page `TFT.pdf`, 0 errors, 0 undefined refs/citations, 0 `Overfull \hbox` ≥ 10pt**
(⚠️ secondary signal only — superseded acceptance criterion, see the phase 7 entry and
REPORT_MAPPING incident 12f).

---

**Diagnostic phase 9 — Subgroup and period breakdown (§5.11): complete.** 442/442 tests
passing (40 new: 29 in `tests/test_subgroup_breakdown.py` + 11 across
`test_memoria_figures.py` / `test_export_latex_tables.py`). Answers the supervisor's
question directly — *does the hybrid outperform in any period or day type (festivos,
puentes, weekends, anomalous demand)?* — searched systematically, with the search designed
so a negative reads as a finding. **HARD CONSTRAINT, same as Phases 7/8**: pure re-analysis
of `data/processed/*_predictions.parquet`, no model trained, nothing enters the master
comparison. `test_phase9_adds_no_model_to_the_master_comparison` pins it; the module has no
`src.models` import, no `.fit(` / `.train(`, no `models/` write.

| Module | Responsibility |
|---|---|
| `src/evaluation/subgroup_breakdown.py` | **new** — 15 pre-registered subgroups as module constants with pinned n-counts; `subgroup_metrics` / `hybrid_win_table` / `bootstrap_delta` / `evaluate_subgroup_criteria` / `rolling_mae`; writes `subgroup_breakdown.parquet`, `subgroup_rolling_mae.parquet` |

**Pre-registration.** `SUBGROUPS`, `MIN_N_INFERENTIAL = 20`, `ANOMALY_LAG/K = 7/3.0`,
`BLOCK_LENGTH = 7`, `N_BOOTSTRAP = 2000`, `FDR_Q = 0.10`, `VERDICT_SCOPE = "test"`,
`REPLICATION_SCOPE = "val"` are all module constants; only the n-counts were computed before
the design froze (pinned by `test_subgroup_sizes_match_the_preregistered_counts`). The
lag-7 anomaly rule is pre-registered; the rolling-median-28 variant (a weekend detector
that would duplicate `domingo`) is not reported. Anomaly MAD estimated on train only
(`_train_mad`, guarded in source and numerically).

**`demanda_anomala` verdict-scope resolution** (the subgroup the supervisor named). n=16 on
test, n=34 on val, pooled val_test n=50. Receives `VERDICT_NON_INFER` on the test verdict
scope — no exception. Its pooled view is reported only as a pre-registered SECONDARY
descriptive scope with the O6 caveat, never a verdict or headline. The pooled scope was
available and **deliberately not** promoted to verdict scope: per-subgroup choice of which
split decides is a researcher degree of freedom, least defensible exactly for the named
subgroup. `evaluate_subgroup_criteria` reads `VERDICT_SCOPE` uniformly; no per-subgroup
verdict scope exists (pinned by two tests + module docstring).

**Result — the hybrid wins nowhere.** Of 15 subgroups, 9 are inferential on test, 4 fall
below n=20 (laborable puente 3, festivo 5, calendario irregular 8, demanda anómala 16), 2
are excluded for lack of val/test overlap (Navidad, Semana Santa; O5). **In 0 of the 9
inferential subgroups does a hybrid rank first**; in 0 does one rank first without evidence.
In all 9 the moving-block bootstrap ΔMAE (best hybrid − best competitor) is positive and
BH-significant at q=0.10 — the hybrid is significantly *worse* by 53k–134k MAE. Under the
null, `9 × 2/11 ≈ 1.6` subgroups would rank first by chance; observed is 0. `verano_jul_ago`
CI suppressed (usable-resample fraction 0.81 < 0.90); p retained. The known festivo finding
(Stage 2 cuts Stage 1 error 1,585,388 → 666,572, −58%) is re-stated as a correction
mechanism, not a subgroup win (n=5, indicative).

**Conditional edits.** No `VERDICT_RANK_ONLY` subgroup → **no eighth limitation paragraph**
in `06_conclusiones.tex` §6.2 (it was conditioned on a rank-only verdict). No
`VERDICT_WIN` → no §6.1 sentence.

**Figures/tables/prose.** `memoria_figures.py` +3 builders (`subgroup_mae_figure`,
`subgroup_delta_ci_figure`, `period_rolling_mae_figure` — no layout title, hollow bars/markers
for non-inferential subgroups, val/test rule via `add_vline`); read `subgroup_breakdown.py`
directly, no `src.models` import (pinned). `export_figures.py` now writes **34 PNGs**;
`export_latex_tables.py` **32 `.tex`** (+`tabla_subgrupos_test`, `tabla_subgrupos_val`,
`tabla_subgrupos_inferencia`; `\resizebox` + `\tabcolsep` 4pt wrap at the `\input` site,
incident 7/9/10 idiom). Verdict strings written without `<` or `/` so `escape_latex` (no
`<` handling) does not break text mode. New §5.11 `sec:desglose_subgrupos` in
`05_resultados.tex` (5.11.1 diseño preregistrado / 5.11.2 resultados / 5.11.3 incertidumbre
y comparaciones múltiples / 5.11.4 vista temporal / 5.11.5 lectura) after
`sec:lectura_conjunta`, before `sec:respuesta_preguntas` (auto-renumbered to §5.12);
roadmap paragraph and a forward `\ref` from §5.4 updated in lockstep. Anexo B (442 tests)
and Anexo C (new `subgroup_breakdown` step) updated. `REPORT_MAPPING.md` §1–§6 + incident 11.
`latexmk` → **111-page `TFT.pdf`, 0 errors, 0 undefined refs/citations, 0 `Overfull \hbox`
≥ 10pt** (⚠️ secondary signal only — superseded acceptance criterion, see the phase 7
entry and REPORT_MAPPING incident 12f; Cuadros 5.15 and 5.16 were outside the right
margin at this point with the counter reading zero);
`sec:respuesta_preguntas` renumbered to §5.12 in `TFT.toc`.

**Post-phase-9 correction pass (supervisor review of §5.9): complete.** 445/445 tests
passing (3 new, all in `test_memoria_figures.py`). Five reported defects plus one found
while verifying them. **No model trained, nothing under `models/`, no computed number
changed**; the phase 7/8/9 guard tests all still pass.

- **Periodogram axis (fig. 5.16).** The reported diagnosis — a surviving DC component
  after the 1/frequency conversion — was **wrong**, and is recorded as wrong in
  `REPORT_MAPPING.md` incident 12a so the mistaken version does not survive.
  `periodogram_frame` already drops the zero bin; its table spans exactly 2.005–393.0
  days. The real cause is `add_vline(x=7.0, annotation_text=...)` on a log axis: Plotly
  converts the *shape* but not the annotation it creates, whose `x` is read as an
  already-log10 value, so the label landed at 10^7 and autorange followed to ~4e7. Fixed
  by placing the label separately at `log10(7)` and pinning the range to the table's own
  min/max. Band is **2 to N, not N/2** — the period-196.5 bin is the second-highest peak
  in the periodogram. Prose numbers re-verified unchanged (1.4436x period-7 share; top
  bin 2.339 d ≈ the 7/3 harmonic).
- **Cuadro 5.15 overflow.** The incident 7/9/10 `\resizebox` idiom was **not sufficient
  alone**: measured on the compiled PDF with `pymupdf`, the table still crossed the
  margin by 3.1pt, because "Significativa" is an unbreakable ~2.2cm word inside a
  `p{1.9cm}` column and `\resizebox` scales the *declared* width. Column widened to
  2.4cm in the builder as well. Effective font **10.24pt**.
- **Cuadro 5.16 (`tabla_cadena_error`) — found, not reported.** 9 columns, ~18.2cm
  natural width against 15.65cm, **58pt outside the margin**, no wrapper at all. Same
  idiom applied; effective font **10.02pt**.
- **Figure 5.19 annotations.** Collided with the neighbouring bar *and* sat beside the
  pair they do not describe (a second defect: `yshift=-26` pushed the `lstm_alone →
  hybrid` step toward `xgboost_alone`). Now anchored at the half-integer categorical
  position, i.e. in the gap between the two bars compared; both the absolute delta and
  the gap share are retained.
- **Prose.** Cuadro 5.14 gains the Cuadros 5.3/5.4 composition note (`laborable` n=269
  is the parent of its two children, not a sibling). §5.10.3 gains a paragraph
  reconciling the 51.6% impurity-gain share with the 0/105 partial correlations and the
  net-negative ablation, via impurity gain's bias toward high-cardinality continuous
  features. **§5.5's causal reading of the 51.6% is corrected at source** to a
  hypothesis that the diagnostic phases test and reject. A grep audit over all of
  `docs/LaTeX/`, `CLAUDE.md` and this log confirmed §5.5 was the **only** home of that
  reading — chapter 6 §6.2 and the resumen already stated it correctly and were left
  untouched.
- **`00_preliminares.tex` was three phases stale.** Both the Spanish resumen and the
  English abstract still claimed "310 pruebas" / "310 automated tests". Root cause is
  worth recording: **no prior sync list included `00_preliminares.tex`** — phases 7, 8
  and 9 each synced Anexo B and Anexo C and assumed the count propagated — so the number
  went stale in the first text a reader sees. Corrected to 445 in both languages.

**Test-count sync list — update all five together** (this list exists because omitting
one of them is exactly how the resumen went stale):

| Location | Text |
|---|---|
| `sections/00_preliminares.tex` (resumen) | "suite de N pruebas automáticas" |
| `sections/00_preliminares.tex` (abstract) | "a suite of N automated tests" |
| `sections/07_anexos.tex` (Anexo B header) | "**N tests**" + per-phase breakdown |
| `sections/07_anexos.tex` (Anexo C sequence) | "la suite completa de N tests" |
| `docs/CLAUDE_PHASE_LOG.md` (current pass entry) | "N/N tests passing" |

Historical phase entries are dated snapshots and are **not** retro-updated.

**Margin-compliance criterion replaced.** "0 `Overfull \hbox` ≥ 10pt", the criterion the
phase 7/8/9 entries used, does not establish margin compliance: the counter measures
horizontal boxes when setting a paragraph and cannot see a `p{}` cell whose content
exceeds its declared column width — TeX sets it without complaint and the glyphs cross
the margin silently. That is exactly how Cuadros 5.15 and 5.16 escaped, 5.16 by 58pt
across all three phases with the counter reading zero. **The criterion is now direct
measurement of rendered content against the text block** (a `pymupdf` pass comparing each
line's right `bbox` against `\textwidth`, 78.0–521.6pt under the current geometry), with
the `Overfull` count retained only as a secondary signal. The phase 7, 8 and 9 entries
have been annotated in place rather than rewritten — the figure each measured at the time
stands, marked as superseded.

- **Deliberately out of scope here:** the 23 pre-existing prose lines that cross the
  right margin (unbreakable `\texttt{}` identifiers in §5.10/§5.11). Not a formatting
  mistake, and they need one consistent remedy decided across the whole document — they
  go to the global audit, not a line-by-line fix in this pass.

`latexmk` → **111-page `TFT.pdf`** (unchanged), 0 errors, 0 undefined refs/citations.
23 lines still cross the right margin, all pre-existing prose with unbreakable
`\texttt{}` identifiers in §5.10/§5.11; none falls in a line range edited here. Note for
future passes: the "0 `Overfull \hbox` >= 10pt" claim in the phase 7-9 entries does not
survive direct measurement — an `Overfull` count cannot see a `p{}` cell overflowing,
which is precisely the defect class of Cuadros 5.15 and 5.16.

**Main workflow notebook (`notebooks/07_flujo_completo.ipynb`): complete.** 470/470 tests
passing (8 new: 5 in `tests/test_flujo_notebook.py`, 3 in `tests/test_features.py` for the
latent guard defect the notebook surfaced). Last outstanding supervisor
deliverable: a notebook that follows the whole flow — data loading and checking, feature
preparation, temporal split, running the main models, results comparison, final figures —
calling into `src/` rather than duplicating code. **Nothing under `models/` or `data/` was
written or changed**; the notebook runs everything with `save=False`.

- **06_dashboard kept, 07 added — not merged.** They are different deliverables: 06 is the
  interactive results panel and the KDD *deployment* stage that §1.3 cites; 07 is the
  end-to-end flow. Merging them would have required rewriting §1.3 — a **body** edit, and
  the body is float-bound at 80 pages — and would have contradicted 06's own first cell
  ("no entrena ni recalcula ningún modelo"). **Body page delta: 0**; every edit landed in
  annexes and front matter.
- **Recompute/load split, measured rather than assumed.** Live: ingestion + integrity
  (0.3 s), the 161-feature matrix (`.equals` with `features_daily.parquet`), split and
  warm-up, the four baselines **and SARIMAX** (0.9 s together, bit-exact against
  `baseline_predictions.parquet`), `xgboost_residual` (14.6 s) and `xgboost_alone`
  (14.9 s) — both bit-exact (`max|Δpred| = 0.0`) with hyperparameters identical to the
  published `.joblib`s. **SARIMAX did not need loading**: the ~6 minutes in Anexo C step 3
  are the AIC *grid search*, whose winner is already persisted in
  `sarimax_selected_order.json`; refitting at that order costs under a second. That leaves
  the **stage-1 LSTM as the only loaded model**, and for a methodological reason, not cost
  — stage 2 must correct residuals from the same stage-1 model that produced the published
  results (see `hybrid_residual/combine.py`'s docstring), so retraining it would measure
  the gap between two networks instead of the pipeline. Explained in prose in the notebook,
  not buried.
- **Self-verification is the point of the notebook.** 22 published rows x (4 metrics + `n`)
  = **110 values** contrasted cell by cell against `full_comparison.parquet`; the notebook
  **raises** above `rtol = 1e-6`. Observed deviation on this machine: `0.000e+00` across
  all 110. The tolerance absorbs no known drift — it exists only for float reassociation in
  XGBoost's multithreaded `hist` histogram across core counts, and 1e-6 relative on a test
  MAE of ~156,000 is 0.16 passengers/day.
- **The failure path was designed for its reader.** The cell **always** prints the full
  contrast table before raising, and the exception message *interprets* the magnitude:
  below `1e-3` relative it names an execution-environment difference and states explicitly
  that the published results stand; above it, a real pipeline change, with what to check.
  The likely reader is the supervisor on his own machine, and his reasonable first
  conclusion from a bare `AssertionError` would be that the work does not reproduce —
  which at float-reassociation scale is wrong.
- **Committed with outputs.** `nbconvert --to notebook --execute --inplace`: 26/26 code
  cells with saved output, 0 errors, 14 embedded Plotly figures, 32.5 s in-kernel /
  37.6 s wall. GitHub renders `.ipynb`, and the notebook will more likely be **read** than
  run; cleared outputs would show code and no results.
- **`src/` additions** (so no logic lives in cells): `ingestion.unify.integrity_report` —
  the four invariants incl. `metro+emt+carretera+cercanias == total`, which until now lived
  only in a docstring and the tests; `evaluation.full_comparison.verify_against_published`
  / `assert_within_tolerance` / `load_published` / `build_comparison_frame`; and
  `assemble_predictions` widened with four optional arguments that all default to reading
  from disk (default behaviour unchanged, pinned by the existing suite) so the notebook
  scores its recomputed pieces through **the same code path** that produced the published
  table.
- **Latent defect found while writing the guard demo, and FIXED.** The `LEAKAGE GUARD` and
  the first half of the `SAME-DAY WEATHER GUARD` inside `build_features` were
  **unreachable**: `FORBIDDEN_SAME_DAY_COLS` holds bare names (`total`, `metro`, …) while
  the check scanned only `feat_`-prefixed columns, so `leaked` could never be non-empty.
  Vacuous across several phases. **No leak ever occurred** — the effective protection was
  always structural (builders emit only lag ≥ 1 and rolling forms), and the per-module
  guards on each block's own output are reachable and do work. `features_daily.parquet` is
  byte-identical before and after; no published number moves.

  Both guards now scan for the **prefixed** name, which is what a builder regression would
  emit and the only thing that can appear in `feats`. Scanning bare names is dropped in
  both: against `feats` it is vacuous, and against `out.columns` it would raise on every
  valid frame, since `RAW_REFERENCE_COLS` deliberately carries the operator columns and
  `TARGET_COL` carries `total`. Fixed rather than documented-as-redundant because the
  structural argument is *conditional* — it holds only until someone adds a lag 0 to `LAGS`
  — which is precisely the regression a secondary net exists for; the block's docstring
  already stated the right intent, only the implementation missed the prefix. +3 tests in
  `test_features.py` (467 → **470**): two trip the guards via `monkeypatch`, the third pins
  the column contract so neither wrong rewrite can be made later.
  `REPORT_MAPPING.md` incident 16.
- **Test-count sync, 462 → 470**, all five prose occurrences updated in lockstep
  (`00_preliminares.tex` resumen + abstract, `07_anexos.tex` Anexo B header **and its
  257+56+58+34+40+17 breakdown, which now carries a seventh `+8` term and re-sums**, Anexo
  C step, Anexo D). `test_prose_claims_match_artifacts` green. `REPORT_MAPPING.md`'s sync
  list said "las cinco" and omitted the Anexo D row — corrected there; the list is stricter
  than the test that pins it, since the test only requires the number to appear somewhere
  in each of the two files.
- **REPORT_MAPPING numbering:** the file already contained **two incidents numbered 13**
  (pre-existing, not introduced here). Not renumbered — that would make existing
  cross-references to "incidencia 13" ambiguous. The new entry takes the next free ordinal,
  **15**, and records the duplication.

**Next (not started):** Colab migration.

### Phase 1 — Ingestion

Modules (`python -m src.ingestion.unify` runs the pipeline end-to-end; `python -m pytest`
runs the suite):

| Module | Responsibility |
|---|---|
| `src/utils/paths.py` | Repository-relative path constants; no absolute paths in `src/` |
| `src/ingestion/load_crtm.py` | CRTM demand → canonical |
| `src/ingestion/load_weather.py` | Open-Meteo weather → canonical |
| `src/ingestion/load_calendar.py` | Working calendar → canonical |
| `src/ingestion/unify.py` | Merge + temporal-integrity validation + parquet write |
| `tests/test_ingestion.py` | 30 tests: schema contracts + negative gap/duplicate tests |

**`pyarrow==22.0.0` was added to `requirements.txt` in this phase** — pandas needs a
parquet engine and Phase 0 did not include one.

#### Physical → canonical column mappings

**CRTM** (`read_excel(sheet_name='diaria', header=1)`; 9 raw columns asserted by exact
label before anything else, so schema drift fails loudly):

| Physical | Canonical | dtype |
|---|---|---|
| `Unnamed: 0` (stray `x`) | *dropped* | — |
| `Unnamed: 1` (unnamed date col) | `date` | datetime64[ns] |
| `Metro de Madrid` | `metro` | int64 |
| `EMT` | `emt` | int64 |
| `Conc. por carretera` | `carretera` | int64 |
| `Renfe Cercanías` | `cercanias` | int64 |
| `Total` | `total` | int64 |
| `Unnamed: 7` (empty) | *dropped* | — |
| `Nota: …provisionales…` | *dropped* | — |

Junk columns are dropped by **explicit label**, never by positional slice, so an inserted
column cannot shift the slice and discard real data.

> ⚠️ **Excel formula round-off.** `2026-07-07` arrives as `carretera=847904.00000001` and
> `total=4748815.00000001` (deviation ~1e-8) — float64 noise from an Excel formula, not
> fractional passenger counts. `load_crtm` rounds values within `INTEGER_TOLERANCE = 1e-6`
> and **raises on anything genuinely fractional**. Do not widen this tolerance to silence a
> future failure: a real fractional value would mean the column stopped meaning "trips".
> Verified invariant: `metro + emt + carretera + cercanias == total` on all 1310 rows.

**Weather** (`read_csv(skiprows=3)` — lines 1-2 metadata, line 3 blank, line 4 the real
header; verified against the bytes). `time` → `date`; all other names keep their source
form with the unit annotation stripped via regex `\s*\([^)]*\)\s*$` and snake_cased, e.g.
`temperature_2m_mean (°C)` → `temperature_2m_mean`, `shortwave_radiation_sum (MJ/m²)` →
`shortwave_radiation_sum`, `weather_code (wmo code)` → `weather_code`.

**Calendar** (`read_csv(sep=';', encoding='utf-8-sig')`; dates parsed with explicit
`format='%d/%m/%Y'`, never `dayfirst` inference):

| Physical | Canonical | dtype |
|---|---|---|
| `Dia` | `date` | datetime64[ns] |
| `Dia_semana` | `day_of_week_es` | object (unaccented ASCII) |
| `laborable / festivo / domingo festivo` | `day_type` | category |
| `Tipo de Festivo` | `holiday_type` | object |
| `Festividad` | `holiday_name` | object |

`day_type` categories are declared explicitly as
`['laborable', 'sabado', 'domingo', 'festivo', 'domingo festivo']`. Unmapped values raise
`ValueError` **before** the categorical cast (casting first would turn them into NaN and
destroy the evidence). Observed counts: laborable 892, domingo 188, sabado 181, festivo 49,
**domingo festivo 0** — the category is carried through ingestion and survives the parquet
round-trip, so a future occurrence encodes correctly instead of becoming NaN.

#### Unified schema — `data/interim/unified_daily.parquet`

**1310 rows × 31 columns, 2023-01-01 → 2026-08-02, daily, zero gaps, zero duplicates,
zero nulls.**

```
 0  date                        datetime64[ns]
 1-5 metro, emt, carretera, cercanias, total          int64
 6-11 temperature_2m_{mean,max,min},
      apparent_temperature_{mean,max,min}             float64
 12  precipitation_sum                                float64
 13-15 relative_humidity_2m_{mean,max,min}            int64
 16-18 wind_speed_10m_{max,mean,min}                  float64
 19-21 pressure_msl_{min,max,mean}                    float64
 22  shortwave_radiation_sum                          float64
 23  precipitation_hours                              float64
 24  rain_sum                                         float64
 25  snowfall_sum                                     float64
 26  weather_code                                     int64
 27  day_of_week_es                                   object
 28  day_type                                         category
 29  holiday_type                                     object
 30  holiday_name                                     object
```

**Holiday-column sentinel:** `holiday_type`/`holiday_name` are NaN on non-holidays in the
source. Unification replaces that with the string `'none'` (`unify.NO_HOLIDAY`). This is
deliberate: it keeps the "zero nulls" invariant *honest*, so a null anywhere in this dataset
always signals a real defect rather than ordinary non-holiday absence, and it gives Phase 2
a clean categorical level to encode. Phase 2 must treat `'none'` as a level, not a value to
drop or impute.

#### Validation performed by `unify_frames`

Merging is an inner join on `date` — a left join would fabricate all-NaN exogenous rows —
but an inner join also *hides* coverage gaps by construction, so every check below exists to
make a swallowed gap raise instead:

1. Per-source duplicate-date check before merging.
2. Post-merge row count compared against **each** source; on mismatch, raises naming the
   specific missing dates rather than reporting a count.
3. Reindex against a complete `pd.date_range(min, max, freq='D')` and compare lengths —
   this catches a hole **common to all three sources**, which step 2 cannot see.
4. Final duplicate and null assertions on the unified frame.

Negative tests inject synthetic gaps into small in-memory fixtures (not the real files) and
assert each path raises, so these guards cannot silently rot into no-ops.

---

### Phase 2 — Feature Engineering

`python -m src.features.build_features` runs the phase end-to-end.

| Module | Produces |
|---|---|
| `src/features/naming.py` | `FEATURE_PREFIX = "feat_"` + `slugify` for category column names |
| `src/features/lag_features.py` | 35 lag columns + the same-day operator leakage guard |
| `src/features/rolling_features.py` | 4 shift-first rolling columns |
| `src/features/fourier_features.py` | 6 deterministic seasonality columns |
| `src/features/calendar_features.py` | 11 one-hot / flag columns |
| `src/features/build_features.py` | Orchestrator, parquet write, NaN report |
| `tests/test_features.py` | 36 tests |

**Output: `data/processed/features_daily.parquet` — 1310 rows × 87 columns,
2023-01-01 → 2026-08-02. Row count UNCHANGED from `data/interim` (1310); no row is
dropped in this phase.**

#### Column groups

Select the model matrix with `build_features.feature_columns(df)` — every engineered
column carries the `feat_` prefix, so the model-ready set is derived mechanically rather
than from a hand-maintained list that would silently drift.

**Lag (35)** — `feat_{total,metro,emt,carretera,cercanias}_lag_{1,2,3,7,14,21,28}`

**Rolling (4)** — `feat_total_roll_{mean,std}_{7,28}`

**Fourier (6)** — `feat_fourier_weekly_k1_{sin,cos}`,
`feat_fourier_annual_k1_{sin,cos}`, `feat_fourier_annual_k2_{sin,cos}`

**Calendar (11)** — `feat_day_type_{laborable,sabado,domingo,festivo,domingo_festivo}`,
`feat_holiday_type_{none,festivo_nacional,festivo_de_la_comunidad_de_madrid,festivo_local_de_la_ciudad_de_madrid}`,
`feat_is_weekend`, `feat_is_bridge_day`

**Raw reference, NOT features (9)** — `date`, `total` (the target), `metro`, `emt`,
`carretera`, `cercanias`, `day_of_week_es`, `day_type`, `holiday_type`, `holiday_name`

**Passthrough weather (21)** — the Phase 1 weather columns, carried through unmodified
and deliberately **not** `feat_`-prefixed. See the open decision below.

#### Same-day operator leakage guard

`metro + emt + carretera + cercanias == total` exactly (verified Phase 1), so a same-day
operator column is not "a strong feature" — it is the target itself, arithmetically. A
model given them would score near-perfectly and collapse at inference, when the day's
operator split is not yet known. **Lagged** operator values are legitimate and are kept:
yesterday's modal split is genuinely known and carries real signal.

The guard is enforced twice: `lag_features._assert_no_same_day_leakage` raises if any of
the four appears in its own output, and `build_features` re-asserts on the **assembled**
frame, since `concat` is where a mis-named column would actually land in the matrix.
Tested by injecting a raw column and asserting the guard fires.

#### ⚠️ Correction to the shift-vs-rolling framing in Leakage Rule 1

Rule 1 above mandates `s.shift(1).rolling(w).mean()` and bans
`s.rolling(w).mean().shift(1)`. **Keep following the rule**, but note the precise facts,
verified empirically in this phase — a vague understanding invites the wrong fix later:

1. The **unshifted** window is the real leak, and it is not subtle: `rolling(w).mean()`
   at row t averages y_{t-w+1}..y_t, containing the value being predicted. On
   `[1,2,4,8,…]` with w=3 it gives 4.667 at index 3 where the correct value is 2.333.
2. For a **trailing** window, shift-then-roll and roll-then-shift are **mathematically
   identical** — confirmed for `min_periods=w`, `min_periods=1`, and for a source series
   containing NaNs. Both give `mean(y_{t-w}..y_{t-1})` with identical NaN placement.
   **Do not go hunting for a bug in roll-then-shift code: there isn't one for trailing
   means.**
3. That equivalence is a property of trailing windows, *not* a general law. With
   `center=True` the orderings genuinely diverge — roll-then-shift emits a final value
   built from future observations where shift-then-roll correctly emits NaN.

Shift-first is mandated anyway because its correctness never depends on noticing which of
(2) or (3) applies. All three cases are pinned by tests.

#### Fourier terms are the one sanctioned exception to Leakage Rule 4

Rule 4 bans global feature engineering before the split because such statistics *estimate
parameters from observed y*. Fourier basis terms estimate nothing: the value at date t is
a closed-form function of t alone, so computing them over the full range is byte-identical
to computing them per-split. Pinned by `test_fourier_is_deterministic_and_split_invariant`.
Fitting a *seasonal model* on full data would still leak — only the deterministic basis is
exempt; its coefficients are learned inside the training fold as normal.

`FOURIER_EPOCH = 2023-01-01` is a fixed constant, **not** `df['date'].min()`, so appending
future data never retroactively changes the encoding of existing rows.

#### Other deliberate decisions

- **`day_of_week_es` is NOT one-hot encoded.** A choice, not an oversight: seven dummies
  would be near-collinear with the weekly Fourier pair and `is_weekend`, inflating
  dimensionality against only 1310 observations. Kept as a raw reference column for EDA;
  revisit in Phase 3 if per-weekday residual structure appears.
- **No implicit reference level.** Every category gets a real column, including the
  `'none'` holiday sentinel and the zero-observation `domingo festivo`. A dropped
  reference level makes an all-zero row ambiguous between "base category" and "encoding
  bug"; both one-hot blocks are tested to sum to exactly 1 per row.
- **`is_bridge_day`** = `laborable` AND adjacent (t-1 or t+1) to `festivo`/`domingo
  festivo`. Reading t+1 is a forward look at the *calendar*, not the target — Spanish
  holidays are published a year ahead — so it is leakage-free. Series boundaries use
  `fill_value=False` (unknown neighbour treated as non-holiday), which can only clear the
  flag, never set it spuriously. **55 bridge days** detected, all `laborable`; spot-checked
  against 2023-05-03 (after the 1–2 May double holiday) and 2023-08-14/16 (around Asunción).

#### NaN policy — left in place, deliberately

The first **28 rows** carry NaNs from the 28-day lag and 28-day rolling window; the first
fully-usable row is **index 28 = 2023-01-29**. 39 of 56 features have NaNs; max 28 in any
one. **These are NOT dropped or imputed in Phase 2.** The decision depends on the
consumer — the LSTM needs a contiguous NaN-free window, XGBoost handles NaNs natively and
would lose 28 observations if rows were dropped — so it belongs to Phase 3. Row count is
asserted unchanged at 1310 by `build_features` and by test.

#### ⚠️ OPEN DECISION for Phase 3 — same-day weather

The 21 weather columns are passed through **unprefixed**, so they are currently **excluded**
from `feature_columns()`. Whether same-day observed weather may serve as an exogenous input
depends on the forecasting protocol: it is standard in demand studies (a weather forecast is
the operational stand-in) but is not strictly known at prediction time. **Resolve this
deliberately in Phase 3** — either promote them to `feat_` or replace them with
lagged/forecast equivalents. Do not let them drift into the model matrix unexamined.

---

### Phase 3 — Splits, Weather Promotion, Baselines

`python -m src.models.baselines.run_baselines` runs the phase end-to-end (~6 min; the
SARIMAX grid search dominates).

| Module | Responsibility |
|---|---|
| `src/features/weather_features.py` | 105 lagged/rolling weather features + same-day guard |
| `src/utils/splits.py` | **Single source of truth** for split boundaries + warm-up policy |
| `src/evaluation/metrics.py` | MAE, RMSE, MAPE, R², comparison table |
| `src/models/baselines/persistence.py` | Naive + seasonal-naive |
| `src/models/baselines/moving_average.py` | Trailing 7-day and 28-day means |
| `src/models/baselines/sarimax.py` | Grid search + walk-forward one-step-ahead |
| `src/models/baselines/run_baselines.py` | Orchestration and reporting |
| `tests/test_splits.py`, `tests/test_baselines.py` | 45 tests |

#### Split boundaries — DO NOT RECOMPUTE LOCALLY

**Every model in Phases 3-5 must import these from `src.utils.splits.chronological_split`.**
If the LSTM and the baselines disagree by even one day about where the test set starts,
their metrics are not comparable and the whole comparison table is meaningless.

| Split | Start | End | Days | Share |
|---|---|---|---|---|
| train | 2023-01-01 | 2025-07-05 | 917 | 70.0% |
| val | 2025-07-06 | 2026-01-17 | 196 | 15.0% |
| test | 2026-01-18 | 2026-08-02 | 197 | 15.0% |

Boundaries are pinned by `test_split_boundaries_are_exact_and_stable`.

> ⚠️ **Floating-point floor.** `0.70 * 1310` evaluates to `916.9999999999999` in binary
> float, so a bare `int()` silently yields a 916-day training set where `floor(0.70n) = 917`
> is intended. `chronological_split` adds `EPS = 1e-9` before flooring. Rounding is resolved
> on **cumulative** fractions (`floor(0.70n)`, `floor(0.85n)`) so the three blocks tile the
> range exactly — per-split rounding would drop or duplicate a boundary day.

#### Warm-up NaN policy

**28 rows dropped (2023-01-01 → 2023-01-28), all inside the training block.** Training
falls 917 → 889 rows; **validation (196) and test (197) are untouched.** After the trim,
zero engineered features are NaN in training.

`apply_warmup_policy` **raises** if the warm-up window would reach into validation or test,
rather than silently shortening the evaluation period — a quietly truncated eval set
produces metrics that are incomparable across models and across re-runs.

#### Weather promotion — lagged only (deliberate conservative choice)

The Phase 2 open question is resolved. **105 new features** = 21 weather variables ×
(lags 1, 2, 3, 7 + `roll_mean_7`). Feature matrix: **56 → 161**; table 87 → 192 columns.
Row count unchanged at 1310; warm-up still 28.

**Same-day weather never enters the model matrix.** It is observed data, not a forecast, so
using it unlagged assumes a perfect weather forecast at prediction time. The inflation is
worst exactly on anomalous days (storms, heatwaves) where the weather signal matters most
and a real forecast is least reliable. The raw columns survive as reference/EDA data only;
`_assert_no_same_day_weather` plus a re-check in `build_features` enforce this.

> **DEFERRED SENSITIVITY ANALYSIS.** The same-day-weather variant is *not discarded* — it
> belongs in the evaluation phase as an explicitly labelled sensitivity analysis answering
> "how much of the achievable gain depends on forecast quality?". Report it as an upper
> bound under a perfect-forecast idealization, clearly marked as such. **Never as the
> headline model.**

**Why lags [1,2,3,7] and not [1,2,3,7,14,21,28] as for `total`:** the two series have
different memory. Demand autocorrelation at lag 28 is *calendar-driven* (four weeks = same
weekday, same weekly regime). Atmospheric autocorrelation has no such mechanism — synoptic
patterns decay over 3-7 days, and temperature 28 days ago carries only the seasonal signal
that the annual Fourier terms already represent more cleanly. Lags 14/21/28 would add 63
near-redundant columns against 1310 observations.

#### SARIMAX design

**Selected order: `SARIMAX(2,1,2)x(0,1,2,7)`, training AIC = 24,616.89**, chosen by AIC over
324 combinations (p,q,P,Q in {0,1,2}; d,D in {0,1}; s=7). Persisted to
`data/processed/sarimax_selected_order.json`. The top 5 lie within 0.8 AIC of each other,
so the specific pick is not strongly identified — treat it as "an ARIMA of roughly this
complexity", not a discovered truth.

Selection uses **training AIC only**, never validation error: tuning on validation would
destroy its status as an out-of-sample estimate for the Phase 4/5 comparison.

**Exog is calendar-only (5 columns)** — `feat_is_weekend`, `feat_is_bridge_day`, and the
three real `holiday_type` dummies. **No lag or rolling features**, enforced by
`assert_no_lag_features_in_exog`: SARIMAX already models autoregression through its own
(p,d,q)(P,D,Q,s) terms, and hand-built lags would express the same dependency twice in two
competing mechanisms. `feat_holiday_type_none` is dropped **here only** as the reference
level — the exhaustive Phase 2 encoding is singular against a regression intercept (dummy
trap). The feature table itself keeps its explicit encoding.

**Forecasting is walk-forward one-step-ahead, not a single static multi-step forecast.**
Parameters are estimated once on train, then applied to the full series via `.filter()`
(Kalman filter only — no re-estimation, so no val/test observation influences a
coefficient); `get_prediction(dynamic=False)` conditions each prediction at t on observed
data through t-1. A one-shot ~390-day forecast would decay toward the series mean and
measure extrapolation distance rather than model quality. Verified: residual lag-1
autocorrelation is about -0.03, and predictions track same-day actuals (MAE 163,627) not
the previous day (909,082), ruling out an alignment bug.

#### Baseline comparison (passengers/day)

**TEST split** (2026-01-18 → 2026-08-02, n=197):

| Model | Description | RMSE | MAE | MAPE | R² |
|---|---|---:|---:|---:|---:|
| sarimax | SARIMAX + calendar exog, walk-forward 1-step | 241,147 | 163,627 | 3.85% | 0.9672 |
| seasonal_naive | ŷ_t = y_{t-7} | 690,480 | 309,817 | 7.16% | 0.7308 |
| moving_average_7 | ŷ_t = mean(y_{t-7}..y_{t-1}) | 1,260,492 | 1,093,679 | 28.13% | 0.1029 |
| moving_average_28 | ŷ_t = mean(y_{t-28}..y_{t-1}) | 1,301,939 | 1,119,987 | 28.74% | 0.0430 |
| persistence | ŷ_t = y_{t-1} | 1,390,444 | 900,570 | 21.79% | -0.0916 |

**VAL split** (n=196), RMSE / MAE / MAPE / R²: sarimax 404,153 / 273,467 / 7.31% / 0.9248 ·
seasonal_naive 883,627 / 460,615 / 12.63% / 0.6408 · moving_average_7
1,237,156 / 1,068,961 / 28.86% / 0.2958 · moving_average_28
1,357,281 / 1,137,759 / 31.12% / 0.1524 · persistence 1,387,393 / 914,119 / 23.72% / 0.1144

**TRAIN split** (n=889): sarimax 411,494 / 241,982 / 6.56% / 0.9088 · seasonal_naive
886,040 / 439,381 / 11.61% / 0.5772 · moving_average_7 1,250,020 / 1,089,063 / 29.84% /
0.1586 · moving_average_28 1,296,740 / 1,133,500 / 31.00% / 0.0945 · persistence
1,385,175 / 921,712 / 24.03% / -0.0332

#### Reading these numbers — three cautions for Phases 4-5

1. **Plain persistence is a weak straw man on this series** (negative R² on train and
   test). Daily demand swings by roughly 2.5M between weekday and weekend, so y_{t-1} is
   wrong by a full weekend transition twice a week. **`seasonal_naive` (ŷ_t = y_{t-7}) is
   the honest naive bar** — beating persistence proves nothing.
2. **The moving averages score poorly** because they smooth *across* the weekly cycle,
   sitting permanently between the weekday and weekend levels. Expected, not a bug.
3. **SARIMAX is a strong baseline** (test MAPE 3.85%, R² 0.967), and the LSTM must be
   compared against it on identical splits and the identical one-step-ahead protocol. Note
   test scores exceed validation scores for every model, so the test period is intrinsically
   easier — **never compare a Phase 4/5 test number against a Phase 3 validation number.**

#### Artifacts written

- `data/processed/features_daily.parquet` — rebuilt, 1310 × 192
- `data/processed/baseline_predictions.parquet` — date, total, and all 5 baseline series
- `data/processed/sarimax_selected_order.json` — selected order, AIC, exog list


---

### Phase 4 — LSTM Stage 1

| Module | Responsibility |
|---|---|
| `src/utils/seed.py` | `GLOBAL_SEED = 42`, applied to random/numpy/TF + op determinism |
| `src/evaluation/residual_diagnostics.py` | Exploratory: where the linear baseline fails |
| `src/models/lstm/sequence_builder.py` | Univariate sliding windows, W parameterised |
| `src/models/lstm/scaling.py` | MinMax fit on train only, persisted for Phase 5 |
| `src/models/lstm/model.py` | Architecture + early stopping config |
| `src/models/lstm/oof.py` | **Walk-forward OOF generation — the critical module** |
| `src/models/lstm/final_model.py` | Final model, val/test predictions, artifacts |
| `src/models/lstm/window_comparison.py` | W in [14, 28, 60] evidence |
| `tests/test_lstm.py` | 27 tests |

#### Step 0 diagnostic — the hybrid's premise is supported

SARIMAX residuals by calendar characteristic, whole period (exploratory, not out-of-sample;
holidays are too rare to split):

| Group | n | mean abs error | mean abs % |
|---|---:|---:|---:|
| festivo | 48 | 492,092 | 20.50% |
| domingo | 184 | 247,373 | 10.16% |
| sabado | 177 | 262,194 | 8.12% |
| laborable | 873 | 212,383 | 4.27% |
| **laborable, bridge day (puente)** | **54** | **518,634** | **12.89%** |
| laborable, ordinary | 819 | 192,191 | 3.71% |

**Bridge days carry 2.70x the mean absolute error of ordinary working days**, and festivo
days 20.5% MAPE against 4.27% on laborable. Signed bias is negative on holidays
(-145k on festivo), i.e. the linear model **over-predicts** demand on holidays — it knows a
holiday is unusual but not how unusual. This is exactly the nonlinear, calendar-conditioned
structure Stage 2 is meant to absorb, so the residual hybrid has real work available to it.

#### Window length — W = 28 (evidence, not assertion)

One final-model run per W, scored on validation (test deliberately not consulted):

| W | train sequences | epochs (best) | val MAE | val RMSE | val MAPE | val R² |
|---:|---:|---:|---:|---:|---:|---:|
| 14 | 875 | 154 (144) | 459,813 | 721,182 | 13.28% | 0.7607 |
| **28** | **861** | **164 (154)** | **411,142** | **655,479** | **11.81%** | **0.8023** |
| 60 | 829 | 87 (77) | 456,063 | 692,825 | 12.89% | 0.7792 |

W=28 wins on every validation metric, and the ~11% MAE gap to both neighbours is wide
enough not to be a coin flip. Single run per W, so this is a documented comparison, not a
tuned search.

#### OOF scheme — expanding-window walk-forward, 5 folds

Training period 2023-01-29 → 2025-07-05 (889 rows post warm-up). Fold k trains on
[0, s_k) and predicts [s_k, s_{k+1}); each fold refits its own scaler on its own training
slice only.

| Fold | Train | OOF block | n | Epochs (best) | MAE | RMSE |
|---:|---|---|---:|---|---:|---:|
| 1 | 2023-01-29→2023-08-16 | 2023-08-17→2023-12-31 | 137 | 15 (5) | **1,461,905** | 1,682,893 |
| 2 | 2023-01-29→2023-12-31 | 2024-01-01→2024-05-16 | 137 | 153 (143) | 659,425 | 882,524 |
| 3 | 2023-01-29→2024-05-16 | 2024-05-17→2024-09-30 | 137 | 228 (218) | 282,122 | 397,512 |
| 4 | 2023-01-29→2024-09-30 | 2024-10-01→2025-02-14 | 137 | 131 (121) | 548,081 | 839,318 |
| 5 | 2023-01-29→2025-02-14 | 2025-02-15→2025-07-05 | 141 | 81 (71) | 506,148 | 718,174 |

**Overall OOF: MAE 690,460 · RMSE 997,520 over 689 rows.**

**Excluded rows: the first 200 training rows (2023-01-29 → 2023-08-16) receive NO OOF
prediction**, because predicting them would need a model trained on less than the minimum
initial window. They are written to the parquet with `has_oof = False` and a stated
`exclusion_reason` rather than dropped, so Phase 5 can see why its residual training set
starts on 2023-08-17. **OOF coverage: 689 / 889 training rows (77.5%).**

> ⚠️ **FOLD 1 IS DEGENERATE — Phase 5 must decide what to do with it.**
> Its OOF predictions have a standard deviation only **0.199×** that of the actuals and
> correlate **0.300** with them: with 200 rows (172 sequences, 147 after the early-stopping
> tail) the model never escaped predicting a near-constant, and early stopping fired at
> epoch 15 on a noisy 25-sequence signal. Folds 2-5 are healthy by comparison
> (correlation 0.79-0.95, std ratio 0.75-0.94).
> **Consequence for Phase 5:** fold 1's residuals are dominated by stage-1 failure rather
> than by the calendar structure XGBoost is supposed to learn. The `fold` column exists in
> the output file precisely so Phase 5 can exclude or down-weight it. Consider dropping
> fold 1 (137 rows, leaving 552) and report that choice explicitly.

#### Final model

`LSTM(units=32, dropout=0.2, window=28)` → Dropout → `Dense(1, linear)`; Adam(lr=1e-3),
MSE, batch 32, early stopping patience 10 with `restore_best_weights`. Trained on all 889
post-warm-up training rows (861 sequences), stopping at epoch 164 (best 154).

**The early-stopping slice is the last 129 training sequences (2025-02-27 → 2025-07-05) —
a tail of TRAINING, never the official val split.** Wiring the official val split into
Keras' `validation_data` would let early stopping select the epoch that best fits val,
turning it into a tuning set and making every val number in the comparison table optimistic
by an unknown margin.

#### LSTM alone vs SARIMAX — the LSTM UNDERPERFORMS, clearly

| Split | Model | MAE | RMSE | MAPE | R² |
|---|---|---:|---:|---:|---:|
| val | **SARIMAX** | **273,467** | **404,153** | **7.31%** | **0.9248** |
| val | LSTM | 411,142 | 655,479 | 11.81% | 0.8023 |
| test | **SARIMAX** | **163,627** | **241,147** | **3.85%** | **0.9672** |
| test | LSTM | 280,952 | 431,714 | 6.62% | 0.8948 |

LSTM train (in-sample, reference only): MAE 365,351 · RMSE 608,326 · MAPE 10.12% · R² 0.8025.

On identical OOF rows (n=689): LSTM OOF MAE 690,460 vs SARIMAX 225,684 vs seasonal_naive
461,087. **The LSTM does not even beat seasonal-naive out-of-fold.**

**This is expected, not a defect, and must not be papered over.** The Stage 1 LSTM is
*univariate by design* — per the architecture, it sees only the past 28 values of `total`
and must infer weekday, holiday, and bridge-day effects from demand history alone. SARIMAX
gets explicit weekly seasonal differencing **and** a calendar exog matrix. The comparison is
therefore not like-for-like: it measures what temporal dynamics alone can do, which is the
question Stage 1 is supposed to answer.

The honest framing for the memoria: **Stage 1 alone is the weaker model, and the hybrid's
case rests entirely on Stage 2 recovering the calendar information Stage 1 structurally
cannot see.** The Phase 5 comparison that matters is `LSTM + XGBoost-on-residuals` versus
**SARIMAX** and versus **XGBoost alone** — not versus the LSTM, which is a low bar.
If the hybrid cannot beat SARIMAX, that is a finding to report, not a failure to hide.

#### Corrections made during this phase

1. **Epoch cap was binding, not early stopping.** The first OOF run used `epochs=200` and
   fold 3 finished at epoch 200 with `best_epoch=200` — still improving when cut off. Since
   folds differ in size they were being trained to *different* effective budgets, so they
   were not comparable to each other. Raised to 500 so `patience=10` governs. Fold 3 then
   ran to 228 epochs and MAE improved 288,194 → 282,122.
2. **Minibatch shuffling was tested, not assumed, and rejected.** Shuffling pre-built
   windows would not be leakage (each sample is self-contained and the early-stopping tail
   is split off chronologically first), but measured on fold 3 it made results *worse*:
   OOF MAE 288k → 309k. `shuffle=False` retained on evidence.

#### Artifacts written

- `data/processed/lstm_oof_predictions.parquet` — date, y_true, y_pred_oof, fold, has_oof, exclusion_reason
- `data/processed/lstm_oof_fold_log.parquet` — per-fold epochs/losses/metrics
- `data/processed/lstm_val_test_predictions.parquet` — final model on val + test
- `data/processed/lstm_window_comparison.parquet`
- `models/lstm_stage1_final.keras`, `models/lstm_stage1_scaler.joblib` — load these in
  Phase 5 rather than retraining, so residuals are computed against the identical model


---

### Phase 5 — Hybrid Residual Model

| Module | Responsibility |
|---|---|
| `src/evaluation/fold_coverage_check.py` | Decision support for the fold 1 exclusion |
| `src/models/hybrid_residual/assemble_residual_dataset.py` | Residual training set + guards |
| `src/models/hybrid_residual/tuning.py` | Shared grid search (identical for both models) |
| `src/models/hybrid_residual/xgboost_residual.py` | Stage 2: XGBoost on OOF residuals |
| `src/models/hybrid_residual/xgboost_alone.py` | Control: XGBoost directly on `total` |
| `src/models/hybrid_residual/combine.py` | `y_final = y_lstm + e_xgb` |
| `src/evaluation/day_type_breakdown.py` | Shared per-day-type error breakdown |
| `src/evaluation/full_comparison.py` | Master results table |
| `tests/test_hybrid_residual.py` | 22 tests |

#### Fold 1 exclusion — evidence, then decision

Fold 1 was excluded from the residual training set. The check that justified it:

| Category | count in fold 1 | count in folds 2-5 | zero coverage elsewhere |
|---|---:|---:|---|
| day_type_laborable | 91 | 377 | no |
| day_type_sabado | 20 | 76 | no |
| day_type_domingo | 20 | 78 | no |
| day_type_festivo | 6 | 21 | no |
| day_type_domingo_festivo | 0 | 0 | no |
| holiday_type_none | 131 | 531 | no |
| holiday_type_festivo_nacional | 5 | 16 | no |
| holiday_type_festivo_de_la_comunidad_de_madrid | 0 | 2 | no |
| holiday_type_festivo_local_de_la_ciudad_de_madrid | 1 | 3 | no |
| is_bridge_day | 9 | 24 | no |

**Verdict: exclusion SAFE.** Every category occurring in fold 1 recurs in folds 2-5, so
dropping it removes redundant coverage rather than unique information. Had any category
shown zero coverage elsewhere, the pipeline stops instead of proceeding.

#### Residual training set

**552 rows, 2024-01-01 → 2025-07-05, 161 features, zero nulls.**
889 train rows − 200 (no OOF, initial history) − 137 (degenerate fold 1) = 552.
Residual stats: mean 109,601 · std 726,669 · range [−3,128,729, +1,899,895].
Sign convention `residual = y_true - y_pred_oof`, so the correction **adds**.

> **Known limitation, stated rather than discovered later.** These residuals come from fold
> models trained on 337-748 rows, while the stage-1 model used at inference trains on all
> 889. The final model is therefore better than the models that generated the residuals, so
> stage 2 learns a residual distribution slightly wider than the one it actually faces.
> This is inherent to OOF stacking and is the accepted cost — in-sample residuals would be
> far worse (CLAUDE.md Critical Design Decision). It is also a plausible contributor to the
> negative result below.

#### Selected hyperparameters

Both models tuned by the **same** 54-point grid and the same chronological internal
holdout (last 15% of each model's own training data by date, refit on full data
afterwards). Neither touched the official val split.

| Model | Train rows | max_depth | n_estimators | learning_rate | min_child_weight | Holdout MAE / RMSE |
|---|---:|---:|---:|---:|---:|---|
| XGBoost residual | 552 | 5 | 300 | 0.01 | 3 | 380,051 / 563,793 |
| XGBoost alone | 889 | 5 | 100 | 0.05 | 10 | 275,152 / 382,713 |

XGBoost-alone deliberately trains on the **full 889-row** Phase 3 split, not the 552-row
residual subset: it has no dependency on the LSTM's OOF folds, so those exclusions do not
apply to it. Handicapping it "for fairness" would have starved the control of a third of
its data for an unrelated reason. Feature matrix and tuning procedure match; training-set
size legitimately does not. Pinned by `test_xgboost_alone_trains_on_the_full_889_row_split`.

Top residual-model feature by gain is **`feat_day_type_festivo` (0.125)** — stage 2 does
key on exactly the calendar signal stage 1 is structurally blind to, which confirms the
mechanism even though the end result below is negative.

#### MASTER COMPARISON

**TEST split** (2026-01-18 → 2026-08-02, n=197):

| Model | RMSE | MAE | MAPE | R² |
|---|---:|---:|---:|---:|
| **xgboost_alone** | **234,154** | **156,121** | **3.30%** | **0.9690** |
| sarimax | 241,147 | 163,627 | 3.85% | 0.9672 |
| hybrid | 328,131 | 234,223 | 5.49% | 0.9392 |
| lstm_alone | 431,714 | 280,952 | 6.62% | 0.8948 |
| seasonal_naive | 690,480 | 309,817 | 7.16% | 0.7308 |
| moving_average_7 | 1,260,492 | 1,093,679 | 28.13% | 0.1029 |
| moving_average_28 | 1,301,939 | 1,119,987 | 28.74% | 0.0430 |
| persistence | 1,390,444 | 900,570 | 21.79% | −0.0916 |

**VAL split** (n=196):

| Model | RMSE | MAE | MAPE | R² |
|---|---:|---:|---:|---:|
| **xgboost_alone** | **355,027** | **244,843** | **5.68%** | **0.9420** |
| sarimax | 404,153 | 273,467 | 7.31% | 0.9248 |
| hybrid | 415,562 | 280,738 | 7.09% | 0.9205 |
| lstm_alone | 655,479 | 411,142 | 11.81% | 0.8023 |
| seasonal_naive | 883,627 | 460,615 | 12.63% | 0.6408 |

**TRAIN split** (n=889, in-sample for the learned models — reference only):
xgboost_alone 114,865 / 72,903 / 1.77% / 0.9929 · sarimax 411,494 / 241,982 / 6.56% / 0.9088 ·
seasonal_naive 886,040 / 439,381 / 11.61% / 0.5772 · persistence 1,385,175 / 921,712 / 24.03% / −0.0332.
XGBoost-alone's near-perfect training fit against a 156k test MAE shows heavy in-sample
memorisation; the test number is the one that counts.

#### THE CENTRAL EMPIRICAL CLAIM — ANSWERED HONESTLY

**Does the hybrid beat SARIMAX on val AND test? NO. It beats it on neither.**

| | val MAE | test MAE |
|---|---:|---:|
| hybrid | 280,738 | 234,223 |
| sarimax | 273,467 | 163,627 |
| **hybrid − sarimax** | **+7,271 (worse)** | **+70,596 (worse)** |

**Does the hybrid beat XGBoost-alone? NO**, and by a wider margin (val +35,895, test
+78,102 MAE). **The residual-correction design is not justified by these results.**

What the residual stage *does* achieve is a large improvement over its own stage 1:
val MAE 411,142 → 280,738 (**−31.7%**), test 280,952 → 234,223 (**−16.6%**). So stage 2
works as designed — it recovers much of the calendar information stage 1 cannot see — but
it starts from a stage 1 so weak that the corrected result still lands behind a plain
SARIMAX and well behind a plain XGBoost on the same features.

#### Day-type breakdown — TEST split (MAE)

| Group | n | sarimax | lstm_alone | xgboost_alone | hybrid |
|---|---:|---:|---:|---:|---:|
| laborable, ordinary | 133 | **148,297** | 247,639 | 169,488 | 211,581 |
| laborable, bridge day | 3 | 247,865 | 176,372 | **185,671** | 399,637 |
| sabado | 27 | 203,910 | 270,915 | **129,381** | 247,793 |
| domingo | 29 | 169,129 | 228,996 | **100,200** | 233,778 |
| festivo | 5 | 271,410 | 1,585,388 | **251,553** | 666,572 |

**Does the hybrid specifically reduce festivo/bridge-day error relative to SARIMAX and
XGBoost-alone? NO — it is worse than both on every group except versus lstm_alone.**

The one genuinely positive finding: on *festivo* days the residual stage cuts stage 1's
error from **1,585,388 → 666,572 (−58%)**, the single largest improvement anywhere in this
project. The mechanism the architecture was designed around is real and measurable. It
simply is not sufficient — 666,572 is still 2.5x SARIMAX's 271,410 on those days.

> ⚠️ **Sample sizes on the interesting groups are tiny: festivo n=5, bridge day n=3.**
> Per-group test numbers for those two rows are indicative only and must be reported with
> their n in the memoria. Do not build a headline claim on 3 observations. The Phase 4
> step-0 diagnostic (n=54 bridge days, n=48 festivo over the full period) is the sounder
> basis for statements about *where* error concentrates.

#### Why the hybrid loses — for the memoria's discussion chapter

1. **Stage 1 is the bottleneck.** A univariate LSTM on ~860 daily observations cannot see
   the calendar, and Phase 4 showed it losing even to seasonal-naive out-of-fold. Stage 2
   must then correct a residual with std 726,669 — nearly the scale of the signal itself.
2. **Stage 2 solves a harder problem than XGBoost-alone.** The control learns
   `total` directly from clean features; the residual model must learn the *error of a
   weak model*, which carries the stage-1 noise on top of the calendar structure.
3. **552 training rows vs 889.** The OOF exclusions cost the residual model 38% of the
   available history — a structural tax the control does not pay.
4. **OOF distribution shift** (see the limitation box above).

**Honest framing: the negative result is a finding, not a failure of execution.** The
comparison is like-for-like on features, tuning, and splits, and every leakage guard from
Phases 2-4 is still enforced (asserted in `assemble_residual_dataset` and pinned by tests).
The defensible thesis conclusion is that for daily aggregate demand at this sample size, a
direct gradient-boosted model on well-constructed exogenous features outperforms a
sequential residual hybrid — and that the hybrid's benefit is confined to rescuing a weak
temporal stage rather than beating strong single-stage models.

#### Artifacts written

- `data/processed/residual_training_set.parquet` — 552 × (5 + 161)
- `data/processed/hybrid_predictions.parquet` — date, y_true, y_pred_lstm, e_pred_xgb, y_pred_hybrid
- `data/processed/full_comparison.parquet` — master table, all splits
- `models/xgboost_residual.joblib`, `models/xgboost_alone.joblib`


---

### Phase 5b — Sensitivity Analysis (bounded, pre-registered, non-exhaustive)

> **SCOPE, STATED UP FRONT.** This phase adds exactly **two** documented data points to the
> discussion chapter. It does **not** reopen Phase 5's headline conclusion, and it is **not**
> a search for a configuration that reverses it. Success criteria were fixed and written
> down **before** any result was seen. No third variant, hyperparameter sweep, or
> architecture change was run on the basis of how these turned out.
>
> **DOES THE PHASE 5 CENTRAL CLAIM STILL STAND? YES.** The hybrid still loses to SARIMAX
> and to XGBoost-alone on both val and test, in both its original and its reweighted form.
> It is now also beaten by a trivial two-model ensemble.

| Module | Responsibility |
|---|---|
| `src/models/hybrid_residual/xgboost_residual_weighted.py` | Experiment A |
| `src/evaluation/ensemble_baseline.py` | Experiment B |
| `src/evaluation/sensitivity_report.py` | Extended tables + mechanical verdict on A |
| `tests/test_sensitivity_analysis.py` | 17 tests |

#### Pre-registered criteria

- **Experiment A succeeds ONLY IF** test MAE on festivo + bridge-day rows falls, **AND**
  overall test MAE is no more than **5%** worse than Phase 5's hybrid (234,223 → ceiling
  245,934).
- **Experiment B is not pass/fail** — a required comparison point wherever it lands.

The criterion for A is evaluated **in code** (`evaluate_criterion_a`), not by eye, so the
verdict cannot drift to fit the outcome.

#### Experiment A — upweight irregular-calendar rows 3x. **CRITERION NOT MET.**

Identical to Phase 5's Stage 2 in every respect except `sample_weight`: same 552-row
residual set, same 161 features, same seed, and hyperparameters **loaded from the Phase 5
artifact** rather than re-declared (asserted equal, so drift on either side fails loudly).
Weight 3.0 was fixed **a priori** as a moderate upweighting — not selected by trying
several values. 45 of 552 rows (8.2%) were upweighted.

| Group | n | hybrid MAE | hybrid_weighted MAE | change |
|---|---:|---:|---:|---|
| **festivo** | 5 | 666,572 | **677,429** | **worse** |
| **laborable, bridge day** | 3 | 399,637 | **432,066** | **worse** |
| sabado | 27 | 247,793 | 185,918 | better |
| domingo | 29 | 233,778 | 243,096 | worse |
| laborable, ordinary | 133 | 211,581 | 200,326 | better |

- Irregular-day MAE (n-weighted over festivo + bridge): **566,472 → 585,418 — NOT improved.**
- Overall test MAE: 234,223 → **220,286 (−5.95%) — within budget.**
- **VERDICT: criterion NOT MET** (the budget clause passed; the targeted-improvement clause
  failed, and both were required).

**The interesting part of this null result:** upweighting the irregular days made the
overall model ~6% better while making *the very days it targeted* worse. The gains landed
on sabado (−25%) and ordinary working days (−5%) instead. The most plausible reading is
that with only 45 upweighted training rows against 161 features, the extra weight does not
buy a better fit on those rows; it perturbs the fit everywhere, and the perturbation
happened to help the populous groups. Reported as-is — this is a data point about the
fragility of small-sample reweighting, not evidence for the weighting scheme.

*Possible future work, named rather than run:* weight values other than 3.0; a
festivo-specific model; class-balanced objectives; or a quantile loss less dominated by the
bulk of ordinary days.

#### Experiment B — ensemble of SARIMAX + XGBoost-alone. **Beats both components.**

No retraining; both variants are pure arithmetic on predictions already saved in Phases 3
and 5.

| Variant | sarimax weight | xgboost_alone weight |
|---|---:|---:|
| ensemble_equal | 0.5000 | 0.5000 |
| ensemble_inverse_mae | 0.4724 | 0.5276 |

| Split | Model | RMSE | MAE | MAPE | R² |
|---|---|---:|---:|---:|---:|
| val | ensemble_inverse_mae | 323,348 | **226,418** | 5.62% | 0.9519 |
| val | ensemble_equal | 324,752 | 227,588 | 5.69% | 0.9515 |
| val | xgboost_alone | 355,027 | 244,843 | 5.68% | 0.9420 |
| val | sarimax | 404,153 | 273,467 | 7.31% | 0.9248 |
| test | ensemble_inverse_mae | 208,339 | **139,681** | 3.11% | 0.9755 |
| test | ensemble_equal | 208,464 | 139,725 | 3.13% | 0.9755 |
| test | xgboost_alone | 234,154 | 156,121 | 3.30% | 0.9690 |
| test | sarimax | 241,147 | 163,627 | 3.85% | 0.9672 |

**Does it beat SARIMAX and XGBoost-alone individually, or merely average between them?
It beats both, decisively.** Test MAE 139,681 is **10.5% below** the better component
(xgboost_alone 156,121) and 14.6% below SARIMAX — well outside the range simple averaging
of errors would produce. This is classic variance reduction from decorrelated errors, and
Phase 5's day-type table predicted it: SARIMAX is stronger on ordinary working days
(148,297 vs 169,488) while XGBoost-alone is much stronger on weekends and holidays
(domingo 100,200 vs 169,129). Averaging models that fail on *different* days recovers both
strengths.

The two variants are near-identical (test MAE differs by 44, ~0.03%), so the inverse-MAE
weighting adds nothing over a plain 50/50 — unsurprising given the weights land at
0.47/0.53. **Prefer `ensemble_equal` for the memoria**: same performance, no tuning step to
justify, and no dependence on validation data.

> **Caveat on `ensemble_inverse_mae`:** its weights are derived from validation MAE, so its
> **val** figure is mildly optimistic (the weights were chosen using the rows being scored).
> Its **test** figure is clean. Cite the test number, or the val number with this attached.

#### Extended master comparison — TEST split (n=197)

| Model | RMSE | MAE | MAPE | R² |
|---|---:|---:|---:|---:|
| **ensemble_inverse_mae** | **208,339** | **139,681** | **3.11%** | **0.9755** |
| ensemble_equal | 208,464 | 139,725 | 3.13% | 0.9755 |
| xgboost_alone | 234,154 | 156,121 | 3.30% | 0.9690 |
| sarimax | 241,147 | 163,627 | 3.85% | 0.9672 |
| hybrid_weighted [5b-A] | 319,447 | 220,286 | 5.17% | 0.9424 |
| hybrid | 328,131 | 234,223 | 5.49% | 0.9392 |
| lstm_alone | 431,714 | 280,952 | 6.62% | 0.8948 |
| seasonal_naive | 690,480 | 309,817 | 7.16% | 0.7308 |
| moving_average_7 | 1,260,492 | 1,093,679 | 28.13% | 0.1029 |
| moving_average_28 | 1,301,939 | 1,119,987 | 28.74% | 0.0430 |
| persistence | 1,390,444 | 900,570 | 21.79% | −0.0916 |

**VAL split (n=196):** ensemble_inverse_mae 323,348 / 226,418 / 5.62% / 0.9519 ·
ensemble_equal 324,752 / 227,588 / 5.69% / 0.9515 · xgboost_alone 355,027 / 244,843 / 5.68% /
0.9420 · sarimax 404,153 / 273,467 / 7.31% / 0.9248 · hybrid_weighted 401,456 / 276,347 /
6.94% / 0.9258 · hybrid 415,562 / 280,738 / 7.09% / 0.9205 · lstm_alone 655,479 / 411,142 /
11.81% / 0.8023.

Note the val ordering of `hybrid_weighted` (276,347) and `sarimax` (273,467) inverts on
RMSE (401,456 vs 404,153) — they are effectively tied on val, and neither ranking should be
cited as a separation.

#### Extended day-type breakdown — TEST split (MAE)

| Group | n | sarimax | xgboost_alone | hybrid | hybrid_weighted | ensemble_equal |
|---|---:|---:|---:|---:|---:|---:|
| laborable, ordinary | 133 | 148,297 | 169,488 | 211,581 | 200,326 | **138,628** |
| laborable, bridge day | 3 | 247,865 | **185,671** | 399,637 | 432,066 | 216,768 |
| sabado | 27 | 203,910 | 129,381 | 247,793 | 185,918 | **138,805** |
| domingo | 29 | 169,129 | **100,200** | 233,778 | 243,096 | 122,591 |
| festivo | 5 | 271,410 | 251,553 | 666,572 | 677,429 | **227,037** |

The ensemble is best on three of five groups and materially better than either component on
festivo (227,037 vs 251,553 / 271,410). Neither hybrid variant is competitive on any group.

> ⚠️ **festivo n=5 and bridge-day n=3.** These two rows remain far too small to support a
> headline claim; cite them with their n, and use the Phase 4 step-0 diagnostic (n=48 and
> n=54 over the full period) for statements about where error concentrates.

#### What Phase 5b changes, and what it does not

- **Unchanged:** the hybrid architecture is not competitive. Both variants sit behind
  SARIMAX, XGBoost-alone, and the ensemble on both splits.
- **Unchanged:** Stage 2 demonstrably works *as a correction* — it still rescues Stage 1 by
  a wide margin — but Stage 1 is too weak a base for that to matter.
- **New:** the best model in this project is now a trivial 50/50 ensemble of two
  single-stage models, requiring no LSTM at all.
- **New:** targeted reweighting of rare calendar days did not improve those days, a
  small-sample fragility result worth one paragraph in the discussion.

#### Artifacts written

- `models/xgboost_residual_weighted.joblib`
- `data/processed/hybrid_weighted_predictions.parquet`
- `data/processed/ensemble_predictions.parquet`
- `data/processed/sensitivity_comparison.parquet`


---

### Phase 6 — Results Dashboard

**No model is trained, refit, or re-predicted in this phase.** Every number and every figure
is read from artifacts already written by Phases 3-5b, so the dashboard cannot disagree with
the tables above. `tests/test_dashboard_data.py` pins the reported test MAEs
(ensemble_equal 139,725 · xgboost_alone 156,121 · sarimax 163,627 · hybrid 234,223 ·
lstm_alone 280,952) against the dashboard's own computation, so a reshaping bug in the
reporting layer fails a test instead of silently producing a different chart.

| Module | Responsibility |
|---|---|
| `src/evaluation/dashboard_data.py` | Loading + aggregation (no plotting) |
| `src/evaluation/dashboard_figures.py` | Pure `go.Figure` builders (no I/O, no display) |
| `src/evaluation/export_figures.py` | Static PNG export, run as a standalone script |
| `notebooks/06_dashboard.ipynb` | Widgets + rendering only, zero business logic |
| `tests/test_dashboard_data.py` | 40 tests |

#### `dashboard_data.py` — public functions

| Function | Returns |
|---|---|
| `load_all_predictions()` | Tidy long frame `[date, split, model, y_true, y_pred]`, 11 models |
| `available_models(split)` | Models with actual coverage on that split |
| `model_comparison_table(split)` | `[model, family, n, MAE, RMSE, MAPE, R2]`, sorted by MAE |
| `error_by_day_type(split)` | `[model, group, n, MAE, MAPE, small_sample]`, **all** models |
| `residual_distribution(model, split)` | `{split: Series, 'train': Series}` for overfit-gap plots |
| `feature_importance_by_group(model)` | `{'by_group': …, 'top_features': …}` for either XGBoost model |
| `actual_vs_predicted(model, split)` | `[date, y_true, y_pred]` for line charts |
| `classify_feature(name)` | One of the 7 groups; **raises** on an unmapped column |

Every loader routes through `_require()`, which raises `FileNotFoundError` naming both the
missing path and the phase command that produces it, so a missing artifact gives an
actionable message rather than a bare pandas error.

> **COVERAGE IS NOT UNIFORM — do not assume it is.** The baselines, `xgboost_alone` and the
> ensembles cover train/val/test. The LSTM-derived models (`lstm_alone`, `hybrid`,
> `hybrid_weighted`) exist **only on val and test**, because Phase 4 saved stage-1
> predictions for those splits alone. Functions return what exists rather than padding
> with NaN; `available_models(split)` reports it.

**Feature grouping** is an explicit ordered constant (`FEATURE_GROUP_PATTERNS`), not
inferred per call: `fourier_weekly`, `fourier_annual`, `calendar`, `lag_total`, `rolling`,
`lag_operator`, `weather`. Order matters because weather columns also carry `_lag_` and
`_roll_mean_` suffixes, so the target-specific patterns must be tested first. A column
matching nothing raises rather than vanishing from the group totals.

#### Notebook sections (`notebooks/06_dashboard.ipynb`, 13 cells, executes clean)

| § | Section | Controls |
|---|---|---|
| a | Setup + `pip install -e .` prerequisite | — |
| b | **Demanda real vs. predicha** | model + split dropdowns, `Mostrar train` checkbox |
| c | **Comparación entre modelos** | split dropdown; colour = model family |
| d | **Error de predicción** | model dropdown; train-vs-test overfit panel |
| e | **Desglose de error por tipo de día** | split dropdown; n<10 groups dimmed |
| f | **Importancia de variables (XGBoost)** | model dropdown (residual vs alone) |
| g | **Resumen ejecutivo** (Spanish, markdown) | — |

Train is **excluded from §b by default** and, when revealed, the figure title is stamped
`⚠ IN-SAMPLE`: it is in-sample for every learned model and a screenshot of it would
otherwise misrepresent performance. §e dims groups with n<10 and prints n under each tick,
carrying the festivo (n=5) / bridge (n=3) caveat into the visualization instead of leaving
it in prose.

#### Exported figures — `reports/figures/` (13 PNG, scale=2)

```
demand_actual_vs_predicted_test.png        model_comparison_bar_test.png
demand_actual_vs_predicted_val.png         model_comparison_bar_val.png
residual_distribution_xgboost_alone.png    error_by_day_type_test.png
residual_distribution_hybrid.png           error_by_day_type_val.png
residual_distribution_ensemble_equal.png
feature_importance_group_xgboost_residual.png   feature_importance_top15_xgboost_residual.png
feature_importance_group_xgboost_alone.png      feature_importance_top15_xgboost_alone.png
```

Regenerate with `python -m src.evaluation.export_figures`.

#### Two environment gotchas worth not rediscovering

> ⚠️ **kaleido must be 1.x, not 0.2.1.** plotly 6.x dropped the v0 API, and
> `fig.write_image()` against kaleido 0.2.1 **hangs indefinitely without raising**. Pinned
> `kaleido==1.3.0`, which drives a headless Chrome.

> ⚠️ **Do NOT export PNGs from inside the notebook.** kaleido 1.x deadlocks when driven
> repeatedly from a Jupyter kernel — observed here: the first two exports succeeded, then
> the kernel blocked for 11+ minutes at near-zero CPU with no error and no traceback. The
> identical calls complete in seconds in a plain sequential process. That is why export
> lives in `export_figures.py` and not in a notebook cell. Both paths use the same builders
> from `dashboard_figures.py`, so the PNGs and the interactive figures cannot drift.

> ⚠️ **Do NOT set `pio.renderers.default = "notebook_connected"`.** That renderer fetches
> plotly.js from a CDN and blocks headless execution (`nbconvert`) silently. The default
> mimetype renderer works in both Jupyter Lab and nbconvert.

#### Packaging

`pyproject.toml` was added this phase so `pip install -e .` makes `src` importable from
`notebooks/` without `sys.path` manipulation. `requirements.txt` remains the single source
of truth for pinned runtime dependencies; the new entries are `ipywidgets==8.1.8` and
`kaleido==1.3.0`.

#### Verification

- **217/217 tests pass** (40 new in `tests/test_dashboard_data.py`).
- `jupyter nbconvert --to notebook --execute` completes with **0 cell errors**, 6/6 code
  cells executed.
- 13 figures written to `reports/figures/`.


---

### Phase 6b — Visual Refinement

> **PRESENTATION ONLY.** This phase changed styling, table formatting, and notebook
> structure. **No data logic, no metric computation, and no model artifact was touched** —
> every number in every figure and table is the same value reported in Phases 3-5b, still
> produced by the same `metrics.py` code path. The Phase 6 tests that pin the reported
> MAEs against the dashboard's own computation all still pass unchanged.

| Module | Responsibility |
|---|---|
| `src/evaluation/theme.py` | **Single source of truth** for every colour, font and template |
| `src/evaluation/report_tables.py` | Spanish, rounded, copy-pasteable DataFrames |
| `src/evaluation/dashboard_figures.py` | Now themed + `caption()` helper (data logic unchanged) |
| `src/evaluation/export_figures.py` | Now verifies the shared theme before each write |
| `tests/test_theme.py` | 40 tests |

#### Palette — `VIU_COLORS`

| Role | Hex | Note |
|---|---|---|
| `primary` | `#E8590C` | ⚠ **APPROXIMATION of VIU's brand orange** |
| `secondary` | `#4A3A52` | muted purple, secondary/digital accent |
| `white` | `#FFFFFF` | |
| `gray` / `gray_light` / `gray_pale` | `#8A8F98` / `#D9DCE1` / `#EDEFF2` | |
| `near_black` | `#1F2328` | body text, actual-demand line |
| `steel` / `sand` / `teal` | `#3D5A80` / `#C9A227` / `#2E6E6B` | support tones |

> ⚠️ **THE ORANGE IS NOT THE OFFICIAL VIU COLOUR.** The exact Pantone/HEX from VIU's brand
> manual was not available. `VIU_COLORS["primary"]` is **the single line to edit** when it
> is supplied — nothing else in the codebase hard-codes an orange, and both the notebook
> and the exported PNGs pick the change up automatically. `test_primary_is_the_documented_approximate_orange`
> pins the current value so replacing it is a deliberate, visible change.

**Typography:** a system sans-serif stack, **not** VIU's corporate typefaces. Periodico
Display and Visuelt Pro are proprietary and are not assumed installed — critically, the
headless Chrome that kaleido drives would silently substitute a fallback and make exported
PNGs differ from the on-screen figures.

**Backgrounds are explicitly white** (both paper and plot area) because these figures are
destined for a Google Docs chapter, where a dark or transparent figure pastes as a broken
block.

#### Accessibility — a real defect found and fixed

The previous palette distinguished the **LSTM family (`#d62728` red)** from the **ensemble
family (`#2ca02c` green)** by hue alone. That is the worst possible pair for deuteranopia
and protanopia (~8% of men), and it is exactly the pair a reader of the results chapter
must tell apart. The replacement separates families along **orange / steel blue / dark
purple / light grey**.

A second defect surfaced from the greyscale test: at the originally chosen `#6B4E71`, the
purple's relative luminance (86.7) sat within **0.2** of steel blue (86.6) — indistinguishable
in a black-and-white print despite reading as clearly different hues on screen. The
secondary was darkened to `#4A3A52`, giving a minimum family luminance gap of ~23.
Pinned by `test_family_colours_differ_in_luminance_for_greyscale_printing`.

#### Shared-theme guarantee, enforced not assumed

- `dashboard_figures.py` and `export_figures.py` both import `VIU_TEMPLATE` from
  `theme.py`; tests assert they are **the same object**, not merely equal.
- Every builder returns `apply_theme(fig)`; a parametrised test checks all six.
- `export_figures.assert_theme_applied()` runs before **every** `write_image`, comparing
  the figure's colorway against `theme_fingerprint()`. A builder that forgot the theme, or
  a second palette defined elsewhere, fails the export instead of producing a silently
  off-brand PNG.
- A test greps `dashboard_figures.py` for literal hex colours (comments stripped) and fails
  if any reappear.

#### `report_tables.py` — plain DataFrames, deliberately

`comparison_table`, `day_type_table`, `hyperparameters_table`, `feature_importance_table`,
`group_importance_table`. Spanish headers, sorted best-to-worst by MAE to match
`comparison_figure`'s ordering.

**They return plain `DataFrame`s, never `.style` objects.** A pandas Styler renders as
CSS-heavy HTML that Google Docs pastes as an image or mangled markup; a plain DataFrame
pastes as a **native, editable table with selectable text**. Cell gradients are not worth
losing that.

> ⚠️ **`R2_DECIMALS` defaults to 4, not 2. MAPE remains at 2.** Rounding is whole
> passengers for MAE/RMSE, **two decimals for MAPE, four for R²**.
>
> The asymmetry is deliberate and load-bearing. At two decimals the test-split R² of
> **`ensemble_equal` (0.9755)**, **`xgboost_alone` (0.9690)** and **`sarimax` (0.9672)** all
> collapse to **0.98 / 0.97 / 0.97** — and the ordering of those three models *is* the
> thesis's central empirical claim (the ensemble beats both single-stage models; the hybrid
> beats neither). A results table that renders them as indistinguishable erases exactly the
> distinction the chapter exists to make. MAPE does not suffer this: at this project's scale
> the same three models read 3.13% / 3.30% / 3.85%, already well separated at two decimals.
>
> `comparison_table(split, r2_decimals=2)` remains available for a table that deliberately
> wants coarser figures. Pinned by `test_r2_default_is_four_decimals` and
> `test_default_r2_separates_the_three_leading_models`.

#### Notebook — 22 cells, additive changes only

The existing widget-driven interactive cells are **unchanged**. Added: one
"Cómo reutilizar este cuaderno en la memoria" markdown cell after setup, and after each
section a static table cell (where applicable) plus a blockquoted caption cell. Captions
are numbered Figura 1-6 and Tabla 1-5, and their wording is **pulled from the existing
notebook markdown** rather than rewritten, so caption and surrounding prose cannot
contradict each other. `caption()` in `dashboard_figures.py` is the programmatic source;
`test_caption_reuses_notebook_wording` pins the shared phrases.

#### Verification

- **257/257 tests pass** (40 new in `tests/test_theme.py`).
- All **13 PNGs re-exported**, each passing the shared-theme check.
- `jupyter nbconvert --to notebook --execute` completes with **0 cell errors**, 9/9 code
  cells executed.

