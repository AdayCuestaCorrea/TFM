# REPORT_MAPPING.md — Trazabilidad memoria ↔ código ↔ artefactos

Documento de referencia técnica, no de lectura para el tribunal. Enumera, para cada
figura y tabla de la memoria (`docs/LaTeX/TFT.tex`), la cadena completa
**fase del proyecto → módulo Python que la genera → artefacto persistido en disco →
fichero `.tex` que la incluye → capítulo de destino**. Su propósito es que un cambio en
un número de `CLAUDE.md` (Fases 0-6b) se pueda propagar a la memoria sin releer todo el
código: se localiza la fila, se re-ejecuta el generador citado, y se recompila.

Generado en la **Fase 11** del plan de modularización LaTeX. No documenta el contenido
narrativo de los capítulos (eso vive en el propio `.tex`); documenta de dónde sale cada
número y cada imagen.

---

## 1. Estructura del documento

```
docs/LaTeX/
├── TFT.tex                    orquestador: portada + \input de preamble y sections/*
├── preamble.tex                paquetes, colores VIU, comandos \headlinecolor, \tfttitulo
├── Bibliografia_TFT.bib        referencias biblatex-apa / biber, APA 7 (incidencia 21)
├── sections/
│   ├── 00_preliminares.tex     Resumen, Abstract, palabras clave/keywords, Agradecimientos
│   ├── 01_introduccion.tex     objetivos, alcance, cronograma (Gantt + tabla)
│   ├── 02_estado_arte.tex      revisión de literatura + tabla comparativa apaisada
│   ├── 03_datos_features.tex   fuentes de datos, esquema unificado, feature engineering
│   ├── 04_metodologia.tex      SARIMAX, LSTM Stage 1 (OOF), XGBoost Stage 2, hiperparámetros
│   ├── 05_resultados.tex       comparación de modelos, desglose por tipo de día, importancia
│   ├── 06_conclusiones.tex     síntesis del hallazgo central y trabajo futuro
│   └── 07_anexos.tex           catálogo de variables, suite de tests, reproducibilidad
├── tables/                     32 ficheros .tex generados por export_latex_tables.py
└── Images/                     activos estáticos de portada (no generados por código)
```

Todas las figuras se referencian con ruta relativa `../../reports/figures/*.png` desde
`sections/*.tex` (el compilador se invoca desde `docs/LaTeX/`).

---

## 2. Figuras — 34 PNG, `reports/figures/`

Generador único: `python -m src.evaluation.export_figures`. Internamente compone dos
módulos de builders (ambos devuelven `go.Figure` puros, sin I/O):

- `src/evaluation/dashboard_figures.py` — 13 figuras heredadas de la Fase 6/6b (dashboard).
- `src/evaluation/memoria_figures.py` — 21 figuras: 9 de la Fase 8 (serie completa, EDA,
  splits, diagnóstico OOF) + 7 de la fase diagnóstica 7 (anatomía del residuo de la
  etapa 1, §5.9) + 2 de la fase diagnóstica 8 (ablación de bloques y paridad LSTM, §5.10)
  + 3 de la fase diagnóstica 9 (desglose por subgrupos y periodos, §5.11).
  Las de las fases 7, 8 y 9 leen de parquets ya persistidos
  (`stage1_residual_anatomy.py`, `lstm_learning_curve.parquet`,
  `feature_block_ablation.parquet`, `lstm_parity_control.parquet`,
  `subgroup_breakdown.py`); ninguna importa `src.models`.

Todas aplican `VIU_TEMPLATE` (`src/evaluation/theme.py`); `export_figures.py` verifica el
tema antes de cada `write_image` (`assert_theme_applied`).

| PNG | Builder (`src/evaluation/...`) | Fuente de datos (`data/processed/...`) | Incluida en | Fase origen |
|---|---|---|---|---|
| `fig_serie_completa_total.png` | `memoria_figures.total_series_figure` | `features_daily.parquet` | `03_datos_features.tex` | 8 |
| `fig_demanda_por_operador.png` | `memoria_figures.operator_demand_figure` | `features_daily.parquet` | `03_datos_features.tex` | 8 |
| `fig_estacionalidad_semanal.png` | `memoria_figures.weekly_seasonality_figure` | `features_daily.parquet` | `03_datos_features.tex` | 8 |
| `fig_estacionalidad_anual.png` | `memoria_figures.annual_seasonality_figure` | `features_daily.parquet` | `03_datos_features.tex` | 8 |
| `fig_meteo_vs_demanda.png` | `memoria_figures.weather_vs_demand_figure` | `features_daily.parquet` | `03_datos_features.tex` | 8 |
| `fig_splits_cronologicos.png` | `memoria_figures.splits_figure` | `features_daily.parquet` + `src/utils/splits.py` | `03_datos_features.tex` | 8 |
| `fig_oof_folds.png` | `memoria_figures.oof_folds_figure` | `lstm_oof_predictions.parquet`, `lstm_oof_fold_log.parquet` | `04_metodologia.tex` | 8 |
| `fig_oof_fold1_degenerado.png` | `memoria_figures.oof_fold1_degenerate_figure` | `lstm_oof_predictions.parquet` | `04_metodologia.tex` | 8 |
| `demand_actual_vs_predicted_test.png` | `dashboard_figures.demand_figure("test")` | `dashboard_data.load_all_predictions` | `05_resultados.tex` | 6 |
| `demand_actual_vs_predicted_val.png` | `dashboard_figures.demand_figure("val")` | idem | `05_resultados.tex` | 6 |
| `model_comparison_bar_test.png` | `dashboard_figures.comparison_figure("test")` | `dashboard_data.model_comparison_table` | `05_resultados.tex` | 6 |
| `model_comparison_bar_val.png` | `dashboard_figures.comparison_figure("val")` | idem | `05_resultados.tex` | 6 |
| `error_by_day_type_test.png` | `dashboard_figures.day_type_figure("test")` | `dashboard_data.error_by_day_type` | `05_resultados.tex` | 6 |
| `error_by_day_type_val.png` | `dashboard_figures.day_type_figure("val")` | idem | `05_resultados.tex` | 6 |
| `residual_distribution_xgboost_alone.png` | `dashboard_figures.residual_figure("xgboost_alone")` | `dashboard_data.residual_distribution` | `05_resultados.tex` | 6 |
| `residual_distribution_hybrid.png` | `dashboard_figures.residual_figure("hybrid")` | idem | `05_resultados.tex` | 6 |
| `residual_distribution_ensemble_equal.png` | `dashboard_figures.residual_figure("ensemble_equal")` | idem | `05_resultados.tex` | 6 |
| `feature_importance_group_xgboost_alone.png` | `dashboard_figures.importance_group_figure("xgboost_alone")` | `dashboard_data.feature_importance_by_group` | `05_resultados.tex` | 6 |
| `feature_importance_top15_xgboost_alone.png` | `dashboard_figures.importance_top_figure("xgboost_alone")` | idem | `05_resultados.tex` | 6 |
| `feature_importance_group_xgboost_residual.png` | `dashboard_figures.importance_group_figure("xgboost_residual")` | idem | `05_resultados.tex` | 6 |
| `feature_importance_top15_xgboost_residual.png` | `dashboard_figures.importance_top_figure("xgboost_residual")` | idem | `05_resultados.tex` | 6 |
| `fig_dispersion_errores_sarimax_xgb.png` | `memoria_figures.error_dispersion_figure` | `dashboard_data.load_all_predictions` (sarimax vs xgboost_alone, test) | `05_resultados.tex` | 8 |
| `fig_acf_pacf_residuo_lstm.png` | `memoria_figures.stage1_acf_pacf_figure` | `stage1_residual_anatomy.residual_autocorrelation` ← `lstm_val_test_predictions.parquet` | `05_resultados.tex` (§5.9) | 7 |
| `fig_periodograma_residuo_lstm.png` | `memoria_figures.stage1_periodogram_figure` | `stage1_residual_anatomy.residual_periodogram` ← idem | `05_resultados.tex` (§5.9) | 7 |
| `fig_residuo_por_dia_semana.png` | `memoria_figures.stage1_weekday_figure` | `stage1_residual_anatomy.weekday_residual_profile` ← idem + `features_daily.parquet` | `05_resultados.tex` (§5.9) | 7 |
| `fig_residuo_vs_meteo.png` | `memoria_figures.stage1_weather_corr_figure` | `stage1_residual_anatomy.residual_weather_correlation` ← idem + `features_daily.parquet` | `05_resultados.tex` (§5.9) | 7 |
| `fig_cadena_error.png` | `memoria_figures.stage1_error_chain_figure` | `stage1_residual_anatomy.error_chain_table` ← `dashboard_data.load_all_predictions` | `05_resultados.tex` (§5.9) | 7 |
| `fig_fold_size_vs_error.png` | `memoria_figures.stage1_fold_size_figure` | `stage1_residual_anatomy.fold_size_vs_error` ← `lstm_oof_*`, `baseline_predictions.parquet` | `05_resultados.tex` (§5.9) | 7 |
| `fig_curva_aprendizaje_lstm.png` | `memoria_figures.stage1_learning_curve_figure` | `lstm_learning_curve.parquet` + `dashboard_data.model_comparison_table("val")` | `05_resultados.tex` (§5.9) | 7 |
| `fig_ablacion_bloques.png` | `memoria_figures.block_ablation_figure` | `feature_block_ablation.parquet` | `05_resultados.tex` (§5.10.1) | 8 |
| `fig_paridad_lstm.png` | `memoria_figures.parity_control_figure` | `lstm_parity_control.parquet` + `lstm_learning_curve.parquet` + `dashboard_data.model_comparison_table("val")` | `05_resultados.tex` (§5.10.2) | 8 |
| `fig_subgrupos_mae.png` | `memoria_figures.subgroup_mae_figure("test")` | `subgroup_breakdown.subgroup_metrics` ← `dashboard_data.load_all_predictions` + `features_daily.parquet` | `05_resultados.tex` (§5.11.2) | 9 |
| `fig_subgrupos_delta_ic.png` | `memoria_figures.subgroup_delta_ci_figure("test")` | `subgroup_breakdown.bootstrap_delta` + `hybrid_win_table` ← idem | `05_resultados.tex` (§5.11.3) | 9 |
| `fig_mae_movil_periodo.png` | `memoria_figures.period_rolling_mae_figure` | `subgroup_breakdown.rolling_mae` ← idem + `src/utils/splits.py` | `05_resultados.tex` (§5.11.4) | 9 |

> ⚠️ **PNG export debe correr FUERA de un kernel de Jupyter** (kaleido 1.x deadlock —
> ver `CLAUDE.md`, Fase 6). `export_figures.py` es un script ejecutado como proceso
> secuencial plano, nunca desde `notebooks/06_dashboard.ipynb`.

---

## 3. Tablas — 32 `.tex`, `docs/LaTeX/tables/`

Generador único: `python -m src.evaluation.export_latex_tables`. Módulo:
`src/evaluation/export_latex_tables.py`. Convenciones compartidas por todas las tablas:

- Formato numérico español (`format_number`): punto de millar, coma decimal.
  MAE/RMSE enteros, MAPE 2 decimales, R² 4 decimales (asimetría deliberada, ver
  `CLAUDE.md` Fase 6b — a 2 decimales, `ensemble_equal`/`xgboost_alone`/`sarimax`
  colapsan a valores indistinguibles).
- `escape_latex` neutraliza `%`, `_`, `&`, `#` antes de volcar cualquier texto libre.
- Estilo `booktabs` (`\toprule/\midrule/\bottomrule`), cabecera en naranja VIU.
- Caracteres no-ASCII (`Δ`, `−` tipográfico) evitados deliberadamente: LaTeX con la
  codificación de este documento no los tiene declarados y rompen la compilación
  (incidente Fase 10, ver §5). `Δ MAE` se escribe como columna `Delta MAE`.

| Tabla `.tex` | Builder (`export_latex_tables.build_...`) | Fuente | Incluida en | Fase origen |
|---|---|---|---|---|
| `tabla_cronograma_fases.tex` | `build_cronograma_fases` | constante `PHASES` (hardcoded, refleja `CLAUDE.md`) | `01_introduccion.tex` | 9 |
| `tabla_comparativa_estado_arte.tex` | `build_comparativa_estado_arte` | constante `LITERATURE_COMPARISON` | `02_estado_arte.tex` (apaisada, `pdflscape`) | 9 |
| `tabla_fuentes_datos.tex` | `build_fuentes_datos` | descripción estática de `data/raw/*` (Fase 1) | `03_datos_features.tex` | 9 |
| `tabla_esquema_unificado.tex` | `build_esquema_unificado` | esquema de `unified_daily.parquet` (Fase 1) | `03_datos_features.tex` | 9 |
| `tabla_grupos_features.tex` | `build_grupos_features` | `build_features.feature_columns` (Fase 2/3) | `03_datos_features.tex`, `07_anexos.tex` | 9 |
| `tabla_particiones.tex` | `build_particiones` | `src.utils.splits.chronological_split` | `03_datos_features.tex` | 9 |
| `tabla_sarimax_orden.tex` | `build_sarimax_orden` | `data/processed/sarimax_selected_order.json` | `04_metodologia.tex` | 9 |
| `tabla_ventana_lstm.tex` | `build_ventana_lstm` | `data/processed/lstm_window_comparison.parquet` | `04_metodologia.tex` | 9 |
| `tabla_folds_oof.tex` | `build_folds_oof` | `data/processed/lstm_oof_fold_log.parquet` | `04_metodologia.tex` | 9 |
| `tabla_hiperparametros.tex` | `build_hiperparametros` | `models/xgboost_residual.joblib`, `models/xgboost_alone.joblib` (params) | `04_metodologia.tex` | 9 |
| `tabla_comparacion_test.tex` | `build_comparacion("test")` | `report_tables.comparison_table("test")` → `full_comparison.parquet` | `05_resultados.tex` | 9 |
| `tabla_comparacion_val.tex` | `build_comparacion("val")` | idem, split val | `05_resultados.tex` | 9 |
| `tabla_desglose_dia_test.tex` | `build_desglose_dia("test")` | `report_tables.day_type_table` → `error_by_day_type` | `05_resultados.tex` | 9 |
| `tabla_desglose_dia_val.tex` | `build_desglose_dia("val")` | idem, split val | `05_resultados.tex` | 9 |
| `tabla_importancia_grupo_alone.tex` | `build_importancia_grupo("xgboost_alone")` | `report_tables.group_importance_table` | `05_resultados.tex` | 9 |
| `tabla_importancia_grupo_residual.tex` | `build_importancia_grupo("xgboost_residual")` | idem | `05_resultados.tex` | 9 |
| `tabla_top15_alone.tex` | `build_top15("xgboost_alone")` | `report_tables.feature_importance_table` | `05_resultados.tex` | 9 |
| `tabla_top15_residual.tex` | `build_top15("xgboost_residual")` | idem | `05_resultados.tex` | 9 |
| `tabla_delta_hibrido.tex` | `build_delta_hibrido` | `full_comparison.parquet` (hybrid vs sarimax/xgboost_alone) | `05_resultados.tex` | 9 |
| `tabla_criterio_experimento_a.tex` | `build_criterio_experimento_a` | `sensitivity_comparison.parquet` (Fase 5b, Exp. A) | `05_resultados.tex` | 9 |
| `tabla_ensemble_pesos.tex` | `build_ensemble_pesos` | `ensemble_predictions.parquet` metadata (Fase 5b, Exp. B) | `05_resultados.tex` | 9 |
| `tabla_autocorrelacion_residuo.tex` | `build_autocorrelacion_residuo` | `stage1_residual_anatomy.residual_autocorrelation` + `ljung_box` | `05_resultados.tex` (§5.9) | 7 |
| `tabla_residuo_dia_semana.tex` | `build_residuo_dia_semana` | `stage1_residual_anatomy.weekday_residual_profile` | `05_resultados.tex` (§5.9) | 7 |
| `tabla_residuo_calendario.tex` | `build_residuo_calendario` | `stage1_residual_anatomy.residual_by_calendar_group` | `05_resultados.tex` (§5.9) | 7 |
| `tabla_residuo_meteo.tex` | `build_residuo_meteo` | `stage1_residual_anatomy.residual_weather_correlation` (top 10) | `05_resultados.tex` (§5.9) | 7 |
| `tabla_curva_aprendizaje.tex` | `build_curva_aprendizaje` | `data/processed/lstm_learning_curve.parquet` | `05_resultados.tex` (§5.9) | 7 |
| `tabla_cadena_error.tex` | `build_cadena_error` | `stage1_residual_anatomy.error_chain_table` ← `metrics.all_metrics` | `05_resultados.tex` (§5.9) | 7 |
| `tabla_ablacion_bloques.tex` | `build_ablacion_bloques` | `data/processed/feature_block_ablation.parquet` (`completo` ← `full_comparison.parquet`) | `05_resultados.tex` (§5.10.1) | 8 |
| `tabla_paridad_lstm.tex` | `build_paridad_lstm` | `data/processed/lstm_parity_control.parquet` | `05_resultados.tex` (§5.10.2) | 8 |
| `tabla_subgrupos_test.tex` | `build_subgrupos("test")` | `subgroup_breakdown.hybrid_win_table` + `evaluate_subgroup_criteria` | `05_resultados.tex` (§5.11.2) | 9 |
| `tabla_subgrupos_val.tex` | `build_subgrupos("val")` | idem, split val (veredicto único, test-scoped) | `05_resultados.tex` (§5.11.2) | 9 |
| `tabla_subgrupos_inferencia.tex` | `build_subgrupos_inferencia` | `subgroup_breakdown.bootstrap_delta("test")` (bloque móvil 7, 2000 remuestreos, BH q=0,10) | `05_resultados.tex` (§5.11.3) | 9 |

---

## 4. Correspondencia capítulo ↔ fase(s) del proyecto (`CLAUDE.md`)

| Capítulo (`sections/*.tex`) | Fase(s) de `CLAUDE.md` que documenta |
|---|---|
| `01_introduccion.tex` | Fase 0 (andamiaje) + cronograma íntegro (Fases 0-6b) |
| `02_estado_arte.tex` | N/A — revisión bibliográfica, no ligada a una fase de código |
| `03_datos_features.tex` | Fase 1 (ingestión) + Fase 2 (features) + Fase 3 (splits, promoción meteo) |
| `04_metodologia.tex` | Fase 3 (SARIMAX) + Fase 4 (LSTM, OOF) + Fase 5 (hiperparámetros híbrido/alone) |
| `05_resultados.tex` | Fase 5 (comparación maestra) + Fase 5b (sensibilidad, ensemble) + Fase 6/6b (dashboard) + fase diagnóstica 7 (§5.9, anatomía del residuo de la etapa 1) + fase diagnóstica 8 (§5.10, paridad informativa y ablación de bloques) + fase diagnóstica 9 (§5.11, desglose por subgrupos y periodos) |
| `06_conclusiones.tex` | Síntesis transversal de Fases 5-5b (hallazgo central); §6.2 incluye la limitación de veredictos ambiguos de la fase 7 y la del veredicto ambiguo del control de paridad `completo` de la fase 8. La fase 9 no añade limitación nueva: su resultado (el híbrido no gana en ningún subgrupo, y es significativamente peor en los 9 inferenciales) refuerza el hallazgo central sin abrir una salvedad — la octava limitación estaba condicionada a un veredicto `primero sin evidencia`, que no se produjo |
| `07_anexos.tex` | Fase 2 (catálogo `feat_`) + suites de test de todas las fases |

Si `CLAUDE.md` cambia un número reportado (p. ej. tras una nueva ejecución de un
pipeline), la fila correspondiente en §2/§3 indica exactamente qué generador
re-ejecutar; no es necesario editar manualmente ningún `.tex` de `tables/`.

---

## 5. Incidencias de compilación resueltas (Fase 10-11)

Registradas aquí, no en `CLAUDE.md`, porque son defectos de la capa de presentación
LaTeX, no de los datos ni de los modelos.

1. **Carácter Unicode `Δ` (U+0394) no soportado.** La columna "Δ MAE" en
   `tabla_delta_hibrido.tex` y su mención en la leyenda de `05_resultados.tex` rompían
   la compilación (`! LaTeX Error: Unicode character Δ ... not set up`). Solución:
   renombrada a `Delta MAE` en `export_latex_tables.build_delta_hibrido` y en la
   leyenda; sin impacto en el valor numérico ni en los tests que la referencian.
2. **`Timestamp` de pandas no serializable a JSON por kaleido.** `memoria_figures.splits_figure`
   pasaba objetos `Timestamp` directamente a `fig.add_vrect(x0=..., x1=...)`; kaleido
   1.x serializa la figura a JSON antes de invocar Chrome headless y no sabe
   serializar `Timestamp`. Solución: `.isoformat()` antes de pasarlos a Plotly.
3. **Gantt (`pgfgantt`) desbordaba el margen derecho de la página** con `x unit=1.3cm`
   sobre 10 columnas. Solución: `x unit=1.0cm` + `\resizebox{\textwidth}{!}{...}`
   envolviendo el entorno `ganttchart`, y título de cabecera acortado (el detalle
   "0 = Andamiaje, 9 = Redacción" se movió al pie de figura).
4. **Bibliografía vacía tras la primera compilación (`LaTeX Warning: Empty
   \thebibliography environment`).** Esperado en este punto: `Bibliografia_TFT.bib`
   tiene entradas, pero la prosa de los capítulos aún no incluye comandos `\cite{}`
   apuntando a ellas. No es un defecto de Fase 10-11; se resuelve al redactar la
   prosa final (fuera del alcance de este plan, ver `CLAUDE.md` "Next").
5. **Referencias `\pageref{LastPage}` indefinidas y `Overfull/Underfull \hbox`
   dispersos.** Advertencias, no errores — la primera se resuelve sola tras una
   segunda pasada de `latexmk` (ya incluida en su bucle estándar); los `hbox` son
   ajustes cosméticos de justificación en tablas/figuras anchas, pendientes de pulido
   visual pero sin efecto en el contenido.
6. **`@misc` dentro de un comentario `%%` en `Bibliografia_TFT.bib` abortaba
   BibTeX** (`I was expecting a '{' or a '('`). BibTeX no tiene comentarios de
   línea: cualquier `@tipo` en el fichero se interpreta como inicio de entrada,
   también dentro de una línea que empieza por `%%`. Solución: reformular el
   comentario de cabecera para que no contenga la secuencia `@misc` (se escribe
   "entrada de tipo misc"). Sin esto, `latexmk` reportaba 45 citas y 323
   referencias sin resolver.
7. **Desbordamiento horizontal y vertical de los cuadros 1.1 (cronograma), 2.1
   (comparativa apaisada), 4.2 (ventana LSTM), 4.3 (folds OOF) y 4.4
   (hiperparámetros).** El cuadro 2.1, además, disparaba `Float too large for page
   by 42pt`. Solución uniforme aplicada en los ficheros `sections/*.tex` (no en
   `export_latex_tables.py`, para no regenerar las 21 tablas): envolver el
   `\input{tables/...}` en `\resizebox{\textwidth}{!}{...}` más `\setlength
   {\tabcolsep}{4pt}`, y `\footnotesize` + `\arraystretch` 0,95 en los cuadros 1.1
   y 2.1. El escalado a `\textwidth` elimina el desbordamiento de ancho y, al
   reducir proporcionalmente la altura, también el `Float too large`.
8. **`makridakis2018m4` como `@inproceedings` con campo `journal`.** Corregido a
   `@article` (International Journal of Forecasting, 34(4):802-808). Añadida además
   la entrada `makridakis2022m5` (M5, 38(4):1325-1336), citada junto a M4 en
   `02_estado_arte.tex`, `05_resultados.tex` y `06_conclusiones.tex` §6.1; en
   `06_conclusiones.tex` §6.3 la mención a arquitecturas de parches/transformers
   se **desvinculó** de `makridakis2018m4` (anacronismo). `bates1969combination`:
   `journal` `OR` → `Operational Research Quarterly`. `surribas_sayago_diffuse`:
   `author` `... and et al.` → `... and others`, tipo `@misc`, nota sin la marca
   "PENDIENTE DE VERIFICAR".

**Estado final de compilación (cierre editorial):** `latexmk -pdf
-interaction=nonstopmode -halt-on-error TFT.tex` produce `TFT.pdf`, **84 páginas,
0 errores, 0 citas indefinidas, 0 referencias indefinidas**. Portada y
preliminares (Resumen, Abstract, palabras clave/keywords, Agradecimientos)
completos. Quedan 16 `Overfull \hbox` menores (<20pt): títulos de sección con
palabras compuestas no separables (`LSTM-XGBoost`, `CNN+LSTM+MLP`), entradas del
índice de figuras/cuadros y algún float de tabla con 3-11pt de exceso; todas
cosméticas y sin efecto en el contenido.

9. **Fase diagnóstica 7 — §5.9 ``Anatomía del residuo de la etapa 1''.** Añade 7
   figuras y 6 tablas. El `tabla_curva_aprendizaje` (8 columnas) desbordaba el
   margen derecho con `\small` + `column_spec` fijo; solución uniforme con el
   idioma ya establecido (incidencia 7): `\resizebox{\textwidth}{!}{\input{...}}`
   + `\tabcolsep` 4pt en `05_resultados.tex`, sin regenerar la tabla. Tras el
   ajuste (incluida la actualización de Anexo B/C en `07_anexos.tex`): **96 páginas,
   0 errores, 0 referencias/citas indefinidas, 0 `Overfull \hbox` >= 10pt**.
   ⚠️ **Criterio de aceptación corregido a posteriori (incidencia 12f).** El
   "0 `Overfull \hbox` >= 10pt" citado aquí **no acredita** cumplimiento de márgenes:
   un contador de `Overfull` no detecta una celda `p{}` cuyo contenido excede su ancho
   declarado. La cifra se conserva tal como se midió entonces, pero como señal
   secundaria; el criterio vigente es la medición directa con `pymupdf`.
   `stage1_residual_anatomy.py` y `lstm_learning_curve.py` no
   escriben ningún artefacto bajo `models/`; una prueba
   (`test_phase7_adds_no_model_to_the_master_comparison`) fija que ni
   `full_comparison.parquet` ni `sensitivity_comparison.parquet` ni
   `dashboard_data.MODEL_NAMES` cambian.

10. **Fase diagnóstica 8 — §5.10 ``Paridad informativa entre etapas''.** Añade 2
    figuras (`fig_ablacion_bloques`, `fig_paridad_lstm`) y 2 tablas
    (`tabla_ablacion_bloques` 8 columnas, `tabla_paridad_lstm` 8 columnas), ambas
    envueltas en `\resizebox{\textwidth}{!}{\input{...}}` + `\tabcolsep` 4pt con el
    idioma de la incidencia 7/9 — sin desbordamiento tras el ajuste. Un defecto real
    de compilación se corrigió en origen, no rodeándolo: `$\geq 60\,\%$` y
    `$\leq 25\,\%$` dentro de modo matemático disparaban `! Incompatible glue units`
    (el `\%` de babel-spanish, `\es@sppercent`, usa pegamento incompatible en modo
    math); solución, sacar el porcentaje del modo math: `$\geq$ 60\,\%`. La base de
    comparación univariante de la curva de aprendizaje se extendió de 3 a 5 semillas
    en la fracción 1,00 (semillas 3 y 4 añadidas a `lstm_learning_curve.parquet` sin
    tocar filas existentes) para mantener la comparación como-por-como con el control
    de paridad de 5 semillas; la mediana no cambia (416.324). Tras el ajuste
    (incluida la actualización de Anexo B/C en `07_anexos.tex`): **102 páginas,
    0 errores, 0 referencias/citas indefinidas, 0 `Overfull \hbox` >= 10pt**
    (⚠️ señal secundaria; criterio corregido en la incidencia 12f).
    `feature_block_ablation.py` y `lstm_parity_control.py` no escriben ningún
    artefacto bajo `models/`; una prueba
    (`test_phase8_adds_no_model_to_the_master_comparison`) fija que ni
    `full_comparison.parquet` ni `sensitivity_comparison.parquet` ni
    `dashboard_data.MODEL_NAMES` cambian.

11. **Fase diagnóstica 9 — §5.11 ``Desglose por subgrupos y periodos''.** Añade 3
    figuras (`fig_subgrupos_mae`, `fig_subgrupos_delta_ic`, `fig_mae_movil_periodo`)
    y 3 tablas (`tabla_subgrupos_test`, `tabla_subgrupos_val` de 8 columnas,
    `tabla_subgrupos_inferencia` de 8 columnas), las tres envueltas en
    `\resizebox{\textwidth}{!}{\input{...}}` + `\tabcolsep` 4pt con el idioma de las
    incidencias 7/9/10 — sin desbordamiento tras el ajuste. Las cadenas de veredicto
    (`subgroup_breakdown.VERDICT_*`) se redactaron sin `<` ni `/` para que
    `escape_latex` (que no neutraliza `<`) no rompiera el modo texto: `n insuficiente`
    en vez de `n < 20`, `sin solape val-test` en vez de `val/test`. `subgroup_breakdown.py`
    no importa `src.models`, no contiene `.fit(` ni `.train(` y no escribe nada bajo
    `models/`; una prueba (`test_phase9_adds_no_model_to_the_master_comparison`) fija
    que ni `full_comparison.parquet` ni `sensitivity_comparison.parquet` ni
    `dashboard_data.MODEL_NAMES` cambian. Tras el ajuste (incluida la actualización de
    Anexo B/C en `07_anexos.tex`): **111 páginas, 0 errores, 0 referencias/citas
    indefinidas, 0 `Overfull \hbox` >= 10pt** (⚠️ señal secundaria; criterio corregido
    en la incidencia 12f — de hecho el cuadro 5.16, añadido en la fase 7, llevaba
    58pt fuera del margen derecho mientras este contador marcaba cero);
    `sec:respuesta_preguntas` renumera a §5.12 en `TFT.toc`.

12. **Pasada de corrección posterior a la fase 9 (revisión de dirección sobre §5.9).**
    Cinco defectos señalados en revisión; ninguno reentrena nada ni altera una cifra
    calculada.

    a. **Periodograma (figura 5.16): el eje x llegaba a ~4e7 días.** La hipótesis
       inicial —una componente DC superviviente al pasar de frecuencia a periodo— es
       **incorrecta y queda registrada aquí como tal para que no sobreviva en el
       expediente**: `stage1_residual_anatomy.periodogram_frame` ya descarta el bin de
       frecuencia cero (`nonzero = freq > 0`) y su tabla abarca exactamente 2,005–393,0
       días. La causa real está en la capa de figura: sobre un eje logarítmico Plotly
       convierte la *forma* creada por `add_vline`, pero **no la anotación que crea a su
       lado**, cuya `x` se interpreta como un valor ya en log10. Así,
       `add_vline(x=7.0, annotation_text=...)` situaba la etiqueta en 10^7 días y el
       autorango se estiraba hasta ~4e7, comprimiendo toda la banda útil. Solución en
       `memoria_figures.stage1_periodogram_figure`: la línea se dibuja sin anotación y la
       etiqueta se añade aparte con `x=log10(7)`, y el rango se fija a partir del
       mín/máx de la propia tabla —banda 2 a N, **no** 2 a N/2: el bin de periodo 196,5
       días es el segundo pico más alto de todo el periodograma y recortarlo habría sido
       un defecto nuevo, no una corrección—. Ninguna cifra de la prosa cambia (cuota
       espectral de periodo 7 = 1,4436x; pico máximo en 2,339 d ≈ armónico 7/3, ambos
       reverificados). Dos pruebas de regresión lo fijan:
       `test_periodogram_axis_stays_within_the_representable_band` y
       `test_periodogram_period7_marker_is_placed_in_log_coordinates`. El periodograma es
       el único eje logarítmico del proyecto, así que no había más ocurrencias latentes.

    b. **Cuadro 5.15 (`tabla_residuo_meteo`): la columna "Significativa" cruzaba el
       margen.** El `\resizebox{\textwidth}{!}{...}` + `\tabcolsep` 4pt del idioma 7/9/10
       resultó **insuficiente por sí solo**, y la verificación por conteo de
       `Overfull \hbox` no lo habría detectado: medido con `pymupdf` sobre el PDF
       compilado, la tabla seguía sobresaliendo 3,1pt. Causa: "Significativa" es una
       palabra indivisible de ~2,2cm a `\small` dentro de una columna `p{1.9cm}`, de modo
       que desborda su propia celda y `\resizebox` escala el ancho *declarado*, dejando
       fuera los glifos sobrantes. Solución completa: columna ampliada a `p{2.4cm}` en
       `export_latex_tables.py` **más** el envoltorio en `05_resultados.tex`. Tamaño de
       fuente efectivo resultante **10,24pt** (frente a 11,96pt del cuerpo), medido en el
       PDF, holgadamente legible.

    c. **Cuadro 5.16 (`tabla_cadena_error`): desbordamiento no detectado hasta ahora.**
       Encontrado al medir el margen derecho página a página durante (b): 9 columnas,
       ancho natural ~18,2cm frente a 15,65cm de `\textwidth`, **58pt fuera del margen**
       y sin envoltorio. Corregido con el mismo idioma; fuente efectiva **10,02pt**.

    d. **Figura 5.19 (cadena de error): anotaciones ilegibles, y en el lado
       equivocado.** Estaban ancladas a una barra con `yshift=-26`, lo que las hacía
       colisionar con la barra vecina y —defecto adicional no señalado en la revisión—
       las situaba junto al par de barras que *no* describen: el salto de la fila
       `hybrid` describe `lstm_alone → hybrid`, pero el desplazamiento hacia abajo lo
       empujaba hacia `xgboost_alone`. Solución: cada etiqueta se ancla en la posición
       semientera del eje categórico, es decir en el hueco *entre* las dos barras que
       compara, con `height` 380→460 y `bargap` 0,45. No se elimina información: el
       Delta MAE absoluto y la cuota del hueco siguen ambos presentes, fijado por
       `test_error_chain_annotations_sit_between_bars`.

    e. **Prosa: cuadro 5.14 y la tensión ganancia/ablación.** Al cuadro 5.14 se le añade
       la nota de composición de los cuadros 5.3/5.4 (la fila `laborable` es la suma de
       sus dos hijas, no una categoría hermana). Y §5.10.3 gana un párrafo que reconcilia
       los tres hechos sobre meteorología retardada —51,6 % de ganancia (§5.5), 0/105
       correlaciones parciales significativas (§5.9.2), bloque net negativo en la
       ablación (§5.10.1)— por la vía de que la ganancia por impureza premia a las
       variables continuas de alta cardinalidad. En consecuencia **§5.5 se corrige en
       origen**: su lectura causal del 51,6 % pasa a declararse como hipótesis, con
       referencia explícita a las dos secciones que la rechazan. Auditado con `grep`
       sobre todo `docs/LaTeX/`, `CLAUDE.md` y el log de fases: §5.5 era la **única**
       aparición de esa lectura causal; el capítulo 6 (§6.2) y el resumen ya sostenían la
       versión correcta y no se han tocado.

    f. **Criterio de aceptación de márgenes: sustituido, y las fases 7-11 corregidas.**
       Las incidencias 9, 10 y 11 acreditaban el cumplimiento de márgenes con
       "0 `Overfull \hbox` >= 10pt". **Ese criterio no es válido para la clase de defecto
       que efectivamente ocurrió**: el contador de `Overfull` mide el desbordamiento de
       una caja horizontal al componer un párrafo, y no puede ver una celda `p{}` cuyo
       contenido es más ancho que el ancho declarado de la columna —TeX compone la celda
       sin protestar y los glifos sobrantes cruzan el margen en silencio—. Es
       exactamente así como escaparon los cuadros 5.15 y 5.16; el 5.16 llevaba **58pt
       fuera del margen** durante las fases 7, 8 y 9 con el contador marcando cero en las
       tres.

       **Criterio vigente:** el cumplimiento de márgenes se verifica por **medición
       directa del contenido renderizado contra la caja de texto** — pasada `pymupdf`
       sobre el PDF compilado, comparando el `bbox` derecho de cada línea contra
       `\textwidth` (caja de texto 78,0–521,6pt con la geometría actual). El conteo de
       `Overfull \hbox` se conserva **solo como señal secundaria**: sigue siendo útil
       para prosa, pero no acredita nada por sí mismo. Las incidencias 9-11 se han
       anotado en su sitio en lugar de reescribirlas: la cifra que se midió entonces se
       conserva, marcada como señal secundaria y remitida aquí.

    g. **`00_preliminares.tex` llevaba tres fases con el recuento de tests obsoleto.**
       El resumen (línea 25) y el abstract inglés (línea 67) seguían diciendo
       "310 pruebas" / "310 automated tests" desde antes de la fase 7. La causa es
       registrable: **ninguna lista de sincronización previa incluía
       `00_preliminares.tex`** —las fases 7, 8 y 9 sincronizaron Anexo B y Anexo C y
       dieron el recuento por propagado—, de modo que el número quedó obsoleto en el
       primer texto que lee un lector. Corregido a 445 en ambos idiomas y añadido a la
       lista de sincronización de abajo.

    **Lista de sincronización del recuento de tests** (actualizar **todas** a la vez;
    esta lista existe porque omitir una de ellas fue exactamente el fallo de (g)):

    | Ubicación | Contenido |
    |---|---|
    | `sections/00_preliminares.tex` (resumen) | "suite de N pruebas automáticas" |
    | `sections/00_preliminares.tex` (abstract) | "a suite of N automated tests" |
    | `sections/07_anexos.tex` (Anexo B, cabecera) | "**N tests**" + desglose por fase |
    | `sections/07_anexos.tex` (Anexo C, secuencia) | "la suite completa de N tests" |
    | `sections/07_anexos.tex` (Anexo D, implementación) | "la suite de N pruebas automáticas" |
    | `docs/CLAUDE_PHASE_LOG.md` (entrada de la pasada en curso) | "N/N tests passing" |

    > **Corregido en la incidencia 15:** esta lista decía "las cinco" y omitía la fila del
    > Anexo D, que también lleva el recuento. Eran seis. La prueba que la fija solo exige
    > que el número aparezca en `00_preliminares.tex` y en `07_anexos.tex`, de modo que la
    > omisión no la habría hecho fallar: una aparición correcta en el mismo fichero basta
    > para pasarla. La lista es, por tanto, más estricta que su prueba, y hay que leerla.

    Las entradas históricas del log de fases **no** se actualizan: son instantáneas
    fechadas y reescribirlas falsearía el expediente.

    > Desde la incidencia 13 esta lista está **fijada por una prueba**
    > (`test_test_count_is_synchronised_across_every_location`), que recolecta el recuento
    > real y exige que aparezca en las ubicaciones de prosa. La tabla se conserva porque
    > sigue documentando *por qué* existe la regla y cubre la entrada del log de fases,
    > que ninguna prueba puede comprobar.

    **Estado tras la pasada:** `python -m pytest` → **445/445** (3 nuevos); `latexmk` →
    **111 páginas** (sin cambio), 0 errores, 0 referencias/citas indefinidas. Quedan 23
    líneas que cruzan el margen derecho, todas de prosa con identificadores `\texttt{}`
    indivisibles (`test_phase9_adds_no_model_to_the_master_comparison`,
    `feat_is_bridge_day`) en §5.10 y §5.11 y todas **preexistentes** —ninguna cae en un
    rango de líneas editado en esta pasada—. **No se corrigen aquí**: son identificadores
    indivisibles, no un error de formato, y requieren un remedio coherente decidido sobre
    el documento completo; quedan para la auditoría global.

13. **Auditoría global de la memoria (previa a entrega).** Seis ejes: procedencia
    numérica, consistencia interna, referencias y estructura, márgenes, prosa española y
    cumplimiento del encargo. No se reentrenó ni se repredijo nada; `models/` y
    `data/processed/` quedan **sin cambios** (verificado con `git status`).

    a. **Superlativo en doble sentido (S1/S2).** La frase "los dos modelos más precisos de
       todo el estudio" se aplicaba simultáneamente a los dos ensambles (§4.7) y a
       SARIMAX + `xgboost_alone` (§2.2 ×2, §6.1). Sobre `full_comparison.parquet` el orden
       de MAE de test es ensamble_inverse_mae < ensamble_equal < xgboost_alone < sarimax,
       de modo que la segunda atribución era falsa. Corregida con el calificador "de una
       sola etapa", que ya usaba §5.1. El resumen y el abstract, que declaraban
       `ensemble_equal` como "el modelo más preciso", se alinearon con la formulación de
       §4.7: ambas variantes, prácticamente indistinguibles, y la equiponderada
       seleccionada por no derivar sus pesos de la partición de validación.

    b. **Citas a ficheros internos eliminadas del cuerpo de la memoria.** Las siete
       referencias a `CLAUDE.md`, la del registro de fases y la fila de la tabla 1.1
       (esta última en origen, en `export_latex_tables.PHASES`) se reformularon como
       "el documento de contexto técnico del proyecto". Se conserva su **función** —
       acreditar que los compromisos metodológicos se fijaron por escrito y antes de ver
       resultados — y se refuerza haciéndola explícita. Dos de esas citas eran además
       incorrectas: el apartado `"Relation to the reference paper"` no existe, y
       `"OPEN DECISION — same-day weather"` pertenece al registro de fases, no a
       `CLAUDE.md`. La declaración de uso de IA se redactará aparte, según la normativa
       de la universidad.

    c. **Márgenes: 0 infracciones, por medición directa.** Partiendo de 18 líneas fuera de
       la caja de texto (peor caso 76pt), cuatro reglas de documento completo, ninguna
       corrección línea a línea: `\_` redefinido como `\_\allowbreak` en `preamble.tex`
       (extiende a la prosa el idioma que ya emitían las tablas generadas);
       `\emergencystretch` 3em; `\allowbreak{}` tras `/` dentro de `\texttt{}` aplicado
       mecánicamente a los 28 grupos que lo requerían; y `\raggedright` en el formato de
       título de capítulo. Se descartó `hyphenat[htt]` por insertar guiones
       indistinguibles de un carácter real del identificador. Además, el cuadro 5.13
       (`tabla_residuo_dia_semana`) carecía del envoltorio `\resizebox` que sí recibieron
       los demás cuadros anchos en las incidencias 7/9/10/11; añadido, fuente efectiva
       10,91pt.

    d. **Cruces y estructura.** Tres flotantes no se referenciaban desde la prosa
       (`fig:ablacion_bloques`, `tab:subgrupos_inferencia`, `fig:subgrupos_delta_ic`);
       añadidas sus referencias. Bibliografía limpia en ambos sentidos (17/17) y
       numeración de figuras y cuadros monótona respecto al orden de página.

    e. **Cumplimiento de la propuesta.** Se añadieron cuatro señalizaciones para
       desviaciones legítimas pero invisibles: *random forest* omitido (§4.6), AEMET
       sustituido por Open-Meteo (§3.1), RMSE en lugar de MSE (§2.2) y el cuadro de mando
       como entregable de **despliegue** del ciclo KDD (§1.3 y Anexo C). Nueva sección
       **§6.4 "Implicaciones para la planificación"**, que cubre el décimo objetivo
       específico de la propuesta sin introducir ninguna cifra nueva. §1.1 explica ahora
       que los diez objetivos de la propuesta se consolidan en cinco.

    f. **Prosa española.** LanguageTool 6.8 (es) sobre 28.596 palabras: 1.100 coincidencias
       brutas, de las que 1.076 son artefactos del *detex* o falsos positivos —verificado
       por muestreo, no supuesto— y **24 hallazgos reales**, todos de puntuación y
       ortografía: coma ante `sino` (11), `por tanto` sin comas (7 — el documento ya
       escribía `, por tanto,` diez veces, así que era una inconsistencia interna), coma
       ante `pero` (4) y `pre-pandemia` → `prepandemia`. Cero errores de gramática,
       concordancia u ortografía española. Aplicados los 22 de coma y `prepandemia`.

       ⚠️ **Decisión de esta pasada, REVERTIDA en la fase 4 de la revisión formal por
       instrucción de dirección (item 24; incidencia 24c).** Lo que aquí se decidió, y que
       se conserva como historia: `pre-registrado` NO se corregía a `prerregistrado`.
       "Prerregistrado" es la forma correcta según la RAE, pero `pre-registrado` es el
       término técnico de la práctica metodológica (traducción directa de
       *pre-registration*), aparecía seis veces como locución fija en §5 y en el Anexo B, y
       su reconocibilidad para un lector familiarizado con esa práctica se juzgó de más peso
       que la ganancia ortográfica; la nota original pedía que una pasada futura no lo
       "corrigiera". Dirección fija el criterio contrario (homogeneizar como
       `prerregistrado` / `prerregistro`) y la fase 4 lo aplica en toda la memoria y en los
       dos literales del generador. La nota se reescribe en lugar de borrarse para que el
       fichero muestre la decisión y su reversión, no un silencio.

    g. **Nueva prueba permanente: `tests/test_prose_claims_match_artifacts.py`** (17
       pruebas). Sustituye la lista de sincronización escrita por una ejecutable: un
       manifiesto `PROSE_CLAIMS` declara, para cada cifra de alta relevancia, su artefacto
       de origen y **la lista completa de ficheros `.tex` en que debe aparecer**, de modo
       que falla tanto si el artefacto cambia sin propagarse como si un fichero pierde la
       cifra. Cubre las métricas de titular, los tamaños de partición, los recuentos de
       filas y variables, los contadores de figuras y cuadros y
       `TOTAL_TESTS_BEFORE_MEMORIA`. El recuento de tests se recolecta en tiempo de
       ejecución, para que la propia aserción no pueda quedar obsoleta. Las cifras se
       buscan en las dos convenciones numéricas, porque `00_preliminares.tex` contiene el
       resumen y el abstract y este último deriva de forma independiente.

    **Estado tras la auditoría:** `python -m pytest` → **462/462** (17 nuevos); `latexmk`
    → **113 páginas**, 0 errores, 0 referencias/citas indefinidas; medición `pymupdf` →
    **0 infracciones de margen** (10 `Overfull \hbox` como señal secundaria, incidencia
    12f); `models/` y `data/processed/` sin modificar.

    **Pendientes abiertos, fuera del alcance de esta auditoría:** el cuaderno de flujo
    principal y la declaración de uso de IA, ambos solicitados explícitamente por la
    dirección y ambos condición para su revisión final.

13. **Pasada A — correcciones de contenido y registro** (esta pasada). Barrido de tres
    clases completas, no de los tres casos reportados: (1) **metacomentario** — texto cuyo
    contenido es una afirmación sobre el documento y no sobre la materia; 11 pasajes en
    `01`, `02`, `03`, `05`, `06` y `07`, entre ellos §1.4 comprimida de 25 a 8 líneas y la
    deduplicación OE→PI de `01`. (2) **Narración del proceso de dirección** — 8 pasajes de
    `05_resultados.tex` (§5.10, §5.11) reescritos para que las preguntas Q1/Q2 y la
    búsqueda por subgrupos surjan del análisis previo y no de una petición externa; el
    contenido intelectual y las preguntas no cambian, solo el sujeto que las plantea. Las
    citas **del artículo** de Surribas-Sayago et al. (`\cite{surribas_sayago_diffuse}`,
    §2.3) son literatura y **no** se barren: la distinción aplicada es *autor citado* vs.
    *origen de una instrucción*. Portada y agradecimientos, intactos. (3) **Respuestas de
    consulta en prosa corrida** — PI1-PI5 de §5.12 pasan a `description` de `enumitem`
    (pregunta como etiqueta, veredicto en la línea siguiente) y los `\paragraph` de §5.10.2
    pasan a `\textbf` en línea, de modo que el documento use **un solo dispositivo** para
    los ítems titulados en lugar de los cuatro que coexistían.

    **Estado:** `python -m pytest` → **462/462**; `latexmk` → 103 páginas, **cuerpo en 80**
    (sin cambio, dentro del límite), 0 errores, 0 referencias/citas indefinidas; medición
    `pymupdf` → **0 infracciones de margen** fuera de la portada (que corre su propio
    `\newgeometry`), 10 `Overfull \hbox` sin cambio.

    **Por qué el cuerpo no baja de 80 pese a recortar ~50 líneas:** el recuento de páginas
    de los capítulos 1 a 3 es **float-bound, no text-bound**. En el capítulo 1, el diagrama
    de Gantt y el cuadro `tabla_cronograma_fases` ocupan sendas cabeceras de página y dejan
    ~150 pt muertos al pie de cada una; con `placeins[section]`, la barrera de flotantes
    que abre §1.4 obliga a descargar el cuadro pendiente y §1.4 arranca en página nueva.
    Recortar prosa **antes** de esos flotantes no recupera nada. Consolidar ambos flotantes
    en una página liberaría una página completa, pero es ajuste de composición
    (**Pasada B**, junto con el interlineado de filas de cuadro), no de contenido.

14. **Pendiente abierto heredado de la retirada de referencias a herramientas de IA:** tras
    aquella pasada, cuatro pasajes —`01_introduccion.tex:101` y `:145`,
    `02_estado_arte.tex:196`, `07_anexos.tex:153`, más la variante de `07_anexos.tex:242`—
    invocan **«el documento de contexto técnico del proyecto»** como la autoridad que fija
    por escrito un compromiso metodológico (el protocolo OOF como no negociable, el uso de
    Surribas-Sayago solo como inspiración, la conservación de las columnas de operador, la
    convención de reproducibilidad). Ese documento **no se nombra en ningún punto de la
    memoria ni figura en los anexos**, de modo que un lector no tiene forma de saber qué
    es. No se corrige aquí: es un cabo suelto creado por aquella pasada, no por esta, y su
    resolución natural es la **declaración de uso de IA** (pendiente 12, arriba), que es el
    lugar donde corresponde identificar el registro técnico del proyecto. Al redactarla,
    revisar los cinco pasajes para que apunten a un documento identificable.

15. **Cuaderno principal de flujo (`notebooks/07_flujo_completo.ipynb`).** Último
    entregable solicitado por dirección: un cuaderno que permita seguir el flujo completo
    del trabajo —carga y comprobación de datos, preparación de variables, partición
    temporal, ejecución de los modelos principales, comparación de resultados y gráficos
    finales— llamando a las funciones de `src/` sin duplicar código.

    **Decisión sobre el cuaderno preexistente:** se **conserva** `06_dashboard.ipynb` sin
    tocarlo y se **añade** el 07. Son dos entregables distintos: el 06 es el panel de
    resultados (visualización interactiva con widgets sobre artefactos ya persistidos, y
    la etapa de *despliegue* del ciclo KDD que cita §1.3), el 07 es el flujo end-to-end.
    Refundirlos habría obligado a reescribir §1.3 —una edición en el **cuerpo**, que está
    en 80 páginas y es *float-bound* (ver incidencia 13)— y a negar la primera celda del
    06, que declara que no entrena ni recalcula nada. Con esta decisión el delta de
    páginas del cuerpo es **0**: todas las ediciones caen en anexos y preliminares.

    **Política de ejecución (medida, no supuesta).** Recomputa en vivo: ingesta y
    validaciones de integridad (0,3 s), matriz de 161 variables (< 0,1 s, `.equals` con
    `features_daily.parquet`), partición y warm-up, los cuatro baselines y el SARIMAX
    (0,9 s en total, bit-exacto contra `baseline_predictions.parquet`), `xgboost_residual`
    (14,6 s) y `xgboost_alone` (14,9 s), ambos bit-exactos y con los mismos
    hiperparámetros que los `.joblib` publicados. Carga de artefacto **solo** la etapa 1
    LSTM, y por razón metodológica: la etapa 2 debe corregir los residuos del mismo modelo
    de etapa 1 que produjo los resultados publicados (ver el docstring de
    `hybrid_residual/combine.py`), de modo que reentrenarla mediría la diferencia entre
    dos redes en lugar del pipeline. De SARIMAX se carga únicamente el orden
    `(2,1,2)x(0,1,2,7)` de `sarimax_selected_order.json`: los ~6 minutos del paso 3 del
    Anexo C son la **búsqueda AIC en rejilla**, no el ajuste, que cuesta menos de 1 s.

    **Salvaguarda.** Las 22 filas de `full_comparison.parquet` x (4 métricas + `n`) = 110
    valores se contrastan celda a celda; el cuaderno **levanta** si alguno supera
    `rtol = 1e-6`. En la máquina de referencia la desviación observada es exactamente
    `0.000e+00` en los 110. La tolerancia no absorbe deriva conocida: existe solo para la
    reasociación en coma flotante del histograma multihilo de XGBoost
    (`tree_method="hist"`, `n_jobs=-1`) cuando cambia el número de núcleos. Sobre un MAE de
    test de ~156.000 son 0,16 viajeros/día.

    La celda muestra **siempre** la tabla de contraste completa antes de levantar, y el
    mensaje de la excepción **interpreta la magnitud**: por debajo de `1e-3` relativo la
    declara diferencia de entorno de ejecución y afirma explícitamente que los resultados
    publicados se mantienen; por encima, la declara cambio real del pipeline y enumera qué
    revisar. El motivo es que el lector probable es dirección, ejecutando en su propia
    máquina, y su primera conclusión ante una excepción sería que el trabajo no reproduce
    —conclusión errónea a escala de reasociación en coma flotante.

    **Versionado con salidas.** El cuaderno se comitea **ejecutado**
    (`nbconvert --to notebook --execute --inplace`), porque el uso más probable es leerlo
    en GitHub, que renderiza `.ipynb`: con las salidas limpias mostraría código sin
    resultados. La celda de cabecera imprime fecha de ejecución, versiones y tiempo total.
    26/26 celdas de código con salida, 0 errores, 14 figuras Plotly embebidas.

    **Cambios en `src/`** (para no meter lógica en el cuaderno):
    `ingestion/unify.py::integrity_report` (informe de las cuatro invariantes, incluida
    `metro+emt+carretera+cercanias == total`, que hasta ahora solo vivía en un docstring y
    en los tests); `evaluation/full_comparison.py::verify_against_published` /
    `assert_within_tolerance` / `load_published` / `build_comparison_frame`; y
    `assemble_predictions` ensanchada con cuatro parámetros opcionales que por defecto leen
    de disco —comportamiento por defecto idéntico al anterior— para que el cuaderno puntúe
    sus piezas recomputadas por **la misma** ruta de código que produjo la tabla publicada.

    **Tests:** +5 en `tests/test_flujo_notebook.py`, de 462 a **467**. El test decisivo es
    el negativo: perturba una métrica y exige que la salvaguarda levante **y** clasifique
    bien la magnitud en ambos sentidos. Una salvaguarda que nunca dispara es
    indistinguible de no tener salvaguarda, que es la forma exacta del defecto (g) de la
    incidencia 12 y del de la incidencia 13.

    **Nota de numeración:** este fichero contiene **dos incidencias numeradas 13** (líneas
    ~418 y ~505), defecto **preexistente** a esta pasada. No se renumera: hacerlo volvería
    ambiguas las referencias cruzadas ya escritas contra "la incidencia 13". Se usa el
    siguiente ordinal libre (15) y se deja constancia aquí.

16. **Guardias anti-fuga inalcanzables en `build_features` (defecto latente, corregido).**
    Destapado al escribir la celda de demostración de guardias del cuaderno principal
    (incidencia 15): el primer intento de dispararlas inyectando una columna del mismo día
    **no las hizo saltar**, y la razón resultó ser que no podían saltar.

    **El defecto.** Las dos re-aserciones de `build_features` —el `LEAKAGE GUARD` de
    operador y la primera mitad del `SAME-DAY WEATHER GUARD`— buscaban en el bloque de
    variables el nombre **desnudo** de la columna:

    ```python
    feats = feature_columns(out)                       # solo columnas con prefijo feat_
    leaked = [c for c in FORBIDDEN_SAME_DAY_COLS if c in feats]   # 'total', 'metro', ...
    ```

    `FORBIDDEN_SAME_DAY_COLS` contiene `total`, `metro`, `emt`, `carretera`, `cercanias`;
    `feature_columns()` selecciona por prefijo `feat_`. Ninguno de esos nombres lleva el
    prefijo, luego `leaked` **nunca podía tener elementos**. La comprobación era vacua, y
    lo fue a lo largo de varias fases. El `SAME-DAY WEATHER GUARD` tenía además una segunda
    mitad que sí comparaba contra el nombre prefijado: la asimetría entre ambas guardias es
    lo que delata que la de operador se quedó a medio escribir, no que fuese deliberada.

    **Qué NO ocurrió.** Ninguna fuga. La protección efectiva nunca fue esta comprobación,
    sino una propiedad **estructural**: `add_lag_features` solo emite retardos $\geq 1$ y
    `add_weather_features` solo formas retardadas o móviles, de modo que una columna del
    mismo día no tiene por dónde entrar en la matriz. Las guardias por módulo sobre la
    salida de cada bloque sí son alcanzables y sí funcionan. Ningún número publicado
    cambia, y `features_daily.parquet` es idéntico antes y después de esta corrección.

    **Por qué se corrige en vez de documentarse como redundante.** Se valoraron ambas
    salidas. Se corrige porque el argumento estructural es una garantía *condicional*: se
    sostiene mientras nadie añada un retardo 0 a `LAGS` ni un `passthrough` del mismo día a
    un constructor, que es exactamente la regresión para la que existe una red secundaria.
    El docstring del bloque ya declaraba la intención correcta —"re-assert on the assembled
    frame, since concat is where a mis-named column would actually land in the matrix"—; lo
    que fallaba era la implementación, no el propósito. Y dejar viva una comprobación cuyo
    nombre promete una garantía que no puede dar es el peor de los tres estados posibles:
    el siguiente lector, incluida la preparación de la defensa, la dará por buena.

    **La corrección.** Ambas guardias comparan ahora contra el nombre **prefijado**
    (`feat_metro`, `feat_temperature_2m_mean`), que es lo que produciría una regresión de
    constructor y lo único que puede aparecer en `feats`. Se descarta la comparación contra
    el nombre desnudo en las dos: contra `feats` es vacua, y contra `out.columns` levantaría
    en **todo** frame válido, porque `RAW_REFERENCE_COLS` lleva deliberadamente las cuatro
    columnas de operador y `TARGET_COL` lleva `total`. El comentario del bloque explica las
    tres cosas —qué buscan, por qué no el nombre desnudo, y que la protección primaria es
    estructural— para que la asimetría no se reintroduzca.

    **Tests:** +3 en `test_features.py` (467 → **470**). Dos disparan cada guardia
    simulando la regresión de constructor con `monkeypatch`; el tercero fija el contrato de
    columnas, de modo que ninguna de las dos reescrituras erróneas del guard (contra `feats`
    o contra `out.columns`) pueda hacerse más adelante sin que la suite lo diga.

17. **Cumplimiento de la guia docente 14MBID (ed. octubre 2025-26).** La guia oficial se
    localizo el 09/09/2026 y sus requisitos formales nunca se habian contrastado. Medicion
    completa y remediacion en una pasada; ningun modelo se reentrena, y `models/` y
    `data/processed/` quedan sin tocar.

    a. **Instrumento nuevo: `scripts/audit_guia_docente.py`.** Mide sobre el PDF compuesto
       tipografia, interlineado, margenes, extension del cuerpo, reparto vertical de
       flotantes, recuento de palabras bajo cinco convenciones y los elementos exigidos
       (ODS, URL del repositorio, resumen/abstract). Sustituye a las mediciones ad hoc que
       hasta ahora vivian en scratchpads de sesion y se perdian.

       Dos defectos **del propio instrumento**, corregidos antes de fiarse de el y
       anotados porque ambos daban un veredicto falso en silencio: (i) el corte que separa
       prosa de contenido tabular estaba fijado a 11,5 pt, valido con la clase de 12 pt;
       al bajar a 11 pt clasificaba *toda la prosa como cuadro* y el recuento C4 salia
       negativo. Ahora se deriva del cuerpo medido. (ii) El interlineado se comparaba
       contra el cuerpo de letra y no contra la linea base sencilla de la clase, de modo
       que una composicion correcta a 1,15 medía 1,43 y **fallaba**. La regla: 1,15 es un
       multiplo del sencillo (13,6 pt a 11 pt), nunca del cuerpo.

    b. **Formato (requisito C).** Clase 12pt a 11pt; geometry a 4,4/2,19/3/3 cm;
       `\linespread{1.15}` en lugar de `setspace`, que no esta instalado en esta MiKTeX y
       que para un factor uniforme sobre `\baselineskip` no aporta nada. `footskip` a
       20 pt y pie reducido a una linea: con `bottom=2,19cm` el pie de dos lineas quedaba
       a 0,8 cm del corte. El pie dice ahora "Trabajo Fin de Master", que la guia exige
       que aparezca en la memoria y no aparecia.

    c. **Margenes tras el reflujo.** La caja de texto pasa de 15,65 a 15,00 cm y saco
       fuera de margen cinco cuadros con columnas `p{}` fijas (`tabla_fuentes_datos`,
       `tabla_comparacion_test` y `_val`, `tabla_criterio_experimento_a`,
       `tabla_residuo_calendario`): 32 lineas infractoras, la peor a 15,4 pt. Corregidos
       con el idioma de las incidencias 7/9/10/11 (`\resizebox{\textwidth}` mas
       `\tabcolsep` 4pt). Ademas `\@pnumwidth` ensanchado a 2,2em: el campo de numero de
       pagina del indice esta dimensionado para dos caracteres y los anexos numeran
       A1..A19. **Resultado: 0 infracciones**, por medicion directa (criterio 12f).

    d. **Anexo E (requisito E): `scripts/prepare_anexo_e.py`.** Traslada los once cuadros
       diagnosticos anchos de las secciones 5.9-5.11 a un anexo nuevo. Adoptado del
       borrador `prepare_anexo_d.py` con tres correcciones: la etiqueta que declaraba era
       `cap:anexo4`, **que ya usa el Anexo D** —habria creado una etiqueta duplicada, que
       LaTeX resuelve en silencio a favor de la ultima, desviando al anexo nuevo las
       referencias al Anexo D—; la letra correcta es E y no D, porque se inserta despues
       del Anexo D; y el recuento de tests del docstring seguia en 462. Cuerpo de 79 a 75
       paginas. Ninguna cifra, veredicto ni criterio pre-registrado sale del cuerpo: la
       prosa cita los cuadros por numero, nunca por posicion.

    e. **ODS (requisito G), ausente por completo.** Nueva seccion 6.5. Declara ODS 11
       (meta 11.2) y ODS 13 y **acota ambos de forma explicita**: el hallazgo central es
       negativo, la contribucion es metodologica y no se afirma ninguna relacion causal
       entre calidad de la prediccion y huella ambiental. El vinculo con el ODS 13 no es
       la precision sino el coste computacional evitado, que si es atribuible al trabajo.

    f. **Repositorio (requisito F).** `\url{}` unica en la cabecera del Anexo C. La
       auditoria previa a la publicacion encontro material de terceros versionado:
       `docs/references/ts.pdf` es Zhang (2003), Neurocomputing (**Elsevier, con derechos
       reservados**) y `TFM_Docu_OLIVAS_VIU_26.pdf` es documentacion interna de la VIU.
       Retirados del indice con `git rm --cached` y anadidos a `.gitignore`; **siguen en
       el historial**, de modo que publicar este repositorio tal cual los expondria. No
       hay secretos, rutas absolutas ni datos personales, y `data/` nunca se versiono.

    g. **Resumen y abstract (requisito D).** El abstract ingles estaba en 353 palabras,
       por debajo del minimo de 400 que fija la guia. Ampliado a 465. Ambos declaran ahora
       la relacion con el trabajo previo de la direccion (Surribas-Sayago et al.), que la
       guia pide mencionar de forma explicita y no se mencionaba. Cada uno sigue cabiendo
       en una pagina, comprobado en el PDF.

    h. **Portada e hipotesis.** La portada desbordaba a dos paginas y le faltaban la fecha
       en formato mes/ano y la identificacion "Trabajo Fin de Master"; anadidas ambas y
       reducidos los `\vspace` para que quepa en una. Frase al final del capitulo 2 que
       remite a PI1-PI5 como hipotesis de trabajo, que es donde la guia las espera; las
       preguntas no se mueven.

    i. **Recuento de palabras (requisito B): NO resuelto, y deliberadamente.** La guia
       limita a 30.000 "el texto completo, incluido anexos" sin definir que cuenta.
       Medido: 40.278 / 36.120 / 35.686 / 31.068 / 29.286 palabras segun la convencion
       (de todo lo compuesto a solo prosa corrida). Solo la mas estrecha cumple. **No se
       ha recortado nada**: cual rige es una decision de direccion y no un hecho medible,
       y cada convencion implica un recorte distinto, de cero a un tercio del documento.
       Consulta redactada en `docs/CONSULTA_DIRECCION.md`, que ademas plantea el titulo:
       lleva los acronimos "LSTM-XGBoost" contra lo que la guia pide, pero esta en la
       propuesta aprobada y en el pie de cada folio, de modo que no se cambia
       unilateralmente. Se ofrece una alternativa sin acronimos para que la decision venga
       con una opcion concreta.

    j. **Trampa operativa de compilacion, registrada porque produjo cuatro diagnosticos
       falsos seguidos.** Un `latexmk` interrumpido deja un `TFT.aux` truncado, y la
       pasada siguiente falla con `! Extra }, or forgotten \endgroup` o con
       `! File ended while scanning use of \@newl@bel` senalando una linea *que no existe*
       en el `.aux`. **No es un error del `.tex`**: el fichero senalado esta bien.
       `latexmk -C` no siempre lo limpia, porque un `pdflatex` colgado puede seguir
       reteniendolo ("Device or resource busy"). Remedio: matar los `pdflatex`/`latexmk`
       huerfanos y borrar a mano
       `TFT.{aux,toc,lof,lot,out,bbl,blg,fls,fdb_latexmk,log}` antes de recompilar.

    **Estado:** `python -m pytest` -> **470/470**; `latexmk` -> **106 paginas, cuerpo en
    76**, 0 errores, 0 referencias/citas indefinidas; `audit_guia_docente.py` -> **0
    infracciones de margen** y todos los bloqueantes en OK salvo el recuento de palabras,
    en espera de direccion.

18. **Reduccion de extension bajo la lectura literal del limite de 30.000 (tramos 0 y 1).**
    Ejecutada bajo el supuesto de que rige la convencion **C1** (todo lo compuesto, sin el
    pie repetido). **Deroga parcialmente la incidencia 17i**, que decia «no se ha recortado
    nada»: ya no es cierto. La consulta de `docs/CONSULTA_DIRECCION.md` sigue sin respuesta;
    lo que sigue es reversible y no prejuzga la decision.

    **Tramo 0 — material no autoral.** Los 4.158 de indices se midieron pagina a pagina:
    indice general 1.282, de figuras 1.431, de cuadros 1.445. Los dos ultimos son
    opcionales para la guia y se retiran de `TFT.tex` (`\listoffigures`, `\listoftables` y
    sus dos `\addcontentsline`); el general se conserva, es obligatorio. **-2.884.** La
    bibliografia (434) se **reporta y no se toca**: excluirla es reinterpretar el criterio,
    que es justo lo que esta pendiente de direccion.

    **Tramo 1 — redundancia, sin perder ningun hallazgo.** Anexo B, desglose mecanico de
    tests por fichero (**-286**, conservando 470 y 257 y el hallazgo de las guardias
    inalcanzables de la incidencia 16, que pasa a parrafo propio); Anexo C, glosas de la
    lista de 22 comandos y de los parrafos de cuaderno de flujo, semillas e identidad
    visual (**-227**); §2.4, los cinco ejes narrados uno a uno que ya tabula el Cuadro 2.1
    (**-172**, intactos el *research gap* y la lectura de Granger; **corregido de paso un
    contador obsoleto**: «de un total de 310» -> 470); §6.1 y dos parrafos de §6.2
    (**-479**, intactos §6.4, §6.5 y **los dos parrafos de veredicto ambiguo de §6.2**, que
    son objeciones metodologicas de §5.9 y §5.10); §1.3, narracion de proceso (**-225**);
    cabecera del cap. 5 y §5.2, que duplicaban el Anexo C (**-66**).

    **Resultado medido.** C1 **40.278 -> 36.164** (**-4.114**); cuerpo **76 -> 73 paginas**;
    C4 **31.068 -> 29.822** y C5 **29.286 -> 28.284**, ambas ahora **por debajo de 30.000**.
    `pytest` **469/470**, `latexmk` **96 paginas**, 0 errores, 0 referencias indefinidas,
    `audit_guia_docente.py --strict` con **todos los bloqueantes en OK**.

    **Fallo preexistente, no causado por esta pasada:**
    `test_flujo_notebook.py::test_notebook_executes_clean`. `jupyter nbconvert` termina con
    codigo 1 **sin escribir nada en stdout ni stderr** y en ~2,6 s, cuando la ejecucion
    completa cuesta ~40 s: muere antes de abrir el kernel. Verificado con `git stash`:
    falla identicamente sobre el commit `6bfd86d` sin modificar. Es una averia del entorno
    Jupyter, no del cuaderno ni de la memoria.

    **Tramo 2 no ejecutado, y por que.** Quedan **6.164** palabras hasta 30.000 y ya no hay
    de donde sacarlas sin tocar contenido aprobado. La medicion decisiva: retirando *ademas*
    la bibliografia, los tres indices y **el interior de los 32 cuadros y los pies de las 34
    figuras** —vaciar el documento de todo lo que no es prosa de autor— el recuento se
    quedaria igualmente en **32.350**. No existe combinacion de recortes no autorales que
    llegue a 30.000: el resto tiene que salir de §5.9, §5.10 y §5.11 (6.379 palabras), que
    responden a las seis peticiones explicitas de direccion. Se detiene aqui y se pregunta.

19. **Auditoria de los anexos contra el arbol publicado.** Los anexos se habian escrito
    antes del repositorio publico, del cuaderno 07, de los anexos D y E y de la pasada de
    reduccion de la incidencia 18. Se auditaron los 22 pasos del Anexo C, cada ruta,
    modulo e identificador citado en A, B y C, y los recuentos, contra el arbol real. No se
    reentrena nada; `models/` y `data/processed/` quedan sin tocar.

    a. **El Anexo C no generaba `full_comparison.parquet`.** La secuencia nunca ejecutaba
       `python -m src.evaluation.full_comparison`, **unico** escritor de ese artefacto
       (`full_comparison.py:352`), y dos pasos posteriores lo leen: la ablacion de bloques
       (`feature_block_ablation.py:143`) y el cuaderno 07 via `load_published`, cuyo propio
       mensaje de error **nombra el comando que el anexo omitia**. Un lector siguiendo el
       anexo en orden se paraba en la ablacion. Insertado como paso 10, tras `combine`
       —sus dependencias son baselines, hibrido, `xgboost_alone.joblib` y features, todas
       presentes en ese punto; no depende de los ensambles—.

    b. **Los datos de partida eran inobtenibles.** El anexo remitia a «las fuentes abiertas
       identificadas en el cuadro 3.1», y el cuadro 3.1 tenia tres columnas —Fichero,
       Contenido, Formato— que **no identificaban ninguna fuente**. No habia una sola URL de
       datos en `sections/*.tex` ni entrada en el `.bib` para CRTM, Open-Meteo ni el
       calendario. Con `data/` en `.gitignore` y nunca versionado, el paso 1 era irrealizable
       desde un clon. Anadida una cuarta columna `Origen` en `build_fuentes_datos`, con las
       tres direcciones **verificadas por peticion real (HTTP 200, sin redireccion)** antes
       de escribirlas, y reescrita la frase del anexo para decir donde colocar los ficheros.
       Se usan paginas de conjunto de datos y no enlaces directos: el id de recurso rota, la
       pagina no. **No se afirma licencia ni condicion de redistribucion en ningun punto**:
       los terminos no se establecieron, y afirmarlos seria peor que omitirlos.

    c. **No habia paso de instalacion.** El anexo presuponia `venv` activado y citaba
       `requirements.txt` como catalogo de pines sin decir nunca que se instalara. Anadida
       la frase de `python -m venv venv` + `pip install -r requirements.txt`.

    d. **`\url` no rompe donde hacia falta, y el contador de Overfull volvio a no verlo.**
       Con las URL en una columna `p{}` de 3,6 cm, `crtm-evolucion-demanda-diaria/` se salia
       **47,1pt** del margen derecho y `historical-weather-api` 6,7pt, con `Overfull` a cero:
       la incidencia 12b otra vez, detectada solo por la medicion `pymupdf`. Corregido en
       `preamble.tex` anadiendo `-`, `.` y `:` a `\UrlBreaks` (del paquete `url`, que
       hyperref ya carga: sin dependencia nueva) mas `\Urlmuskip`. Regla de documento
       completo, no parche de celda, y no inserta ningun guion.

    e. **El Anexo E no se nombraba en ninguna parte.** `cap:anexo5` aparecia **una sola vez
       en todo `sections/`: su propio `\label`**. La enumeracion de anexos de §1.5 se
       detenia en el D. Anadido alli (unica edicion del cuerpo de esta pasada, +5 palabras,
       sin delta de paginas).

    f. **Dos afirmaciones inexactas.** El cuaderno 06 no lee «exclusivamente de
       `data/processed/`»: carga tambien `models/xgboost_{residual,alone}.joblib`
       (`dashboard_data.py:38-39`). Y el alcance declarado de «el repositorio» cubria solo
       los anexos B y C, dejando fuera la mencion del Anexo A, que ademas nombraba
       `build_features.py` sin su ruta. Corregidas ambas. «470 tests, **ejecutables en su
       totalidad**» pasa a «recolectados y ejecutados»: la frase no afirmaba que pasaran,
       pero se leia asi, y `test_flujo_notebook.py::test_notebook_executes_clean` esta rojo
       por la averia de entorno de la incidencia 18.

    g. **La URL del repositorio no resuelve hoy.** `https://github.com/AdayCuestaCorrea/TFM`
       devuelve **404** y la API de GitHub lista 9 repositorios publicos en esa cuenta,
       ninguno llamado `TFM`. Es deliberado —privado de momento, con la direccion como
       colaboradora, publico llegada la entrega—, de modo que **el texto no se toca**: sera
       cierto en la defensa. Queda registrado como **puerta de salida previa a la entrega**,
       porque mientras siga en 404 la frase «es de acceso publico en...» es falsa y es lo
       primero que un tribunal comprueba.

    **Verificado tambien, y correcto —no volver a auditarlo:** los 19 modulos `python -m`
    del Anexo C existen todos en la ruta citada y en el indice de git; 470 tests
    (`--collect-only`), 257 `TOTAL_TESTS_BEFORE_MEMORIA`, 34 PNG y 32 `tabla_*.tex`
    coinciden con la prosa y con `PROSE_CLAIMS`; todos los identificadores citados en el
    Anexo B existen; cero referencias en `sections/` a `docs/references/`,
    `CONSULTA_DIRECCION*`, `CLAUDE.md` o al directorio antiguo; nada describe el
    repositorio como privado o pendiente; la descripcion del cuaderno 07 (reparto
    recomputa/carga, salvaguarda a `rtol=1e-6`, versionado con salidas) es exacta.

20. **Resuelta la averia de entorno de la incidencia 18: dos defectos, ambos del renombrado
    del directorio, ninguno del cuaderno.** `test_notebook_executes_clean` volvia rojo
    (469/470) saliendo con codigo 1 en ~2,6 s **sin escribir nada en stdout ni stderr**. La
    causa no era Jupyter: era que el directorio de trabajo paso de `TFM` a `TFM-HISTORICO`
    mientras un `TFM` distinto seguia en disco. Ningun modelo se reentrena; el cuaderno **no
    se re-ejecuta ni se modifica**, y `models/` y `data/processed/` quedan sin tocar.

    a. **La salida silenciosa: un `.exe` de consola con el interprete cableado.** El test
       invocaba `-m jupyter nbconvert`. El despachador de `jupyter_core` **no importa**
       nbconvert: **lanza** `venv\Scripts\jupyter-nbconvert.exe`, y esos lanzadores de
       Windows llevan grabada una ruta absoluta fijada al instalar —
       `#!C:\MisCosas\Universidad\TFM\venv\Scripts\python.exe`, que ya no existe—. El
       lanzador muere antes de nada y **no escribe en ninguna de las dos corrientes**, que
       es exactamente el sintoma. Confirmado por contraste: el `.exe` da codigo 1 mudo
       mientras `python -m nbconvert --version` da 0 y `7.17.1`. No es especifico de
       nbconvert (`jupyter kernelspec list` calla igual); `jupyter --version` y `--paths`
       funcionan porque `jupyter_core` los resuelve en proceso sin lanzar nada. **No era
       una dependencia ausente ni un desajuste de version**: nbconvert 7.17.1, nbclient
       0.11.0, nbformat 5.11.0, ipykernel 7.3.0, jupyter_client 8.9.1 y jupyter_core 5.9.1
       estan instalados y son compatibles, y el kernelspec `python3` resuelve bien y
       coincide con el declarado por el cuaderno.

    b. **Enmascarado detras del anterior: la instalacion editable apuntaba al arbol viejo.**
       Al saltarse el despachador aparece el fallo real, un `FileNotFoundError` sobre
       `C:\MisCosas\Universidad\TFM\data\raw\CRTM_...xlsx`: el kernel importaba `src` del
       **directorio hermano obsoleto**, porque el buscador de la instalacion editable
       conservaba `MAPPING = {'src': 'C:\\MisCosas\\Universidad\\TFM\\src'}` de antes del
       renombrado, y aquel arbol no tiene `data/`. El resto de la suite estaba verde porque
       `pytest.ini` fija `pythonpath = .`, que precede al buscador editable en `sys.path`;
       el kernel no tiene esa ayuda, porque nbconvert lo arranca con cwd en `notebooks/`,
       donde `src` no esta.

    c. **La correccion.** En `test_flujo_notebook.py`: punto de entrada de modulo
       (`-m nbconvert`, sin despachador ni dependencia de PATH), `PYTHONPATH` fijado a la
       raiz del repositorio —de modo que la prueba resuelve `src` a **este** arbol pase lo
       que pase con cualquier instalacion editable— y el mensaje de fallo pasa a citar
       `stdout + stderr` y el codigo de salida: con `stderr` a secas, el caso mudo no decia
       **nada**, debilidad que este incidente destapo. Las tres decisiones quedan razonadas
       en el docstring para que una pasada futura no las «ordene» de vuelta. Fuera del
       codigo, `pip install -e . --no-deps` regenera el mapeo a `TFM-HISTORICO\src`
       (`--no-deps` deliberado: **no toca ni un pin**, luego la justificacion de
       TensorFlow/numpy queda intacta y `requirements.txt` no se modifica).

    d. **Medido tras la correccion:** cuaderno en codigo 0 y 37,6 s, 26 celdas de codigo, 0
       errores, 14 figuras Plotly, ninguna celda sin salida, y la salvaguarda declarando
       `110/110 valores coinciden con full_comparison.parquet` con desviacion relativa
       maxima `0.000e+00`. `python -m pytest` **470/470 sin omitidas** (el recuento no
       cambia: no se anade ni se retira ninguna prueba, luego la lista de sincronizacion de
       la incidencia 12g **no se dispara** y `test_prose_claims_match_artifacts` sigue en
       17/17). Cero palabras de delta: no se edita ningun `.tex`.

    e. **Los 60 lanzadores mudos, reparados sin tocar un solo pin.** El renombrado dejo
       **60 de los 62** `.exe` de `venv\Scripts\` apuntando al interprete inexistente —no
       solo los de Jupyter: tambien `pip.exe`, `pytest.exe`, `ipython.exe`,
       `tensorboard.exe`—, todos fallando mudos. La suite era inmune (invoca siempre
       `sys.executable -m ...`), pero un `pytest` tecleado en la terminal fallaba en
       silencio, que es la trampa que costo la sesion de diagnostico entera.

       **Metodo, y por que no `--force-reinstall`.** Un lanzador de Windows es
       `[binario][#!<ruta>\n][zip]`, y el zip admite prefijo de longitud arbitraria (formato
       autoextraible), de modo que la ruta se **reescribe in situ**. No descarga nada, no
       reinstala ningun paquete y no ejecuta resolucion de dependencias: **es imposible que
       mueva un pin**. `pip install --force-reinstall --no-deps <paquete>` sin `==version`
       habria traido la **ultima** version de cada uno —justo lo que el pinado de
       TensorFlow/numpy prohibe—, y con `==version` habria implicado redescargar
       TensorFlow entero para reescribir seis lanzadores. Verificado despues: `kaleido`
       1.3.0, `numpy` 2.2.6, `tensorflow` 2.20.0, `plotly` 6.3.0, `xgboost` 3.0.5,
       `pandas` 2.3.3, `scikit-learn` 1.7.2 — sin mover. Copia de seguridad de `Scripts\`
       antes de tocar nada. Los cuatro comandos exigidos devuelven 0 **con salida**:
       `pytest --version`, `pip --version`, `jupyter --version` y —la que antes callaba—
       `jupyter kernelspec list`.

    f. **Tres dependencias mas de la ruta antigua, que no eran lanzadores.** El barrido
       posterior desmintio la hipotesis de que solo los `.exe` estuviesen afectados:
       `venv\Scripts\activate`, `activate.bat` y `activate.fish` fijaban `VIRTUAL_ENV` al
       directorio inexistente (`Activate.ps1`, el que cita el documento de contexto, se
       salvo porque deriva la ruta en tiempo de ejecucion). Corregidos igual, por
       reescritura de texto. Se dejan a proposito dos residuos inertes: la linea
       `command = ...` de `pyvenv.cfg`, que es el registro historico de como se creo el
       entorno y no se lee en ejecucion, y el `co_filename` grabado en los `.pyc` de
       `site-packages` (solo afecta a la ruta que muestra un traceback; se regeneran solos).
       El bytecode obsoleto de `src/` y `tests/` si se purgo, porque hacia que los
       tracebacks del proyecto citasen el arbol viejo. No queda ningun `.pth`, buscador
       editable, configuracion de Jupyter, `kernel.json` (usa `python` relativo) ni ajuste
       de VS Code apuntando a la ruta antigua.

       **Pendiente, no corregido aqui:** `CLAUDE.md` sigue declarando
       `Project root: C:\MisCosas\Universidad\TFM`. Es fichero versionado y de instrucciones
       del proyecto, no entorno; se deja a decision expresa.

    g. **El directorio `C:\MisCosas\Universidad\TFM` era el arbol publicado — renombrado, no
       borrado.** No es un residuo del renombrado: es un clon de publicacion con **un solo
       commit** (`094ebc6`, 09/09/2026 17:23, «TFM: Madrid public transport demand
       forecasting (memoria + pipeline)»), arbol limpio, sin stashes ni ramas extra, y
       `git ls-remote` confirma que **`refs/heads/main` de GitHub es exactamente ese
       commit**. Los dos repositorios comparten remoto
       (`https://github.com/AdayCuestaCorrea/TFM.git`); el `origin/main` local de
       `TFM-HISTORICO` (`6bfd86d`) es una referencia de seguimiento **obsoleta**, y `094ebc6`
       **no existe** en su base de objetos. Contenido: sus 172 ficheros son **identicos bit a
       bit** a los de `HEAD` en `TFM-HISTORICO` (0 blobs distintos), que ademas tiene 5 mas,
       excluidos de la publicacion a proposito (`docs/CONSULTA_DIRECCION.md`,
       `docs/references/Summary.md`, `Images/image{4,5,6}.png`). **No contiene nada que no
       este ya en `TFM-HISTORICO`**, pero es la unica copia local del commit publicado, de
       modo que borrarlo perderia el vinculo local con la historia de GitHub —problema
       distinto del que se estaba resolviendo—.

       **Resuelto por renombrado:** pasa a `C:\MisCosas\Universidad\TFM-publicado`
       (`Rename-Item`). Conserva intacto el objeto del commit publicado, hace inequivoco su
       papel, elimina la confusion con el directorio de trabajo y **libera el nombre `TFM`**,
       que la secuencia de exportacion necesita libre para volver a correr. Verificado tras
       el renombrado: `HEAD` sigue en `094ebc6`, arbol limpio, `origin/main` intacto.

    **Ensayo de clon limpio.** Clon real, `venv` nuevo, `pip install -r requirements.txt` y
    los pasos 1 y 2 con los tres ficheros crudos copiados en `data/raw/`. Se clono el
    **repositorio local**, no el publico, porque este sigue privado: es una prueba **mas
    debil**, no detecta un fichero que exista en local y quede excluido del arbol publicado.
    Repetir contra el clon publico cuando se publique.

    **Estado:** `python -m pytest` -> **469/470** (unico fallo, el del cuaderno, incidencia
    18); `latexmk` -> **96 paginas, cuerpo en 73**, 0 errores, 0 referencias/citas
    indefinidas; `audit_guia_docente.py --strict` -> **0 infracciones de margen**, todos los
    bloqueantes en OK. Recuento C1 **36.164 -> 36.255** (**+91**); C4 **29.944** y C5
    **28.330**, ambas aun por debajo de 30.000 —C4 con solo 56 palabras de holgura, de modo
    que cualquier adicion futura bajo esa convencion la rompe—.

21. **Revision formal de direccion, fase 1 de 5: sistema de citas APA 7 (items 1-4 y 35).**
    Direccion devolvio 35 puntos, todos formales; el trabajo experimental queda cerrado y
    ningun modelo se reentrena. Esta fase sustituye el sistema numerico (`plain`, «Granger
    [9]») por autor-ano APA 7, retitula la lista «Referencias» ordenada por apellido,
    completa la ficha de Surribas-Sayago y elimina las marcas «ficha sin verificar» / «No
    verificado». **No cambia ninguna afirmacion de contenido**: los items 5-7 (estado del
    arte) y 13-21 (precision metodologica) son las fases 2 y 3.

    a. **Mecanismo: `biblatex` + `style=apa` (biblatex-apa 9.20) + `biber` 2.21**, ambos
       verificados instalados en esta MiKTeX antes de decidir (la ausencia de `setspace`
       en su dia obligaba a comprobar, no a suponer). Se descartaron `natbib`+`apalike`
       (APA 6 aproximado, cadenas en ingles, sin DOI ni sufijos a/b) y `apacite` (APA 6).
       Con babel-spanish, `spanish-apa.lbx` se carga solo: narrativas con «y» («Bates y
       Granger (1969)»), parenteticas y lista con «&» («(Chen & Guestrin, 2016)») —APA 7
       estricto, sin sobrescribir el estilo, decision expresa del autor—, «En R. J. Mammone
       (Ed.)», «(3.ª ed.)», «Artículo 1428», sufijos «2026a/2026b» automaticos para las dos
       fichas de Anthropic. `\printbibliography[heading=bibintoc,title={Referencias}]`:
       la lista **no tenia entrada en el indice general** hasta ahora; ya la tiene.
       hyperref sigue cargandose el ultimo; 0 avisos.

    b. **Las 36 citas, convertidas a mano una por una** (22 narrativas -> `\textcite`,
       retirando el nombre que la prosa ya llevaba para que el estilo no lo duplique; 14
       parenteticas -> `\parencite`). Ninguna prosa dependia de la numeracion. Redaccion
       intacta salvo esa retirada del nombre.

    c. **Auditoria de la bibliografia, entrada por entrada (19 entradas, 19 citadas, 19
       resueltas por biber: limpia en ambos sentidos).** Todos los DOI/URL anadidos los
       verifico el autor contra el registro del editor y se insertaron **textualmente**,
       sin inferir ninguno (el Anexo D declara en letra impresa que toda referencia esta
       verificada contra su fuente, luego un DOI supuesto la falsearia). Hallazgos
       corregidos, que se dejan constancia porque son defectos que un tribunal detecta:
       - **`monje2022deep` llevaba dos nombres de autor equivocados**: «Laura Monje» y
         «Ricardo A. Carrasco». La fuente primaria, en el propio repositorio
         (`docs/references/mathematics-10-01428-v3.pdf`), dice **Leticia Monje** y
         **Ramón A. Carrasco**. En la salida APA (iniciales) el error era invisible, y la
         ficha figuraba como «verificada» desde la pasada del capitulo 1; se corrige de la
         fuente y se registra aqui. Ademas `pages={1428}` era el numero de articulo, no
         un rango: pasa a `eid`, que APA 7 compone como «Artículo 1428».
       - **`makridakis2022m5`: paginas 1325-1336 -> 1346-1364**, segun el registro del
         editor aportado por el autor.
       - **`surribas_sayago_diffuse`**: era una `@misc` con inicial de autor equivocada
         («C.») y titulo distinto del real. Reconstruida como `@inproceedings` con los datos
         que facilito direccion (tres autores, 2026, SOCO 2025, CCIS 2806, pp. 466-475,
         Springer, DOI). Sin campo `editor` por ahora: biblatex-apa la compone en forma de
         articulo de actas (valida en APA 7); cuando direccion facilite los editores del
         volumen pasa a forma de capitulo con un cambio de una linea.
       - `perrone1993networks`: editorial «Chapman-Hall» -> «Chapman & Hall». Capitulo de
         libro de 1993: sin DOI por decision, no por omision.
       - `cerqueira2020evaluating`: faltaba el numero (11). `hyndman2021fpp`: URL del libro
         en linea. Las dos fichas de Anthropic pasan de `@misc`+`howpublished` a
         `@software` con `titleaddon` (descripcion entre corchetes) y `url`.
       - Sin DOI a proposito: `box1970time` (libro) y `perrone1993networks` (capitulo).
       - `makridakis2018m4`: existen dos articulos «The M4 Competition» de los mismos
         autores (IJF 34(4) 2018 y IJF 36(1) 2020); la ficha coincide campo a campo con el
         primero y la prosa lo usa de forma generica junto a M5, de modo que el DOI
         anadido es el de 34(4), 802-808.

    d. **La trampa del generador, evitada.** «ficha sin verificar» y los dos «No
       verificado» **no viven en el `.tex`**: estan en `LITERATURE_COMPARISON` de
       `src/evaluation/export_latex_tables.py`, y editar el cuadro a mano lo habria
       revertido en silencio la siguiente exportacion (como ya paso dos veces: `PHASES` y
       `tabla_cronograma_fases`). Se corrige el generador y se regenera
       `tabla_comparativa_estado_arte.tex` con `python -m src.evaluation.export_latex_tables`
       (las otras 31 tablas salen identicas: `git status` solo marca esa). Las dos celdas
       (b) horizonte y (d) protocolo **se rellenan desde el propio articulo**, no se dejan
       sin verificar ni se preguntan a direccion: seccion 3.1 «Experimental Setup» (p. 5)
       —observaciones minutales de radiacion difusa mas temperatura, humedad relativa y
       presion del Cabauw Experimental Site for Atmospheric Research (CESAR, Paises Bajos),
       diciembre de 2023; «the first 80% of the data» para entrenar y «the remaining 20%»
       para probar, una sola ejecucion, sin validacion cruzada—. Se redactan en paralelo a
       las filas de Monje («Holdout cronologico simple 80/20, una sola ejecucion, sin CV»)
       y Zhang («Holdout unico ... sin OOF») para que el lector compare los protocolos de
       un vistazo. Con esto la declaracion del Anexo D («todas verificadas contra su
       fuente») es cierta tambien para esta entrada. `test_literature_pending_verification_studies_are_flagged`
       pasa a `test_literature_carries_no_pending_verification_marker`, invertido **en su
       sitio** (ninguna celda de ninguna fila lleva marca alguna, y la fila de Surribas
       lleva el ano): el recuento de la suite sigue en 470, que es la cifra impresa en la
       memoria seis veces.

    e. **Efectos colaterales previstos y resueltos.** `scripts/audit_guia_docente.py`
       localizaba la lista por la palabra «Bibliografía» y habria abortado con
       «Referencias»: acepta ambas. Su convencion C5 descarta `\cite`; se anaden
       `\textcite` y `\parencite` para que las 36 claves no cuenten como palabras.
       `.gitignore`: `*.bcf` y `*.run.xml` (artefactos de biber). Anexo D: las dos fichas
       de herramientas pasan a «(2026a)» y «(2026b)» para coincidir con la lista. Un
       `TFT.bbl` **generado por BibTeX** sobrevive a `latexmk -C` (no figura en su base de
       dependencias) y hace abortar a biblatex («File 'TFT.bbl' not created by
       biblatex»): borrarlo una vez; despues, `latexmk` lanza biber solo.

    f. **Extension, medida antes y despues con el mismo instrumento** (ensayo previo en
       copia desechable, luego el documento real): cuerpo **73 -> 73 paginas**; C1
       **36.255 -> 36.296** (+41, cifra conocida y comunicable: direccion admite «36.000 o
       algo mas»); bibliografia 434 -> 435 palabras (pierde la numeracion, gana los DOI);
       0 infracciones de margen; tipografia, interlineado y margenes sin cambio.

    **Estado (final de la fase, con el cuadro 2.1 regenerado):** `python -m pytest` ->
    **470/470**; `latexmk` desde limpio -> **96 paginas, cuerpo en 73**, 0 errores, 0
    referencias/citas indefinidas, biber 0 avisos, «Referencias» en el indice (A16);
    `audit_guia_docente.py --strict` -> todos los bloqueantes en OK, 0 infracciones de
    margen; C1 **36.255 -> 36.309** (+54), C4 29.985, C5 28.284; bibliografia 436
    palabras. `grep "sin verificar|No verificado"` sobre docs/, src/, tests/ y scripts/:
    solo quedan las menciones historicas de este registro, del diario de fases y las
    aserciones negativas del test.

22. **Revision formal de direccion, fase 2 de 5: estado del arte (items 5-7).** Cuatro
    referencias nuevas, todas verificadas por el autor contra el registro del editor y con
    las celdas del cuadro 2.1 leidas del propio articulo; ninguna afirmacion metodologica
    cambia y nada se reentrena.

    a. **Cobertura (item 5).** De los siete temas que pide direccion, tres ya estaban
       cubiertos (demanda de transporte: Monje, Toque, Cardozo; LSTM en transporte: Monje;
       ensambles: Wolpert, Granger, Bates y Granger, Perrone y Cooper, M4/M5) y cuatro no:
       demanda diaria a escala de sistema, XGBoost aplicado a demanda, hibrido aplicado a
       transporte, e influencia de calendario y meteorologia —este ultimo el mas grave, porque
       §2.1 lo afirmaba citando solo el capitulo 5—. Se cubren con `yazicioglu2025passenger`
       (Applied Sciences 15(11), 6260: Estambul, validaciones horarias, XGBoost como selector
       por ganancia y ranking encabezado por hora, dia lectivo, dia de la semana, festivo y
       mes, con precipitacion y nieve entre las ultimas: replica independiente del resultado
       calendario-dominante de la memoria), `arana2014weather` (TR Part A 59, 1-12: Gipuzkoa,
       tarjetas inteligentes, viento y lluvia reducen los viajes) y `chen2019emd` (PLoS ONE
       14(9), e0222365: hibrido EMD-LSTM sobre una estacion del metro de Chengdu que bate a
       LSTM, BPN y ARIMA con 14 dias de entrenamiento y un unico viernes de prueba, sin
       exogenas) y `singhal2014weather` (TR Part A 69, 379-391: metro de Nueva York,
       demanda diaria y horaria de 2010-2011 sobre los mismos datos, regresion OLS
       explicativa sin particion temporal —propiedad del estudio, no critica—; es el unico
       trabajo revisado que compara ambas escalas, y respalda la agregacion diaria de este
       TFM como decision de diseno). Singhal se anadio en un segundo paso: el articulo no
       se pudo leer desde esta sesion (ScienceDirect y Semantic Scholar rechazan la
       peticion; Crossref solo confirma la ficha) y la regla de la fase es no inferir
       ninguna celda, de modo que la fila espero a que el autor confirmase las cinco celdas
       sobre el resumen y los *highlights* del articulo.

       **Dos trampas de ficha, registradas porque un tribunal las detecta** (segundo y tercer
       defecto de exactitud de citas que destapa esta revision, tras los nombres de Monje en
       la incidencia 21c):
       - El DOI de Arana que circula en fuentes secundarias (`10.1016/j.trb.2013.10.010`,
         prefijo de Part B para un articulo de Part A) es erroneo; el correcto,
         `10.1016/j.tra.2013.10.019`, se tomo del registro de Crossref.
       - Los nombres de pila de los siete autores de Chen: el articulo y el propio PLoS solo
         dan iniciales («Chen Q, Wen D, Li X, Chen D, Lv H, Zhang J, et al.»), y un primer
         borrador de la ficha los completo **de memoria** como «Qingchao, Ding, Xuyang,
         Dongjie, Hongxia, Jing, Peng». Contrastado con Crossref antes de escribir nada:
         cuatro de los siete eran falsos; los correctos son Quanchao, Di, Xuqiang, Dingjun,
         Hongxia, Jie y Peng. En la salida APA (iniciales) el error habria sido invisible,
         exactamente como el de Monje. Regla que se fija: **ningun campo de una ficha se
         completa de memoria**; si la fuente da iniciales, se consulta el registro del editor
         o se dejan iniciales.
       - Chen tiene una correccion (PLoS ONE 15(3), e0231199, 2020) que retira un fichero de
         apoyo incluido por error sin alterar resultados: por decision de direccion se cita
         el articulo original y la correccion queda anotada en la cabecera del `.bib`, no en
         la memoria.

    b. **Cuadro 2.1: 6 -> 10 filas, en el generador.** Filas nuevas en `LITERATURE_COMPARISON`
       con comentario de procedencia, en el orden transporte-pronostico (Monje, Yazicioglu,
       Toque), transporte sin pronostico (Cardozo, Arana, Singhal), hibrido en transporte (Chen),
       metodologicos (Zhang, Surribas) y «Este TFM». Regenerado con
       `python -m src.evaluation.export_latex_tables` (`git status` marca solo esa tabla).
       Medido en el PDF: cuadro mas pie pasan de 235 a ~335 pt de los 425 pt de alto util de
       la pagina apaisada (`\resizebox` escala al ancho, que no cambia, luego cada fila suma
       su altura natural: 38, 30 y 31 pt); no hay corte de pagina ni cambio de emplazamiento.
       `test_literature_table_has_six_rows_and_five_axes` pasa a
       `test_literature_table_has_one_row_per_study_and_five_axes` **en su sitio** (recuento
       derivado de la constante, «Este TFM» ultima, columnas exactas): la suite sigue en 470.

    c. **Item 6, Surribas-Sayago.** La prosa ya decia «inspiracion, nunca arquitectura
       replicada», pero en ningun sitio decia lo que direccion pide: que **no es un
       antecedente de prediccion de demanda de transporte publico**. Parrafo de §2.3
       reescrito para abrir con esa declaracion, enumerar que se toma (retardos, Fourier,
       rama exogena) y que no (la fusion paralela), y retirar la apelacion al «documento de
       contexto tecnico del proyecto» (cabo suelto de la incidencia 14, una de las cinco
       apariciones; quedan cuatro en `01` y `07`).

    d. **Item 7, absolutos.** Barrido de toda `sections/` con los patrones ningun/ninguna
       (estudio|trabajo|otro|antecedente), no existe, por primera vez, unico/unica, sin
       precedente, inedito, ausencia de, hueco exacto. Siete pasajes corregidos, todos en
       `02_estado_arte.tex`: el *research gap* con la formula literal de direccion («Entre los
       trabajos revisados, no se ha identificado un antecedente directo que analice un
       resultado hibrido negativo bajo un protocolo equivalente de validacion temporal y
       controles diagnosticos»), «Ningun estudio de los citados» -> «Entre los trabajos
       revisados no se ha identificado uno que», «hueco exacto» -> «el hueco que se propone
       ocupar», «unica entidad» -> «la entidad», y los tres «es el unico que / ningun otro» de
       §2.4 acotados explicitamente a los recogidos en el cuadro. Quedan cuatro coincidencias
       y se dejan a proposito: «por primera vez» en `04:429` y `05:281` (orden interno de
       exposicion/observacion, no literatura) y «ningun otro elemento/fichero» en `05:428` y
       `07:346` (hechos del codigo). Resumen, abstract, introduccion y conclusiones no
       contenian absolutos bibliograficos.

    e. **Extension, medida.** Primera compilacion: cuerpo 73 -> **75** (el encabezado de §2.4
       cayo solo en una pagina antes del cuadro apaisado, y una linea se salio despues).
       Compensado con ~100 palabras de recorte antes del cuadro (SARIMAX «no es un baseline
       decorativo», clausulas redundantes de Zhang y del *stacking*) y ~20 despues (frase de
       Granger en §2.4), sin tocar ninguna cifra de `PROSE_CLAIMS`. Final: cuerpo **73**,
       96 paginas, 0 errores, 0 referencias/citas indefinidas, biber 0 avisos, 0 `\cite{}`
       desnudos, 0 infracciones de margen. C1 **36.309 -> 36.876** (+567: 114 de las tres
       fichas, ~250 del interior del cuadro, el resto de prosa; direccion admite «36.000 o
       algo mas»); C4 29.985 -> 30.183 (cruza 30.000 bajo esa convencion, que ya tenia 15
       palabras de holgura; la que rige es C1). `python -m pytest` -> **470/470**.

23. **Revision formal de direccion, fase 3 de 5: precision metodologica (items 13-21 y la
    frase sin cita de §6.3).** Es la unica fase de la revision que cambia **lo que la memoria
    afirma** y no como se lee; por eso cada edicion se barrio por clase de afirmacion y no
    por instancia reportada, y se cerro con una lectura cruzada afirmacion a afirmacion
    (resumen → abstract → §2.3 → §4.7 → §5.6.1 → §5.10.3 → §5.12 → §6.1 → §6.2), que es la
    salvaguarda contra el defecto que este documento ya tuvo dos veces (incidencia 13a).
    Ningun modelo se reentrena, ninguna cifra medida cambia, `data/processed/` y `models/`
    quedan sin tocar (`git status`). Item 30 (barrido global de «demuestra/descarta»),
    fase 4, queda fuera salvo donde una frase editada aqui lo contenia; lo que se deja se
    lista en (h).

    a. **Item 14, «161 variables exogenas» → «predictoras».** 25 apariciones en siete
       ficheros de `sections/` renombradas; las 16 en que «exogena» designa informacion
       externa a la serie (SARIMAX, eje (c) del cuadro 2.1, `solo_exogeno` 122, Surribas)
       se conservan a proposito. PI2 se reformula con redaccion **identica** en §1.2 y
       §5.12 (verificado por script, las cinco PI identicas). En el **generador**
       (`LITERATURE_COMPARISON`, fila «Este TFM»), la celda de la columna «(c) Matriz
       exogena» decia «161 variables: …», que bajo esa cabecera era falso: pasa a «122
       exogenas de las 161 predictoras: 105 meteo + 6 Fourier + 11 calendario». Cuadro
       regenerado con el exportador; `git status` marca solo esa tabla. `PROSE_CLAIMS`
       exige «161» en ocho ficheros y ninguna edicion retira el numero: 17/17.

    b. **Item 13, fuga de informacion.** «libre de fuga» (×3 en `01`), «garantias
       anti-fuga» (`01`, titulo del Anexo B, `07:176`) y «limpia de fuga» (`05`) pasan a
       «controles especificos contra las principales formas de fuga identificadas en el
       diseno experimental» / «guardias anti-fuga». §6.1 decia «ninguna de las guardias
       anti-fuga ha fallado en ningun momento del proyecto», que contradecia al propio
       Anexo B (dos guardias *no podian* fallar durante varias fases, incidencia 16); pasa
       a «sus guardias anti-fuga —cada una con una prueba negativa que la obliga a
       disparar— no han detectado ninguna fuga en la matriz publicada».

    c. **Item 15, horizonte de SARIMAX.** Frase de direccion insertada en §4.1 (un dia
       hacia adelante, actualizado a diario, no un pronostico estatico), replicada en OE2,
       §2.4 y la celda de protocolo del generador. Se anade ademas lo que la memoria **no
       declaraba en ningun sitio**: que todos los modelos se evaluan al mismo horizonte de
       un dia (retardos ≥ 1 en la matriz, ventana de la etapa 1 en $[t-W,t)$), de modo que
       la tabla maestra compara pronosticos a un dia vista en todos los casos. §6.4 («el
       horizonte es de dias») se ajusta a «el dia siguiente, el unico evaluado».

    d. **Item 16, Granger.** La contradiccion senalada (A: «ambas etapas observan el mismo
       pasado», ×9; B: «la etapa 2 ve lo que la etapa 1 no puede ver», ×8) no se resuelve
       eligiendo un bando: ambas son ciertas a distinto nivel, y la frase recomendada por
       direccion, aplicada sola, habria retirado la lectura de Granger de seis lugares y
       dejado el resultado negativo sin teoria. Lo que se corrige es la **caracterizacion
       de los conjuntos**: no son identicos (la etapa 2 anade calendario, meteo, Fourier y
       operadores) pero tampoco distintos en el sentido de la condicion, porque el pasado
       reciente que ve la etapa 1 esta **representado** en la matriz de la etapa 2 (siete
       retardos y cuatro estadisticos moviles sobre la misma ventana de 28 dias): conjuntos
       **anidados, no complementarios**. En esa situacion la combinacion secuencial no puede
       superar en teoria al mejor modelo sobre el conjunto mayor, que existe en la
       comparacion (`xgboost_alone`); la ablacion (§5.10.1) y la paridad (§5.10.2) miden las
       dos mitades de esa lectura. La frase de direccion se incorpora **ademas**, como
       lectura empirica. Propagado a resumen, abstract, §2.3, §4.7, §5.6.1 (×3), §6.1 (×2).
       Se dice «representado», nunca «contenido integramente»: la ventana cruda es mas fina
       que 7 retardos + 4 moviles (O4, §5.10.2, sigue siendo cierto).

       ⚠️ **Atribucion, decision expresa del autor.** La consecuencia «conjuntos anidados
       ⇒ la combinacion no supera al mejor modelo sobre el conjunto mayor» **no se atribuye
       a Granger (1989)**: el articulo no esta en `docs/references/` y no se ha verificado
       que la formule asi. Se enuncia como consecuencia logica de la condicion citada y como
       hecho sobre esta matriz. En la literatura de combinacion el resultado se conoce como
       *forecast encompassing*; si el autor verifica una fuente, entra en una linea como en
       las fases 1 y 2. Hasta entonces la frase se sostiene sola.

    e. **HALLAZGO DE EXACTITUD (tercero de la revision, tras los nombres de Monje, 21c, y
       de Chen, 22a): §4.7 afirmaba que `xgboost_alone` aprende «sin ningun termino
       autorregresivo».** Falso: la matriz lleva siete retardos de `total` y cuatro
       estadisticos moviles, y la propia memoria mide en §5.10.1 que retirar ese bloque
       temporal cuesta **+30,9 %** de MAE de validacion. El argumento de que el ensamble
       satisface la condicion de Granger descansaba, por tanto, en un hecho falso. No lo
       senalo direccion: lo destapo el barrido del item 14 al releer cada «exogenas».
       Corregido: SARIMAX y `xgboost_alone` tienen conjuntos que **se solapan pero no se
       anidan** (historial completo + cinco indicadores de calendario, frente a memoria
       limitada a 7 retardos + 4 moviles mas operadores, Fourier, calendario completo y
       meteo, «sin estructura autorregresiva ni diferenciacion»). La condicion se sigue
       cumpliendo; la razon dada era incorrecta.

    f. **HALLAZGO DE EXACTITUD (cuarto): la logica de dos frases sobre Bates y Granger
       estaba invertida.** §5.8 cerraba con «la ganancia observada es mayor de lo que un
       simple promedio de errores sin ninguna estructura comun explicaria, lo que confirma
       que procede de esta descorrelacion genuina», y §5.7.2 con «por un margen que una
       simple media de errores independientes no explicaria por si sola». Con la identidad
       $\sigma^2(1+\rho)/2$ que la propia seccion deriva, errores **sin** estructura comun
       ($\rho=0$) reducen la varianza **a la mitad**, mas que los $\rho=0{,}576$ medidos
       (79 %): errores independientes explican *mas* ganancia, no menos, de modo que ambas
       frases afirmaban lo contrario del mecanismo que invocaban. Eliminadas; §5.8 pasa a la
       formula de direccion («coherente con el mecanismo … y con la correlacion imperfecta
       observada»), con la salvedad explicita de que la identidad, formulada en varianza y
       bajo varianzas iguales, no demuestra la ganancia. Ademas §4.7 describia un contraste
       «reduccion de varianza observada frente a la predicha» que §5.8 **no realiza** (mide
       MAE, no la varianza del error del ensamble): reformulado. Resumen, abstract, §2.3
       («verifica empiricamente» → «contrasta»), §5.3, §5.6.1 («demuestra» → «muestra»),
       §6.1 (×2) y PI5 alineados a «coherente con / en consonancia con». Ninguna cifra
       nueva: la restriccion de la fase prohibe recomputar, y no se introduce
       $\sqrt{0{,}79}$ ni ningun otro numero derivado.

       **Patron que queda registrado.** Cuatro defectos de exactitud en tres fases, ninguno
       visible en la salida compuesta: dos nombres de autor invisibles bajo iniciales APA,
       un hecho sobre la matriz contradicho por la propia ablacion cuatro secciones mas
       adelante, y dos frases cuya logica contradecia la identidad que citaban dos parrafos
       antes. Los cuatro se detectaron releyendo contra la fuente primaria (el PDF, Crossref,
       el propio §5.10.1, la propia formula), nunca por el aspecto del texto.

    g. **Items 17, 21 y §6.3.** Item 17: §5.12 adopta la formula de direccion («reduce la
       plausibilidad … no obstante, el ambito `completo` presenta un resultado mixto …»),
       §5.10.3 pasa de «no es un problema de acceso a la informacion: es la combinacion de
       …» a «no parece explicarse en lo esencial … la lectura mas plausible … con la
       salvedad que el ambito `completo` deja abierta», §5.10.2 anade la misma salvedad al
       cierre de Q1 y pasa la regla O4 a condicional («una derrota clara seria una conclusion
       fuerte»); §6.2 (septima limitacion) ya sostenia la version prudente y se conserva
       como destino de las nuevas remisiones. Item 21: las tres frases del 2/11 y del
       $9\times2/11\approx1{,}6$ salen de §5.11.3; el parrafo conserva «preregistrado» ahora
       referido a la correccion BH y a la replica en validacion. En codigo, decision del
       autor: se retira el `print` de `main()` y se reescribe el docstring de
       `expected_first_by_chance` (se conserva por su test, no se reporta; motivo declarado);
       funcion y test intactos, la suite sigue en 470. §6.3: «han mostrado, en la
       literatura reciente …» pasa a «se han propuesto precisamente para … ; si sobre esta
       serie lo consiguen es una cuestion empirica abierta» — sin cita nueva, por decision:
       la tesis no necesita la afirmacion y el item 7 fijo la regla de no invocar literatura
       que no se pueda sostener.

    h. **Dejado a proposito para la fase 4 (item 30):** `05:654` («Lo que este analisis
       descarta», veredicto meteorologico limpio y preregistrado), `05:736` («lo que la
       curva zanja»), `05:1038` («El resultado es inequivoco»), `02:78` («como demuestra el
       capitulo 5»), y los «confirma» de §5.5/§4.4/PI4 sobre `feat_day_type_festivo`.

    i. **Resumen y abstract volvieron a dos paginas** con las ~25 palabras anadidas a cada
       uno (los dos bloques de palabras clave cayeron a la pagina siguiente y el auditor
       reporto ademas «0 palabras clave», sintoma del mismo desbordamiento). Recuperado sin
       retirar ninguna afirmacion: `\vspace` previo a las palabras clave 0,5 → 0,2 cm en
       ambos, y ~30 palabras de tejido conectivo en el resumen («En cuanto a su relacion con
       trabajos previos, este …» → «Este …», fechas en ISO, «decisiones operativas reales» →
       «decisiones operativas», «ninguna de las cifras aqui presentadas» → «ninguna cifra aqui
       presentada»); en el abstract, solo la primera de esas. Resumen 469 palabras, abstract
       480, ambos ≥ 400 y en una pagina, palabras clave 6/6.

    **Estado, medido:** `python -m pytest` → **470/470**; `latexmk` desde limpio → **97
    paginas, cuerpo 73 → 74** (una linea de §5.6.1 empuja un flotante; aceptado por
    decision expresa del autor, sin recortar contenido aprobado, item 33), 0 errores, 0
    referencias/citas indefinidas, biber 0 avisos; `audit_guia_docente.py --strict` → todos
    los bloqueantes en OK, 0 infracciones de margen; C1 **36.876 → 37.492** (+616, de los
    que ~90 son la celda ampliada del cuadro 2.1; direccion admite «36.000 o algo mas»), C4
    30.679, C5 28.928. `sections/`: +182/−139 lineas en siete ficheros. Comparacion literal
    PI1–PI5 entre §1.2 y §5.12: cinco identicas. `grep` de las formulaciones retiradas
    («mismo pasado», «mismo conjunto de informacion», «same history», «libre de fuga»,
    «garantias anti», «2/11», «literatura reciente», «informacion exogena sin»): 0 en
    `sections/`.

24. **Revision formal de direccion, fase 4 de 5: lenguaje, registro y terminologia (items
    22-30).** Cambia como se lee la memoria, no lo que afirma; cada item se barrio por clase
    sobre `sections/*.tex` y sobre los literales de generador que alimentan `tables/` y
    `reports/figures/`, no por la instancia reportada. Ningun modelo se reentrena, ninguna
    cifra medida cambia, `data/processed/` y `models/` sin tocar. 165 sustituciones,
    aplicadas por un motor que exige que cada cadena aparezca exactamente el numero de
    veces esperado antes de escribir nada (una referencia de linea obsoleta aborta, no
    salta).

    a. **Item 22.** `03:264` «la reparto modal» → «el reparto modal»; las otras dos
       apariciones ya eran masculinas.

    b. **Item 23, «ingestion» → «ingesta».** Seis sitios en `01` (incluida la cabecera de
       OE1 que la fase 3 dejo intacta a proposito, y la barra del Gantt) y `03`, mas el
       literal `PHASES` «Fase 1» del generador; `tabla_cronograma_fases` regenerada por el
       exportador. `src/ingestion/` (rutas de codigo) y el cuaderno 07 (6 apariciones en
       celdas markdown, fuera de la memoria) no se tocan.

    c. **Item 24, «prerregistrado».** Direccion revierte la decision de 13f (ver la nota
       reescrita alli). El estado previo ni siquiera era la locucion fija que 13f defendia:
       convivian «pre-registrado» (×7) y «preregistrado» (×9). Los 16 sitios de `05` y `07`
       pasan a «prerregistrad-», y los dos literales del generador (`PHASES` «Fase 5b» y
       el veredicto del experimento A) tambien; `tabla_cronograma_fases` y
       `tabla_criterio_experimento_a` regeneradas. `pre-registered` del abstract ingles se
       conserva. Ninguna prueba fijaba la cadena espanola.

    d. **Items 25-26, anglicismos.** Inventario completo de 38 terminos ingleses en prosa
       espanola (abstract y `\texttt{}` excluidos), con tres tratamientos: **introducir en
       el primer uso del cuerpo y mantener** (pipeline, baseline, walk-forward,
       out-of-fold/OOF, in-sample, holdout, fold, dropout, bootstrap, p-hacking, one-hot,
       random forest, trailing, bin, research gap), **sustituir** (grid search → busqueda en
       rejilla, ranking → clasificacion, brief → documentacion inicial, kernel → nucleo,
       pines → versiones fijadas, lag → retardo en prosa, same-day → del mismo dia, join →
       union, Dashboard → Cuadro de mando en Gantt y generador) y **conservar** (los
       identificadores de particion `test`/`val`/`train`, acronimos, codigo, y los terminos
       ya introducidos: stacking, gradient boosting, warm-up, inner join). «Ingenieria de
       caracteristicas» ya era la unica forma en la memoria: nada que hacer. «5 bloques» /
       «cinco folds» alternaban para el mismo esquema OOF: el cuerpo se unifica en «folds»
       tras introducirlos en `02:107`; resumen y abstract («cinco bloques», «five-block») se
       dejan, son resumenes autocontenidos. Item 26: «tests» → «pruebas automatizadas» en
       los 12 sitios de suite (titulo del Anexo B incluido, resumen «automaticas» →
       «automatizadas»); los **cuatro sitios en sentido estadistico** (`05:1066` test
       pareado, `05:1090` tests dependientes, `05:1110/1116` «sin test») pasan a
       «contraste», no a «prueba automatizada», porque es otro sentido. Rutas
       `tests/test_*.py` y `pytest` intactos. Generador: columna «Tests acumulados» →
       «Pruebas acumuladas» (`PHASES`, docstring) y la cadena en
       `test_export_latex_tables.py:112`; etiqueta de familia «Ensemble» → «Ensamble» en
       `report_tables.py` (sin prueba que la fijara); `tabla_comparacion_{test,val}`
       regeneradas.

    e. **Item 27, nombres de modelo.** Distincion fijada: el **identificador tecnico**
       (`\texttt{xgboost\_alone}` ×57, `hybrid`, `hybrid\_weighted`, `lstm\_alone`,
       `ensemble\_*`, `xgboost\_residual`) no cambia —es lo que usan las tablas generadas
       (columna `Modelo`, cabeceras) y `PROSE_CLAIMS`, que ademas fija cifras y no nombres,
       verificado—; el **nombre de prosa** se unifica. `xgboost_alone` → «XGBoost
       independiente» (formula de direccion) sustituye a las tres variantes que convivian:
       «XGBoost-alone» (`05:347`, titulo de `sec:por_que_ensemble`, `05:507`, `07:274`),
       «XGBoost directo» (`05:671/679/763`) y «XGBoost solo» (celda del cuadro 2.1, en el
       generador). La glosa descriptiva «un XGBoost entrenado directamente sobre las
       mismas variables» se conserva donde *define* el control (resumen, abstract, `01:38`,
       `04:390`, `05:339`): es una explicacion, no un tercer nombre. Diez pies de figura y
       cuadro llevaban el identificador sin `\texttt{}` (`05:97/121/137/154/245/263/271/
       302/310/347`): envueltos. Generador: «Peso XGBoost-alone» → «Peso XGBoost
       independiente» (`:785`, `:1095`) y la cadena en `test_export_latex_tables.py:294`;
       `tabla_ensemble_pesos` y `tabla_comparativa_estado_arte` regeneradas. Figura:
       titulo del eje de `fig_dispersion_errores_sarimax_xgb` (`memoria_figures.py:402`).
       Los demas nombres de prosa ya eran consistentes (hibrido / hibrido reponderado /
       etapa 1 en solitario / modelo residual / ensamble equiponderado / ponderado por el
       inverso del MAE / persistencia estacional).

    f. **Item 28, R².** Ningun `.tex` contenia «R2»: la prosa usa `$R^2$` y las 32 tablas
       «R²». Lo que direccion vio esta en la **figura 5.1** (`model_comparison_bar_test.png`):
       el titulo de subpanel «R2 (mayor mejor)» de `dashboard_figures.py:108`, construido
       con la clave de columna. Corregido solo el rotulo mostrado (la clave `"R2"` de
       `metrics.py` es contrato de codigo y no cambia); figuras reexportadas. Los dos pies
       que escribian «R²» en Unicode (`05:70`, `05:121`) pasan a `$R^2$` para una sola
       convencion en prosa. Fuera de alcance: el cuaderno 07 muestra «R2» como nombre de
       columna de pandas en sus salidas guardadas.

    g. **Item 29, expresiones defensivas y coloquiales.** 33 sitios (lista completa en el
       plan de la fase; muestra): «no es un baseline decorativo» → formula de direccion
       («se incorpora como un modelo de referencia competitivo y metodologicamente
       relevante») seguida del hecho medido; «no se fabrica ningun valor p … la respuesta
       honesta» → «no se realiza inferencia estadistica debido al reducido tamano muestral:
       se reporta unicamente el MAE puntual con su n»; `04:215` «la respuesta honesta» →
       «los resultados obtenidos indican»; «Este TFM no oculta ese resultado negativo …» →
       «convierte ese resultado negativo en el objeto de estudio central»; «con la misma
       honestidad / transparencia» (`01:35`, `04:189`), «con la misma prominencia que
       tendria un resultado positivo», «que un lector atento habra advertido», «y este
       proyecto no se conforma con esa ambiguedad» eliminados; «hombre de paja» → «modelo de
       referencia»; «liston», «barra honesta» → «umbral»/«referencia»; «rentable» →
       «ventajoso»; «a ojo» → «mediante inspeccion manual»; «tecleada a mano» →
       «introducida manualmente»; «honesta/honestamente» (×8) → «valida», «rigurosa» o
       suprimido; «no es un ejercicio academico … reales» (`01:9`) alineado con el resumen
       de la fase 3. Conservados a proposito: el parrafo de p-hacking (`05:961-965`,
       justificacion de diseno), «no frente a un baseline trivial», «Tampoco debe leerse
       como una arquitectura sin merito» (es exactamente el encuadre que pide el item 31).

    h. **Item 30, verbos categoricos.** Regla aplicada: el verbo categorico **se mantiene**
       cuando su objeto es (a) un recuento o diferencia medidos sobre un artefacto
       persistido, (b) un veredicto prerregistrado evaluado en codigo, o (c) una garantia
       mecanica fijada por una prueba; **se suaviza** cuando el objeto es una interpretacion
       (una clasificacion de importancia, una lectura teorica); los intensificadores
       retoricos («inequivoco», «sin ambiguedad», «decisiva», «demostrable») se sustituyen
       por la forma neutra aunque la afirmacion siga siendo categorica, porque el recuento
       ya lleva la fuerza. Sitios de 23h: `05:654` («Lo que este analisis descarta …
       sobre la mesa» → «Lo que este analisis indica es que no se observa senal
       meteorologica retardada sin explotar»: un resultado nulo no descarta); `05:736`
       («zanja» → «si muestra», la meseta ~166k por encima es medida); `05:1038` («El
       resultado es inequivoco» → «El resultado es negativo en todos los subgrupos
       inferenciales»: **sigue categorico**, 0 de 9 con BH, se sustituye el adjetivo por el
       recuento); `02:78` («demuestra» → «muestra», precedente de la fase 3). Barrido
       adicional (`02:50` «confirma … la inmensa mayoria» → «respalda … la mayor parte»,
       `02:103` «demuestran» sobre Bergmeir/Cerqueira → «muestran», `02:135` «sin
       ambiguedad» fuera, `05:107` «es la evidencia de» → «es indicativa de», `05:337` «e
       inequivoca» fuera, `05:623` «de forma inequivoca» → «con claridad» (6,52× frente a
       umbral 2, sigue categorico), `05:709` «evidencia decisiva» → «principal» (la propia
       seccion llama «parcial» al resultado), `05:799` «demostro» → «mostro», `05:905` «de
       forma demostrable» → «medible», `06:49` «PI5 confirma» → «muestra»).

       **Se mantienen categoricos, con motivo:** `05:340-342` «no, en ninguna de las dos
       comparaciones y en ninguna de las dos particiones» (cuatro diferencias medidas);
       `05:1041-1043` «En 0 de los 9 subgrupos …» y `05:1094` «significativamente peor»
       (recuento bajo diseno prerregistrado, bootstrap con BH q=0,10 fijado en codigo);
       `05:194` «Ningun modelo domina al otro» (leido del cuadro 5.3); `04:56` «El
       resultado confirma la premisa» (2,70× en puentes, medido); `04:143` «descartando asi
       un desfase temporal accidental» (comprobacion de implementacion, 163.627 frente a
       909.082); `04:229` «se probo y se descarto el barajado» (decision sobre MAE medido);
       `05:44`, `05:62` (la figura y el colapso de los ingenuos son las mismas cifras del
       cuadro); `05:553`, `05:767` «confirmatorio, no exploratorio» (termino tecnico del
       analisis prerregistrado); `05:797` «corroboracion mecanica» y `05:1180` «confirma
       mecanicamente» (criterio del 5 % evaluado en codigo; §5.12 ya lo reformulo la fase
       3, item 17, y no se reabre); `05:1124`, `05:1130` (recuento); `06:12` «se ha
       cumplido en su totalidad» (item 31 de direccion); `04:110`, `05:543`, `07:230`
       (negaciones: suavizarlas invertiria el sentido); `03:288` (hecho mecanico del
       calendario); los diez puntos del Anexo B (cada uno con su prueba).

    i. **HALLAZGO DE COHERENCIA INTERNA, destapado por el item 30 (quinto de la revision,
       tras Monje 21c, Chen 22a, «sin termino autorregresivo» 23e y Bates-Granger 23f):
       el mecanismo de la etapa 2 se «confirmaba» con una cuota de ganancia por impureza.**
       §4.4 (`04:365`), §5.5 (`05:319`, pie `05:330`), la respuesta a PI4 (`05:1163`) y
       §6.1 (`06:42`) usaban la ganancia de `feat_day_type_festivo` (0,125, primera de la
       lista) para «confirmar» que la etapa 2 aprende «exactamente» la senal de calendario.
       Pero §5.10.3, cincuenta paginas despues, establece —con el bloque meteorologico, que
       acaparaba el 51,6 % de la ganancia y resulto no aportar nada en la ablacion— que
       una cuota de ganancia por impureza **no es evidencia de utilidad predictiva y no debe
       leerse como tal sin un contraste de retirada**. La memoria aplicaba a la senal de
       calendario un estandar mas laxo que el que se impone a si misma con la meteorologia.
       No es una cuestion de registro: es una correccion de coherencia interna que el
       barrido de «confirma» hizo visible. Se corrige **solo la mitad basada en ganancia**
       («confirma … exactamente» → «apunta a / es coherente con»); la mitad **medida** de
       PI4 (la etapa 2 reduce el MAE de la etapa 1 un 16,6 % en test y un 58 % en
       festivos; `05:206` «muestra de nuevo que el mecanismo de correccion funciona») sigue
       categorica, y la respuesta a PI4 sigue siendo «Si». PI1-PI5 literalmente identicas
       entre §1.2 y §5.12 (verificado por script). Decision expresa del autor en la revision
       del plan.

    j. **Figuras: verificado byte a byte.** `export_figures` reescribe las 34; `git status`
       marca exactamente las tres esperadas (`model_comparison_bar_{test,val}.png` por el
       rotulo R², `fig_dispersion_errores_sarimax_xgb.png` por el eje) y ninguna otra.
       Tablas: exactamente las seis esperadas.

    **Estado, medido:** `python -m pytest` → **470/470** (ninguna prueba anadida ni
    retirada; dos cadenas de columna actualizadas en `test_export_latex_tables.py`);
    `latexmk` desde limpio → **97 paginas, cuerpo 74** (sin cambio), 0 errores, 0
    referencias/citas indefinidas, biber 0 avisos; `audit_guia_docente.py --strict` → todos
    los bloqueantes en OK, 0 infracciones de margen; C1 **37.492 → 37.483** (−9; las ~45
    palabras de las introducciones de anglicismos quedan compensadas por el item 29), C4
    30.679 → 30.671, C5 28.928 → 28.898 (−30); resumen 469 → 468 palabras, abstract 480,
    ambos en una pagina con 6 palabras clave. `grep` de las formulaciones retiradas de las
    fases 1-3 mas las de esta fase («ingestion», «pre-registr», «preregistr»,
    «XGBoost-alone», «XGBoost directo», «XGBoost solo», «suite de tests», «no a ojo»,
    «honest», «hombre de paja», «zanja», «sin ambiguedad», «decorativ», «lector atento»,
    «sobre la mesa», «si cabe», «tecle», «R2», «brief», «kernel», «ranking», «same-day»,
    «grid search»): 0 en `sections/`, salvo el `pre-registered` del abstract ingles y el
    «inequivoca» negado de `04:110` (lista de conservados). `sections/`: +174/−175 lineas
    en ocho ficheros. `git status`: nada bajo `models/` ni `data/`.

25. **Revision formal de direccion, fase 5 de 5: verificacion final (regresion, coherencia
    entre fases, codigo y repositorio, formato).** Fase sin cambios planificados: su objeto es
    establecer que las cuatro rondas anteriores dejaron la memoria correcta, coherente y
    alineada con el codigo y con el arbol publicado. Toda edicion que produce es un defecto
    corregido, no una mejora; se listan en (f). Ningun modelo se reentrena; `data/` y
    `models/` sin tocar. **Cada resultado lleva la medicion detras**: las sondas viven en el
    scratchpad de sesion y sus salidas se transcriben aqui; no entra ningun script nuevo en el
    repositorio.

    a. **Puerta 0: la fase 4 no estaba confirmada en git.** `HEAD` era `a0018a2` (fase 3) y el
       arbol de trabajo llevaba exactamente la huella de la fase 4 (25 ficheros: 8 `sections/`
       con +174/−175, 6 cuadros, 3 PNG, 4 `.py`, `TFT.pdf` y registros). Se confirma como
       commit propio (`f062c71`) antes de cualquier comprobacion, para que el historial siga
       siendo fase a fase y la reexportacion tenga un `HEAD` que la contenga. El motor de la
       fase 4 habia escrito CRLF en los 19 ficheros de texto que toco; git los normaliza a LF
       al confirmar (aviso, no defecto).

    b. **Regresion contra lo que direccion dio por correcto (items 8-12, 18, 19, 31-34).**
       Metodo: para cada item, localizar el pasaje, extraer con `git diff -U0 6f9e162 HEAD`
       (el estado que direccion reviso frente al actual) los *hunks* que lo tocan, y leerlos.
       Resultado: **ninguno de los once items alterado.** Los pasajes de los items 8 y 34 no
       tienen ningun hunk; los demas solo llevan las ediciones registradas en 21-24
       (`\cite`→`\textcite`/`\parencite`; «exógenas»→«predictoras»; «la reparto»→«el reparto»
       dentro del pasaje del item 10; «trailing»→«retrospectiva (\emph{trailing})»;
       «honesta»→«sólida» en la extension meteorologica del item 12; «confirma»→«muestra» en
       PI4/PI5; la reformulacion de conjuntos anidados del item 16 en §6.1; «no es un
       baseline decorativo»→formula de direccion en §6.1). Los siete MAE de test que cita el
       item 18 estan en `tabla_comparacion_test.tex` (139.681 / 139.725 / 156.121 / 163.627 /
       220.286 / 234.223 / 280.952); §5.9-5.11 siguen numeradas asi en `TFT.toc`; las seis
       limitaciones del item 32 estan en los siete parrafos de §6.2; §6.1 mantiene «se ha
       cumplido en su totalidad» (item 31, conservado categorico en 24h).

    c. **Coherencia entre fases.**
       - Formulaciones retiradas (listas de las fases 3 y 4, 35 cadenas) sobre `sections/`:
         **0**, salvo las dos conservadas a proposito (`src.ingestion.unify`, ruta de codigo,
         `07:245`; `pre-registered` del abstract ingles). «variables exógenas» aparece 8 veces,
         todas en el sentido de informacion externa que la fase 3 conservo (vector exogeno de
         SARIMAX `04:120`, eje (c) del cuadro 2.1 `02:237/249`, bloque meteorologico `03:237`,
         `02:74`, `04:71/365/387`); ninguna dice «161 variables exógenas».
       - Las tres formulaciones de la fase 3 leidas en los nueve lugares del cruce (resumen,
         abstract, §2.3, §4.7, §5.6.1, §5.10.3, §5.12, §6.1, §6.2): conjuntos **anidados, no
         complementarios** (`00:33-36`, `00:82-86`, `02:200-209`, `04:454-464`, `05:378-399`,
         `06:32-35`, `06:60-62`); paridad **«reduce la plausibilidad … mixto/ambiguo»**
         (`05:886-893`, `05:909-914`, `05:1184-1190`, `06:142-149`); Bates-Granger
         **«coherente con / en consonancia»** (`00:41-43`, `00:92-94`, `02:196-198`
         «contrasta», `04:476-478`, `05:144-149`, `05:538-544`, `05:1175`, `06:54-58`,
         `06:60-62`). Las ediciones de la fase 4 en esos mismos parrafos (items 29-30) no
         reintrodujeron «demuestra», «descarta» ni «mismo conjunto».
       - Citas: `\textcite` 30, `\parencite` 13 (14 antes de (f)-F-g), `\cite` desnudo 0,
         forma numerica `[n]` 0. Lista: 23 claves en el `.bib` = 23 citadas, ninguna huerfana
         en ningun sentido; `TFT.blg` 0 avisos. Las cuatro referencias de la fase 2 en ambos
         sentidos: `yazicioglu2025passenger` 4 sitios, `chen2019emd` 2, `arana2014weather` 1,
         `singhal2014weather` 1, cada una con su entrada.
       - Generadores frente a artefactos: `export_latex_tables` y `export_figures` reejecutados;
         `git status` sobre `tables/` y `reports/figures/` **vacio** (32 cuadros y 34 PNG
         byte a byte identicos). Los renombrados de la fase 4 leidos en el artefacto: «Peso
         XGBoost independiente» (cabecera de `tabla_ensemble_pesos`), «Ensamble» (familia en
         `tabla_comparacion_*`), «Pruebas acumuladas», «Ingesta», «prerregistrado»; en las
         figuras, «R² (mayor mejor)» en `model_comparison_bar_test.png` y «Residuo XGBoost
         independiente» en el eje de `fig_dispersion_errores_sarimax_xgb.png` (inspeccion
         visual de los PNG reexportados).

    d. **Coherencia global del documento.**
       - `test_prose_claims_match_artifacts.py`: 17/17 + 2. PI1-PI5 literalmente identicas
         entre §1.2 y §5.12 (script).
       - **Cifras fuera del manifiesto**: como ningun artefacto cambio en las fases 1-4, una
         cifra de prosa solo puede estar mal si una fase edito digitos. Diferencia del
         multiconjunto de tokens numericos por fichero entre `6f9e162` y `HEAD`: 6 retirados,
         17 anadidos, **los 23 explicados** por ediciones registradas —fechas ISO del resumen
         (fase 3), «161» reenunciado con «predictoras» (item 14), «etapa 1/2» y «28» dentro de
         las frases de conjuntos anidados (item 16), «15 minutos / 14 días» de la fila de Chen
         (fase 2, leidos del articulo) y los tokens retirados `2`, `11`, `9`, `6`, `0` del
         razonamiento 2/11 (item 21)—. Ninguna cifra medida se movio. Tras (f): delta 0.
       - `\ref`/`\label`: todo `\ref` resuelve salvo `LastPage` (lo define el paquete
         `lastpage` en compilacion; `TFT.log` 0 «undefined»); todas las etiquetas `fig:`/`tab:`
         referenciadas al menos una vez; 27 PNG incluidos, todos en disco (los 7 restantes de
         los 34 se exportan para el cuadro de mando y no se incluyen: decision previa);
         32 cuadros `\input`, todos en disco. Numeracion de flotantes leida de `TFT.aux`:
         28 figuras y 33 cuadros, secuencial y sin duplicados en todos los capitulos.
       - Resumen ↔ abstract: mismas cifras (1.310, 161, 470, 234.223/163.627/156.121,
         139.725/139.681, 0,9755), misma secuencia de afirmaciones, con **una asimetria**
         conocida y dejada a proposito: el abstract lleva la frase «The contribution … is
         therefore methodological … under a pre-registered protocol …» y el resumen no (la
         fase 3 la recorto para recuperar la pagina unica; anadirla volveria a empujar las
         palabras clave, incidencia 23i). Sin contradiccion; 467 / 479 palabras, una pagina
         cada uno, 6 palabras clave.

    e. **Codigo y repositorio.**
       - `python -m pytest -q` → **470 passed** (207 s); `--collect-only` → 470. Las seis
         apariciones de 470 en prosa: `00:24`, `00:75`, `02:273`, `07:81`, `07:302`, `07:368`
         (la prueba fija solo `00` y `07`; la lista de seis es la comprobacion manual, mas
         estricta).
       - Rutas e identificadores: 198 tokens `\texttt{}` en `sections/`; 135 resueltos por
         script (modulos `python -m`, rutas, nombres de prueba, guardias, constantes), 29
         comprobados a mano (los 12 `*.parquet` citados sin directorio existen en
         `data/processed/`; `CRTM_…xlsx` en `data/raw/` con hojas `['mensual','diaria']`;
         `stage1_residual_anatomy.py`, `test_prose_claims_match_artifacts.py`,
         `test_phase{7,8,9}_adds_no_model…`, `slcolor` en `preamble.tex`, los nueve pines de
         `requirements.txt`, `GLOBAL_SEED = 42`, `FOURIER_EPOCH`, `#E8590C`), 34 son cadenas
         de prosa (`'none'`, `header=1`, `shift(1)`…). **0 sin resolver.**
       - **Repositorio publico: obsoleto, y ahora realmente publico.**
         `https://github.com/AdayCuestaCorrea/TFM` responde **HTTP 200 sin autenticar** (era
         404/privado en 19g), de modo que «es de acceso público» del Anexo C es cierto y la
         obsolescencia es visible. `main` remoto = `92c000e` = `TFM-publicado` local (dos
         commits), arbol identico a `6f9e162` menos el conjunto de exclusion, **rederivado**
         por `comm` de `git ls-tree`: exactamente `docs/LaTeX/Images/image{4,5,6}.png` y
         `docs/references/Summary.md`. Entre `6f9e162` y `HEAD` no se anadio ni borro ningun
         fichero (`git diff --name-status` solo `M`); sin ficheros sin seguimiento; el `.docx`
         de la revision y `docs/references/*.pdf` estan en `.gitignore`. **El conjunto de
         exclusion se mantiene.** Reexportacion y puerta de verificacion en (h).

    f. **Defectos corregidos (ocho), con su origen.** Ninguno mueve una cifra, un cuadro, una
       figura ni una pagina; decisiones expresas del autor sobre la tabla de la fase.
       - F-a `07:81` «pruebas automatizadas, recolectadas y **ejecutados**»: concordancia rota
         al cambiar «tests»→«pruebas» (**regresion de la fase 4**, diff: era «recolectados y
         ejecutados»). → «ejecutadas».
       - F-b `07:368` «pruebas **automáticas**»: el barrido del item 26 dejo este sitio (Anexo
         D). → «automatizadas».
       - **F-c `05:847` — HALLAZGO DE EXACTITUD (sexto de la revision)**: «(411.142, la cifra
         citada en la sección §5.12 y en el capítulo 4)» era **falso**: 411.142 no aparece en
         §5.12 ni en el capitulo 4, solo en el pie del cuadro E.8 (`07:495`); la remision quedo
         obsoleta en la pasada de reduccion (incidencia 18). → «(411.142; cuadro
         \ref{tab:paridad_lstm})».
       - F-d `06:226` «es **la primera** de las líneas enunciadas en §6.3»: es la segunda (la
         primera es el modelado por operador). → «la segunda».
       - F-e `07:273` paso `ensemble_baseline` remitia a `sec:mecanismo_ensemble` (§5.8); el
         experimento B es §5.7.2. → `\ref{sec:por_que_ensemble}`.
       - F-f `00:46`, `00:100` «Surribas Sayago et al.» → «Surribas-Sayago et al.» en resumen y
         abstract (nombran la obra citada; es la forma de la lista de referencias, del cuerpo y
         de la propia publicacion). La portada (`TFT.tex:53`), que lleva el nombre de la
         persona, no se toca: el autor lo confirmara con direccion.
       - **F-g `02:141` — HALLAZGO DE EXACTITUD (septimo, de citas)**: «Zhang ajusta primero el
         componente lineal (ARIMA) y corrige después con una red neuronal
         \parencite{hochreiter1997lstm}» atribuia a Zhang (2003) una arquitectura que no uso:
         su red es un perceptron multicapa, no una LSTM. Sobrevivio porque era un `\cite`
         numerico desnudo que la fase 1 convirtio en su sitio. → se retira la cita de esa
         frase; Hochreiter y Schmidhuber siguen citados en `04:155`, donde se introduce la LSTM.
       - F-h `04:435` «El segundo control de la sección anterior, SARIMAX»: §4.5 describe
         `xgboost_alone` y `hybrid_weighted`, no SARIMAX. → «SARIMAX (sección
         \ref{sec:baselines}) y el control \texttt{xgboost\_alone} de la sección anterior».

       **Dejados a proposito, por decision del autor:** F-i (asimetria resumen/abstract, (d));
       F-j «documento de contexto técnico del proyecto» (`01:106`) / «directriz del contexto
       técnico» (`07:45`), cabo suelto ya registrado (14, 22c) y fuera de la revision.
       **Observado, fuera de alcance:** la leyenda de `model_comparison_bar_{test,val}.png`
       lleva las claves de familia en ingles («naive baseline», «single-stage», «LSTM family»,
       «ensemble»), preexistente y no incluido en los items de direccion.

       **Patron, actualizado:** con F-c y F-g son **siete** los defectos de exactitud
       destapados por la revision formal (nombres de Monje 21c, nombres de Chen 22a, «sin
       término autorregresivo» 23e, Bates-Granger invertido 23f, «confirma» con ganancia por
       impureza 24i, remision falsa a 411.142, cita de LSTM atribuida a Zhang), ninguno
       visible en el texto compuesto: todos se detectaron cotejando contra la fuente (el
       propio documento, el articulo, la formula), nunca por el aspecto de la frase.

    g. **Formato (guia 14MBID), medido con `audit_guia_docente.py --strict` sobre el PDF
       final:** NimbusSanL (Helvetica) OK; 10,91 pt (clase 11pt) OK; interlineado 1,147 OK;
       margenes 4,40/2,19/3,00/3,00 cm OK; **0 infracciones de margen**; cuerpo **74 paginas**
       (40-80); resumen **467** palabras y abstract **479**, 6 palabras clave, una pagina cada
       uno. Recuento: C1 **37.483 → 37.467** (−16, por F-c y F-g y los dos guiones), C4 30.655,
       C5 28.885. C1 excede el limite de 30.000; segun el autor, la coordinacion del master
       acepto por escrito la cifra actual —esa aceptacion **no esta en el repositorio**
       (`docs/CONSULTA_DIRECCION.md` se retiro) y aqui se consigna como informacion del autor,
       no como hecho verificado—. `latexmk` desde limpio (con `TFT.bbl`/`.bcf` borrados) →
       **97 paginas, 0 errores, 0 referencias/citas indefinidas, biber 0 avisos**.

    h. **Reexportacion del repositorio publico** (preparada; el autor ejecuta el push):
       `git archive HEAD` de `TFM-HISTORICO` extraido sobre `TFM-publicado`, retirados los
       cuatro ficheros del conjunto de exclusion, `git add -A`, commit **encima** de `92c000e`
       (nunca reescritura ni *force push*). Puerta: (i) `git ls-files` publicado = `HEAD` de
       `TFM-HISTORICO` menos las cuatro exclusiones; (ii) 0 blobs distintos entre ambos en
       toda ruta comun; (iii) `git log` con exactamente 3 commits y padre `92c000e`; (iv) arbol
       limpio, nada bajo `data/` ni `models/`; (v) clon fresco del repositorio local → `data/`
       copiado (nunca versionado) → `pytest` con el venv del proyecto y `latexmk` en el clon.
       Tras el push, repetir (v) contra el **clon publico** (la prueba mas fuerte, que la
       incidencia 19 no pudo ejecutar mientras el repositorio era privado). Resultados de la
       puerta: ver el final de esta incidencia.

    i. **Pendientes que se comunican a direccion, no se ocultan:** la ficha de Surribas-Sayago
       sin campo `editor` (biblatex-apa la compone como articulo de actas, valido en APA 7;
       una linea cuando direccion facilite los editores); la consecuencia «conjuntos anidados
       ⇒ la combinacion no supera al mejor modelo sobre el conjunto mayor» enunciada sin
       atribuir a Granger (1989) por falta de fuente verificada de *forecast encompassing*
       (23d); la extension bajo C1; el repositorio publico obsoleto hasta el push.

    **Estado, medido:** `python -m pytest` → **470/470**; `latexmk` desde limpio → **97
    paginas, cuerpo 74**, 0 errores, 0 indefinidas, biber 0 avisos; auditoria → todos los
    bloqueantes OK, 0 infracciones de margen; C1 37.467; resumen 467 / abstract 479 en una
    pagina; PI1-PI5 identicas; `PROSE_CLAIMS` 17/17; formulaciones retiradas 0; cuadros y
    figuras regenerados byte a byte identicos; `sections/`: 8 lineas en 6 ficheros.

    **Puerta de la reexportacion (h), medida.** Commit `d03962d` en `TFM-publicado`, padre
    `92c000e`, 3 commits en total, arbol limpio. (i) `git ls-files` publicado = `HEAD` de
    `TFM-HISTORICO` menos las cuatro exclusiones: **identico, 172 ficheros**. (ii) Blobs
    distintos en toda ruta comun: **0**. (iv) Nada bajo `data/`; `models/` lleva los mismos
    cinco artefactos publicados desde `094ebc6`, blob a blob identicos. (v) Clon fresco del
    repositorio local con `data/` copiado: `pytest` **470 passed** (162 s); `latexmk` en el
    clon → **97 paginas, 0 errores, 0 indefinidas, biber 0 avisos** (las fuentes publicadas
    compilan por si solas). Pendiente del autor: `git push origin main` (*fast-forward*) y
    repetir (v) contra el clon publico.

---

## 6. Cómo regenerar todo desde cero

Si solo se regenera la memoria a partir de artefactos ya persistidos:

```powershell
# 0. (fase 7) curva de aprendizaje LSTM -> data/processed/lstm_learning_curve.parquet
#    Reentrena val-only, no persiste modelo. Un solo comando genera el artefacto completo:
#    3 semillas por fraccion, mas las semillas 3 y 4 en la fraccion 1.00 (base del control
#    de paridad de la fase 8). assemble_full_curve deduplica (fraccion, semilla).
python -m src.evaluation.lstm_learning_curve

# 0b. (fase 8) controles diagnosticos -> feature_block_ablation.parquet,
#     lstm_parity_control.parquet. Reentrenan (val-only / holdout interno);
#     NO persisten ningun modelo bajo models/.
python -m src.evaluation.feature_block_ablation
python -m src.evaluation.lstm_parity_control

# 0c. (fase 9) desglose por subgrupos -> subgroup_breakdown.parquet,
#     subgroup_rolling_mae.parquet. Reanalisis puro de las predicciones ya
#     persistidas; NO reentrena ni predice nada.
python -m src.evaluation.subgroup_breakdown

# 1. Figuras (34 PNG en reports/figures/)
python -m src.evaluation.export_figures

# 2. Tablas (32 .tex en docs/LaTeX/tables/)
python -m src.evaluation.export_latex_tables

# 3. Compilación (latexmk lanza biber solo; si queda un TFT.bbl de BibTeX, borrarlo antes)
cd docs/LaTeX
latexmk -pdf -interaction=nonstopmode -halt-on-error TFT.tex
```

El paso 0 reentrena la etapa 1 en 18 ajustes (6 fracciones x 3 semillas) para la curva de
aprendizaje de §5.9; si `lstm_learning_curve.parquet` ya existe puede omitirse. Los pasos
1-3 no reentrenan ni re-predicen ningún modelo: `python -m
src.evaluation.stage1_residual_anatomy` (§5.9), `python -m
src.evaluation.subgroup_breakdown` (§5.11, reanálisis puro de las predicciones ya
persistidas) y los dos exportadores leen exclusivamente de
`data/processed/`, `models/` y las constantes estáticas descritas en §2-§3.

La secuencia completa de reproducción desde datos crudos —ingestión, features,
baselines, `lstm.window_comparison`, `lstm.oof`, `lstm.final_model`,
`xgboost_residual`, `xgboost_alone`, `combine`, `full_comparison`,
`xgboost_residual_weighted`,
`ensemble_baseline`, `sensitivity_report`, y luego los dos exportadores, el cuaderno
principal de flujo, `pytest` y `latexmk`— está enumerada paso a paso en el Anexo C de la
memoria (`sections/07_anexos.tex`).

## 7. Cuadernos

| Cuaderno | Papel | Ejecuta modelos | Se versiona con salidas |
|---|---|---|---|
| `notebooks/06_dashboard.ipynb` | Panel de resultados interactivo (ipywidgets); etapa de *despliegue* del ciclo KDD citada en §1.3 y Anexo C | No — solo lee artefactos | Sí |
| `notebooks/07_flujo_completo.ipynb` | Flujo end-to-end en las seis etapas; se verifica contra `full_comparison.parquet` y falla si alguna métrica se desvía | Sí — todo salvo la etapa 1 LSTM | Sí (`nbconvert --execute --inplace`) |

Ninguno de los dos escribe en `data/` ni en `models/`, y ninguno exporta PNG: kaleido 1.x
bloquea el kernel, de modo que las 34 figuras estáticas se generan siempre con
`python -m src.evaluation.export_figures`, fuera de todo kernel de Jupyter.
