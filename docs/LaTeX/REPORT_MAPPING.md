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
├── Bibliografia_TFT.bib        referencias BibTeX (Fase 10)
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

       ⚠️ **Decisión deliberada: `pre-registrado` NO se corrige a `prerregistrado`.**
       "Prerregistrado" es la forma correcta según la RAE, pero `pre-registrado` es el
       término técnico de la práctica metodológica (traducción directa de
       *pre-registration*), aparece seis veces como locución fija en §5 y en el Anexo B, y
       su reconocibilidad para un lector familiarizado con esa práctica pesa más que la
       ganancia ortográfica. **Una pasada futura no debe "corregirlo".**

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

# 3. Compilación
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
`xgboost_residual`, `xgboost_alone`, `combine`, `xgboost_residual_weighted`,
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
