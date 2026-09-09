"""Phase 9 tests — the subgroup breakdown is a diagnostic re-analysis, not a model.

The load-bearing test is `test_phase9_adds_no_model_to_the_master_comparison`: nothing this
phase produces may enter chapter 5's master comparison. The rest pin the pre-registered
design (subgroup definitions, thresholds, the verdict/replication split), the anti-p-hacking
machinery (moving-block bootstrap with in-replicate re-selection, BH over the inferential
family), the `demanda_anomala` verdict-scope resolution the supervisor's question makes
load-bearing, and the standing empirical result that the hybrid wins nowhere.
"""

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation import subgroup_breakdown as sb
from src.evaluation.dashboard_data import MODEL_NAMES, load_all_predictions
from src.evaluation.metrics import all_metrics
from src.utils.paths import MODELS_DIR, PROCESSED_DIR

# Frozen local copy (the three-file repo idiom): an accidental edit to dashboard_data is
# caught here too. Matches CLAUDE.md's 11-model master comparison as of Phase 5b.
EXPECTED_MASTER_MODELS = {
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
}

PHASE9_NAMES = set(sb.SUBGROUPS) | {
    "subgroup_breakdown",
    "hybrid_win",
    "rolling_mae",
    "demanda_anomala",
}


@pytest.fixture(scope="module")
def preds() -> pd.DataFrame:
    return load_all_predictions()


@pytest.fixture(scope="module")
def criteria(preds) -> pd.DataFrame:
    return sb.evaluate_subgroup_criteria(preds)


# --------------------------------------------------------------------------------------
# The mechanical no-candidate guard
# --------------------------------------------------------------------------------------


def test_phase9_adds_no_model_to_the_master_comparison():
    assert set(MODEL_NAMES) == EXPECTED_MASTER_MODELS

    full = pd.read_parquet(PROCESSED_DIR / "full_comparison.parquet")
    sensitivity = pd.read_parquet(PROCESSED_DIR / "sensitivity_comparison.parquet")

    assert set(full["Model"]) <= EXPECTED_MASTER_MODELS
    assert set(sensitivity["Model"]) == EXPECTED_MASTER_MODELS

    assert PHASE9_NAMES.isdisjoint(set(full["Model"]))
    assert PHASE9_NAMES.isdisjoint(set(sensitivity["Model"]))


def test_phase9_does_not_extend_the_ordering_constants():
    from src.evaluation.sensitivity_report import ORDER as SENSITIVITY_ORDER

    assert len(MODEL_NAMES) == 11
    assert set(SENSITIVITY_ORDER) == EXPECTED_MASTER_MODELS


def test_subgroup_module_persists_no_model_artifact():
    source = Path(sb.__file__).read_text(encoding="utf-8")
    assert "MODELS_DIR" not in source
    assert "joblib.dump" not in source
    assert ".save(" not in source
    assert ".keras" not in source


def test_subgroup_module_never_imports_src_models():
    source = Path(sb.__file__).read_text(encoding="utf-8")
    import_lines = [
        ln for ln in source.splitlines() if ln.strip().startswith(("import ", "from "))
    ]
    assert not any("src.models" in ln for ln in import_lines)
    # This phase trains nothing at all — stronger than Phase 8.
    assert ".fit(" not in source
    assert ".train(" not in source


def test_main_writes_two_parquets_and_nothing_under_models():
    before = {p.name for p in MODELS_DIR.glob("*")} if MODELS_DIR.exists() else set()
    sb.main()
    after = {p.name for p in MODELS_DIR.glob("*")} if MODELS_DIR.exists() else set()
    assert before == after
    assert sb.SUBGROUP_BREAKDOWN_FILE.exists()
    assert sb.SUBGROUP_ROLLING_MAE_FILE.exists()


# --------------------------------------------------------------------------------------
# Pre-registered design
# --------------------------------------------------------------------------------------


def test_subgroups_are_a_module_constant_with_a_frozen_key_set():
    assert isinstance(sb.SUBGROUPS, dict)
    assert set(sb.SUBGROUPS) == {
        "laborable_ordinario", "laborable_puente", "sabado", "domingo", "festivo",
        "fin_de_semana", "entre_semana", "calendario_irregular", "calendario_ordinario",
        "verano_jul_ago", "resto_del_anio", "periodo_navidad", "periodo_semana_santa",
        "demanda_anomala", "demanda_regular",
    }
    assert sb.EXCLUDED_SUBGROUPS == frozenset({"periodo_navidad", "periodo_semana_santa"})


def test_thresholds_are_module_constants():
    assert sb.MIN_N_INFERENTIAL == 20
    assert sb.ANOMALY_K == 3.0
    assert sb.ANOMALY_LAG == 7
    assert sb.BLOCK_LENGTH == 7
    assert sb.N_BOOTSTRAP == 2000
    assert sb.ROLLING_WINDOW == 28
    assert sb.FDR_Q == 0.10
    # The verdict function must reference the constants, not re-typed literals.
    src = inspect.getsource(sb.evaluate_subgroup_criteria)
    assert "VERDICT_SCOPE" in src and "REPLICATION_SCOPE" in src
    assert "MIN_N_INFERENTIAL" in src


def test_verdict_split_is_test_and_replication_is_val():
    assert sb.VERDICT_SCOPE == "test"
    assert sb.REPLICATION_SCOPE == "val"
    assert sb.SCOPES == ["test", "val", "val_test"]


def test_subgroup_sizes_match_the_preregistered_counts():
    masks = sb.subgroup_masks()
    feats = pd.read_parquet(PROCESSED_DIR / "features_daily.parquet").sort_values("date")
    from src.utils.splits import chronological_split, split_masks

    bounds = chronological_split(feats["date"])
    split = split_masks(feats.reset_index(drop=True), bounds)
    m = masks.set_index("date")
    for key, expected in sb.PREREGISTERED_N.items():
        col = m[key].to_numpy()
        for scope in ("val", "test"):
            n = int((col & split[scope].to_numpy()).sum())
            assert n == expected[scope], f"{key} {scope}: {n} != {expected[scope]}"


def test_preregistered_counts_and_subgroups_agree_on_keys():
    assert set(sb.PREREGISTERED_N) == set(sb.SUBGROUPS)


def test_subgroup_masks_are_deterministic_views_not_a_partition():
    masks = sb.subgroup_masks().set_index("date")
    # calendario_ordinario and calendario_irregular DO partition the calendar.
    assert (masks["calendario_ordinario"] ^ masks["calendario_irregular"]).all()
    # demanda_anomala and demanda_regular are disjoint (regular excludes the warm-up NaNs).
    assert not (masks["demanda_anomala"] & masks["demanda_regular"]).any()
    # Overlapping views: entre_semana strictly contains laborable_ordinario.
    assert (masks["laborable_ordinario"] & ~masks["entre_semana"]).sum() == 0
    assert masks["entre_semana"].sum() > masks["laborable_ordinario"].sum()


def test_anomaly_scale_is_estimated_on_train_only():
    src = inspect.getsource(sb.subgroup_masks) + inspect.getsource(sb._train_mad)
    assert "train_mask" in src
    assert "_train_mad" in inspect.getsource(sb.subgroup_masks)

    feats = sb._features()
    from src.utils.splits import chronological_split, split_masks

    bounds = chronological_split(feats["date"])
    split = split_masks(feats, bounds)
    delta7 = feats["total"] - feats["total"].shift(sb.ANOMALY_LAG)

    mad_train = sb._train_mad(delta7, split["train"].to_numpy())
    mad_all = sb._train_mad(delta7, np.ones(len(feats), dtype=bool))
    # The eval splits are more volatile: a full-series scale would differ materially, and
    # using it would change which days count as anomalous.
    assert mad_train != pytest.approx(mad_all, rel=1e-3)


# --------------------------------------------------------------------------------------
# Metrics reuse — cannot drift from chapter 5
# --------------------------------------------------------------------------------------


def test_metrics_come_from_all_metrics():
    src = inspect.getsource(sb)
    assert "from src.evaluation.metrics import all_metrics" in src
    assert "all_metrics(" in inspect.getsource(sb._safe_metrics)


def test_subgroup_metrics_reproduce_chapter5_split_totals(preds):
    # calendario_ordinario u calendario_irregular = the whole split, so aggregating both
    # must reproduce the published test MAE (CLAUDE.md / tabla_cadena_error).
    masks = sb.subgroup_masks().set_index("date")
    test_rows = preds[preds["split"] == "test"]
    union = masks["calendario_ordinario"] | masks["calendario_irregular"]
    for model, expected in (("xgboost_alone", 156_121), ("hybrid", 234_223)):
        pair = test_rows[test_rows["model"] == model].set_index("date")
        sel = union.reindex(pair.index).fillna(False)
        assert sel.all()  # every test day is in exactly one of the two
        mae = all_metrics(pair["y_true"], pair["y_pred"])["MAE"]
        assert mae == pytest.approx(expected, abs=50)

    metrics = sb.subgroup_metrics("test", preds)
    assert list(metrics.columns) == [
        "subgroup", "scope", "n", "model", "MAE", "RMSE", "MAPE", "R2", "rank_MAE",
        "inferential",
    ]


def test_small_subgroups_are_not_ranked():
    metrics = sb.subgroup_metrics("test")
    festivo = metrics[metrics["subgroup"] == "festivo"]  # n=5 < SMALL_SAMPLE_THRESHOLD
    assert festivo["rank_MAE"].isna().all()
    big = metrics[metrics["subgroup"] == "laborable_ordinario"]
    assert big["rank_MAE"].notna().all()
    assert set(big["rank_MAE"]) >= {1}


# --------------------------------------------------------------------------------------
# The deliverable win table and the standing empirical result
# --------------------------------------------------------------------------------------


def test_hybrid_win_table_is_the_deliverable_shape():
    win = sb.hybrid_win_table("test")
    assert list(win.columns) == [
        "subgroup", "n", "inferential", "best_model", "best_MAE", "best_hybrid",
        "best_hybrid_MAE", "delta_MAE", "delta_pct", "hybrid_ranks_first",
    ]
    # best_model is always a competitor, best_hybrid always a hybrid.
    assert set(win["best_model"]).issubset(set(sb.COMPETITOR_MODELS))
    assert set(win["best_hybrid"]).issubset(set(sb.HYBRID_MODELS))
    # delta_MAE sign matches the ranking flag.
    assert (win["hybrid_ranks_first"] == (win["delta_MAE"] < 0)).all()


def test_hybrid_wins_in_no_inferential_subgroup_on_the_verdict_scope(criteria):
    """The standing empirical result: mandated explicit win count, even when zero."""
    inferential = criteria[criteria["verdict"].isin(
        {sb.VERDICT_WIN, sb.VERDICT_RANK_ONLY, sb.VERDICT_NO_WIN}
    )]
    assert (inferential["verdict"] == sb.VERDICT_NO_WIN).all()
    assert (criteria["verdict"] == sb.VERDICT_WIN).sum() == 0
    assert (criteria["verdict"] == sb.VERDICT_RANK_ONLY).sum() == 0


def test_excluded_subgroups_get_no_verdict(criteria):
    for sg in sb.EXCLUDED_SUBGROUPS:
        row = criteria.set_index("subgroup").loc[sg]
        assert row["verdict"] == sb.VERDICT_EXCLUDED
    assert (criteria["verdict"] == sb.VERDICT_EXCLUDED).sum() == 2


# --------------------------------------------------------------------------------------
# demanda_anomala — verdict-scope resolution (supervisor's named subgroup)
# --------------------------------------------------------------------------------------


def test_demanda_anomala_is_non_inferential_on_the_verdict_scope(criteria):
    row = criteria.set_index("subgroup").loc["demanda_anomala"]
    assert row["n_test"] == 16 and row["n_test"] < sb.MIN_N_INFERENTIAL
    assert row["verdict"] == sb.VERDICT_NON_INFER
    # Its pooled val_test view reaches n=50 but that is a SECONDARY descriptive scope.
    pooled = sb.hybrid_win_table("val_test").set_index("subgroup")
    assert int(pooled.loc["demanda_anomala", "n"]) == 50


def test_no_code_path_assigns_a_per_subgroup_verdict_scope():
    src = inspect.getsource(sb.evaluate_subgroup_criteria)
    # No subgroup name appears as a literal in the verdict logic — the scope is the same
    # module constant for every subgroup, including demanda_anomala.
    for sg in sb.SUBGROUPS:
        assert f'"{sg}"' not in src and f"'{sg}'" not in src
    # The docstring records that the pooled scope was deliberately not promoted.
    assert "DELIBERATELY NOT" in sb.__doc__
    assert "demanda_anomala" in sb.__doc__


# --------------------------------------------------------------------------------------
# Bootstrap + BH
# --------------------------------------------------------------------------------------


def test_bootstrap_is_seeded_and_reproducible(preds):
    a = sb.bootstrap_delta("test", n_boot=250, seed=42, predictions=preds)
    b = sb.bootstrap_delta("test", n_boot=250, seed=42, predictions=preds)
    pd.testing.assert_frame_equal(a, b)
    c = sb.bootstrap_delta("test", n_boot=250, seed=7, predictions=preds)
    assert not np.allclose(a["p_raw"], c["p_raw"])


def test_bootstrap_reselects_best_inside_each_replicate():
    src = inspect.getsource(sb.bootstrap_delta)
    assert "comp_cols" in src and "hyb_cols" in src
    # argmin over both families happens inside the per-replicate loop.
    assert "block_ae[:, comp_cols].mean(axis=0).min()" in src
    assert "block_ae[:, hyb_cols].mean(axis=0).min()" in src


def test_bh_correction_uses_multipletests():
    src = inspect.getsource(sb.bootstrap_delta)
    assert "multipletests(" in src
    assert 'method="fdr_bh"' in src
    assert "alpha=FDR_Q" in src


def test_bootstrap_family_is_the_inferential_subgroups_only(preds):
    boot = sb.bootstrap_delta("test", predictions=preds)
    # No non-inferential or excluded subgroup enters the corrected family.
    assert set(boot["subgroup"]).isdisjoint(sb.EXCLUDED_SUBGROUPS)
    assert (boot["n"] >= sb.MIN_N_INFERENTIAL).all()
    assert boot["p_adj_bh"].between(0, 1).all()
    assert (boot["p_adj_bh"] + 1e-12 >= boot["p_raw"]).all()


def test_bootstrap_suppresses_ci_below_the_usable_fraction_threshold(preds):
    boot = sb.bootstrap_delta("test", predictions=preds).set_index("subgroup")
    low = boot[boot["frac_resamples_usable"] < sb.USABLE_RESAMPLE_MIN]
    assert low["ci_low"].isna().all() and low["ci_high"].isna().all()
    ok = boot[boot["frac_resamples_usable"] >= sb.USABLE_RESAMPLE_MIN]
    assert ok["ci_low"].notna().all()


# --------------------------------------------------------------------------------------
# Anti-over-reading: the concrete expected count
# --------------------------------------------------------------------------------------


def test_expected_first_by_chance_is_a_concrete_number():
    chance = sb.expected_first_by_chance("test")
    assert chance["family_size"] == 9
    assert chance["n_models"] == 11
    assert chance["n_hybrid_models"] == 2
    assert chance["p_null"] == pytest.approx(2 / 11)
    assert chance["expected_first"] == pytest.approx(9 * 2 / 11)


# --------------------------------------------------------------------------------------
# Rolling / period view
# --------------------------------------------------------------------------------------


def test_rolling_mae_spans_the_continuous_val_test_window(preds):
    roll = sb.rolling_mae(predictions=preds)
    # 393 contiguous days, 28-day window -> 366 windows per covered model.
    per_model = roll.groupby("model").size()
    assert (per_model == 393 - sb.ROLLING_WINDOW + 1).all()
    assert set(per_model.index) == set(MODEL_NAMES)
    assert (roll["n_window"] == sb.ROLLING_WINDOW).all()


def test_scopes_pool_val_and_test_contiguously(preds):
    pred, y_true, models = sb._eval_matrix("val_test", preds)
    assert len(pred) == 393
    assert pred.index.is_monotonic_increasing
    assert (pred.index.to_series().diff().dropna().dt.days == 1).all()


# --------------------------------------------------------------------------------------
# Aggregate
# --------------------------------------------------------------------------------------


def test_summarise_returns_every_table(preds):
    out = sb.summarise(preds)
    expected = {
        "metrics_test", "metrics_val", "metrics_val_test",
        "win_test", "win_val", "win_val_test",
        "bootstrap_test", "criteria", "rolling_mae", "month_breakdown",
    }
    assert set(out) == expected
    assert all(isinstance(v, pd.DataFrame) and not v.empty for v in out.values())
