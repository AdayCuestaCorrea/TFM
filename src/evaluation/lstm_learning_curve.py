"""Learning curve of the Stage 1 LSTM — does training size explain its weakness?

DIAGNOSTIC CONTROL, NOT A CANDIDATE MODEL. Per the Phase 7 hard constraint, nothing this
module produces enters the master comparison table of chapter 5. Its fits are in-memory
and discarded; only the metrics parquet is persisted, and no artifact is written under
`models/`. It is the ONE place in this phase that retrains, and it does so because the
per-fold evidence (`stage1_residual_anatomy.fold_size_vs_error`) cannot answer the
question: 5 points, non-monotonic, and training size confounded with which period each
fold evaluates. A learning curve removes the confound by construction — the evaluation set
is held fixed (val) and only training size varies.

DESIGN.
  - Trailing training windows anchored at `train_end`, at fractions of the 889-row
    post-warm-up training block. Trailing rather than leading so no time gap opens between
    training and the val split. The trailing anchor is also deliberately CONSERVATIVE:
    the smallest fractions train on the rows closest in time to val, giving small samples
    a recency advantage, which biases the experiment AGAINST finding a sample-size effect.
    A null result is therefore weaker evidence than it first appears, and an effect that
    survives is more credible. (`train_final_model` carries the same note.)
  - 3 seeds per fraction, reported as median with a min-max band: single runs are too
    noisy to read (`window_comparison.py` says the same in its own docstring). GLOBAL_SEED
    is NOT redefined — seeds are passed to the existing `train_final_model(seed=...)`
    parameter and 42 stays primary. The fraction-1.00 point additionally gets
    `EXTRA_FRACTION1_SEEDS` (2 more), because Phase 8's `lstm_parity_control.py` reads it
    as a 5-seed median baseline; `assemble_full_curve` / `main()` build that in so one
    command reproduces the whole artifact.
  - Evaluated on VAL ONLY. The test split is never read, consistent with
    `window_comparison.py`.
  - Per fit, degeneracy is recorded, not just val MAE: a collapsed fit (near-constant
    prediction) and a merely-undertrained one produce different MAEs for entirely
    different reasons, and the curve must distinguish "a small sample trains a worse
    model" from "a small sample fails to train a model at all". Both are sample-size
    evidence, but they are different claims.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.models.lstm.final_model import train_final_model
from src.models.lstm.model import LSTMConfig
from src.utils.paths import PROCESSED_DIR
from src.utils.seed import GLOBAL_SEED

LEARNING_CURVE_FILE: Path = PROCESSED_DIR / "lstm_learning_curve.parquet"

# Fractions of the 889-row post-warm-up training block (CLAUDE.md Phase 3): ~222 -> 889.
FRACTIONS: list[float] = [0.25, 0.40, 0.55, 0.70, 0.85, 1.00]

# 42 stays primary (GLOBAL_SEED); 1 and 2 are the replicate seeds for the min-max band.
SEEDS: list[int] = [GLOBAL_SEED, 1, 2]

# Extra seeds computed at fraction 1.00 ONLY. `lstm_parity_control.py` (Phase 8) compares
# its 5-seed median against the fraction-1.00 point of this curve, so that point carries
# five seeds while every other fraction keeps three. `assemble_full_curve` / `main()` build
# this in, so the documented regen command reproduces the full artifact in one pass.
EXTRA_FRACTION1_SEEDS: list[int] = [3, 4]

# Degeneracy thresholds, anchored on CLAUDE.md Phase 4 rather than invented here:
#   fold 1 (degenerate)  -> OOF pred std ratio 0.199, pred/actual correlation 0.300
#   folds 2-5 (healthy)  -> std ratio 0.75-0.94,      correlation 0.79-0.95
# The cut sits in the wide empty gap between those two regimes. A fit is flagged degenerate
# only when BOTH statistics fall in the degenerate range, matching how fold 1 — low
# variance AND low correlation — was characterised.
DEGENERATE_STD_RATIO_MAX = 0.50
DEGENERATE_CORR_MAX = 0.50


def _val_degeneracy(predictions: pd.DataFrame) -> tuple[float, float, bool]:
    """(pred std / actual std, Pearson corr, degenerate flag) on the val split."""
    val = predictions[predictions["split"] == "val"]
    y_true = val["y_true"].to_numpy(dtype="float64")
    y_pred = val["y_pred_lstm"].to_numpy(dtype="float64")

    std_ratio = float(y_pred.std(ddof=1) / y_true.std(ddof=1))
    corr = float(stats.pearsonr(y_true, y_pred)[0])
    degenerate = std_ratio < DEGENERATE_STD_RATIO_MAX and corr < DEGENERATE_CORR_MAX
    return std_ratio, corr, degenerate


def run_learning_curve(
    fractions: list[float] | None = None,
    seeds: list[int] | None = None,
    verbose: bool = True,
    config: LSTMConfig | None = None,
) -> pd.DataFrame:
    """Fit one Stage 1 LSTM per (fraction, seed) and score it on the val split.

    Args:
        config: LSTM configuration. Defaults to the Phase 4 architecture; a tiny config is
            passed only by the smoke test, which asserts wiring, not accuracy.

    Returns:
        DataFrame [fraction, train_rows, train_sequences, seed, epochs_run, best_epoch,
        val_MAE, val_RMSE, val_MAPE, val_R2, pred_std_ratio, pred_actual_corr, degenerate],
        one row per fit.
    """
    fractions = fractions or FRACTIONS
    seeds = seeds or SEEDS

    # Resolve the fraction -> row count once against the real split, not a hard-coded 889.
    _, ref, _, _ = train_final_model(seed=seeds[0], verbose=False, config=config)
    n_train_full = ref.n_train_rows

    rows = []
    for fraction in fractions:
        max_rows = None if fraction >= 1.0 else int(round(fraction * n_train_full))
        for seed in seeds:
            if verbose:
                print(f"  fraction {fraction:.2f}  seed {seed} ...", flush=True)
            predictions, result, _, _ = train_final_model(
                seed=seed, verbose=False, max_train_rows=max_rows, config=config
            )
            std_ratio, corr, degenerate = _val_degeneracy(predictions)
            val = result.metrics["val"]
            rows.append(
                {
                    "fraction": fraction,
                    "train_rows": result.n_train_rows,
                    "train_sequences": result.n_train_sequences,
                    "seed": seed,
                    "epochs_run": result.epochs_run,
                    "best_epoch": result.best_epoch,
                    "val_MAE": val["MAE"],
                    "val_RMSE": val["RMSE"],
                    "val_MAPE": val["MAPE"],
                    "val_R2": val["R2"],
                    "pred_std_ratio": std_ratio,
                    "pred_actual_corr": corr,
                    "degenerate": degenerate,
                }
            )
            if verbose:
                flag = "  [DEGENERATE]" if degenerate else ""
                print(
                    f"    rows {result.n_train_rows}  val MAE {val['MAE']:,.0f}  "
                    f"std ratio {std_ratio:.3f}  corr {corr:.3f}{flag}",
                    flush=True,
                )
    return pd.DataFrame(rows)


def summarise_by_fraction(curve: pd.DataFrame) -> pd.DataFrame:
    """Median and min-max val MAE per fraction, plus the degenerate-fit count."""
    grouped = curve.groupby("fraction")
    return pd.DataFrame(
        {
            "fraction": list(grouped.groups),
            "train_rows": grouped["train_rows"].first().to_numpy(),
            "val_MAE_median": grouped["val_MAE"].median().to_numpy(),
            "val_MAE_min": grouped["val_MAE"].min().to_numpy(),
            "val_MAE_max": grouped["val_MAE"].max().to_numpy(),
            "n_degenerate": grouped["degenerate"].sum().to_numpy(),
            "n_seeds": grouped["seed"].size().to_numpy(),
        }
    )


def assemble_full_curve(
    fractions: list[float] | None = None,
    verbose: bool = True,
    config: LSTMConfig | None = None,
) -> pd.DataFrame:
    """The 3-seed curve over all fractions, plus `EXTRA_FRACTION1_SEEDS` at fraction 1.00.

    One deterministic pass: running this (via `main()`) from a clean checkout reproduces the
    full persisted artifact, including the five-seed fraction-1.00 point that
    `lstm_parity_control.UNIVARIATE_VAL_MAE` depends on. De-duplicated on (fraction, seed)
    so a re-run is idempotent and cannot double-count a seed.
    """
    main_curve = run_learning_curve(fractions=fractions, verbose=verbose, config=config)
    extra = run_learning_curve(
        fractions=[1.0], seeds=EXTRA_FRACTION1_SEEDS, verbose=verbose, config=config
    )
    combined = pd.concat([main_curve, extra], ignore_index=True)
    return combined.drop_duplicates(subset=["fraction", "seed"], keep="first").reset_index(
        drop=True
    )


def main() -> None:
    print("=" * 88)
    print("LSTM LEARNING CURVE — diagnostic control, val-only, no model artifact persisted")
    print("=" * 88)
    curve = assemble_full_curve()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    curve.to_parquet(LEARNING_CURVE_FILE, index=False)
    print(f"\nsaved: {LEARNING_CURVE_FILE}")

    print("\n--- per fit ---")
    print(curve.to_string(index=False))
    print("\n--- median (min-max) val MAE per fraction ---")
    print(summarise_by_fraction(curve).to_string(index=False))
    print(
        "\nReference val MAE: SARIMAX 273,467 | xgboost_alone 244,843 "
        "(CLAUDE.md Phases 3, 5)."
    )


if __name__ == "__main__":
    main()
