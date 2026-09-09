"""Out-of-fold LSTM predictions via expanding-window walk-forward validation.

THIS IS THE MODULE THE WHOLE ARCHITECTURE DEPENDS ON. Per the CLAUDE.md Critical Design
Decision, the LSTM predictions that Phase 5 trains XGBoost on MUST be out-of-fold. An
in-sample LSTM fit produces residuals that are artificially small and structurally unlike
the residuals seen at inference; XGBoost trained on those learns to correct a distribution
that does not exist on unseen data.

THE INVARIANT: a given date's OOF prediction comes from a model whose training data ended
strictly before that date. Enforced per fold by construction (the validation block starts
at the training slice's exclusive end) and asserted at runtime in `_assert_fold_is_clean`.

SCHEME: expanding window, 5 folds, entirely inside the post-warm-up training period
(2023-01-29 to 2025-07-05, 889 rows). Fold k trains on rows [0, s_k) and predicts rows
[s_k, s_{k+1}). Expanding rather than sliding, because discarding early history to keep a
fixed window would leave each fold with a few hundred daily observations -- too few for a
recurrent model -- and because expanding mirrors deployment, where you retrain on
everything you have.

WHAT IS EXCLUDED, AND WHY (mirrors the Phase 2 warm-up NaN policy -- state it, never
silently backfill): the first `MIN_INITIAL_TRAIN` rows can never receive an OOF prediction,
because predicting them would require a model trained on even less history than the
minimum. Those rows are written to the output file with `has_oof = False` and a stated
reason, not dropped, so Phase 5 can see exactly why its residual training set starts where
it does.

ONE-STEP-AHEAD PROTOCOL WITHIN A BLOCK. When predicting a fold's forward block, the input
windows are drawn from OBSERVED values, including observed values inside the block itself
for later targets in it. This is one-step-ahead forecasting with observed history -- the
identical protocol used for SARIMAX in Phase 3, which is what makes the two comparable. It
is not leakage: predicting day t from actual demand through t-1 is exactly what a deployed
daily forecaster does. What would be leakage, and is prevented, is letting any value from
day t or later influence the model's WEIGHTS.

PER-FOLD SCALING. Each fold fits its own scaler on its own training slice only, never on
the full training period. Reusing one globally fitted scaler would leak the later folds'
range into the earlier folds' inputs.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.metrics import mae, rmse
from src.features.build_features import FEATURES_DAILY_FILE
from src.models.lstm.model import LSTMConfig, build_lstm, early_stopping_callback
from src.models.lstm.scaling import TargetScaler
from src.models.lstm.sequence_builder import build_sequences
from src.utils.paths import PROCESSED_DIR
from src.utils.seed import GLOBAL_SEED, set_global_seed
from src.utils.splits import apply_warmup_policy, chronological_split, split_masks

TARGET_COL = "total"
N_FOLDS = 5
# Below this, a fold has too little history to fit a meaningful recurrent model: with
# W=28 a 200-row slice yields only 172 sequences, which is already thin.
MIN_INITIAL_TRAIN = 200
# Chronological tail of each fold's training slice reserved for the early-stopping signal.
# It is training data, never the official val split.
EARLY_STOPPING_TAIL = 0.15

OOF_PREDICTIONS_FILE: Path = PROCESSED_DIR / "lstm_oof_predictions.parquet"
NO_OOF_REASON = (
    f"insufficient history: within the first {MIN_INITIAL_TRAIN} training rows, "
    "which form the minimum initial training window and are never held out"
)


@dataclass
class FoldResult:
    """Everything the reproducibility convention requires logging for one fold."""

    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    n_train_rows: int
    n_train_sequences: int
    n_val_rows: int
    epochs_run: int
    best_epoch: int
    final_train_loss: float
    final_val_loss: float
    mae: float
    rmse: float
    loss_curve: list[float] = field(default_factory=list)
    val_loss_curve: list[float] = field(default_factory=list)

    def describe(self) -> str:
        return (
            f"fold {self.fold}: "
            f"train {self.train_start:%Y-%m-%d}->{self.train_end:%Y-%m-%d} "
            f"({self.n_train_rows} rows, {self.n_train_sequences} seqs) | "
            f"oof {self.val_start:%Y-%m-%d}->{self.val_end:%Y-%m-%d} "
            f"({self.n_val_rows} rows) | "
            f"epochs {self.epochs_run} (best {self.best_epoch}) | "
            f"loss {self.final_train_loss:.5f}/{self.final_val_loss:.5f} | "
            f"MAE {self.mae:,.0f} RMSE {self.rmse:,.0f}"
        )


def fold_boundaries(
    n_rows: int, n_folds: int = N_FOLDS, min_initial_train: int = MIN_INITIAL_TRAIN
) -> list[tuple[int, int, int]]:
    """Expanding-window fold boundaries as (train_start, train_end, val_end) positions.

    train slice is [0, train_end), OOF block is [train_end, val_end). Blocks tile
    [min_initial_train, n_rows) exactly, with the remainder given to the final fold so no
    training row is silently skipped.

    Raises:
        ValueError: if there is not enough history for the requested configuration.
    """
    if n_rows <= min_initial_train:
        raise ValueError(
            f"fold_boundaries: {n_rows} training rows is not more than the minimum "
            f"initial window ({min_initial_train}); no OOF block could be formed."
        )
    if n_folds < 1:
        raise ValueError("fold_boundaries: n_folds must be >= 1")

    remaining = n_rows - min_initial_train
    if remaining < n_folds:
        raise ValueError(
            f"fold_boundaries: {remaining} rows after the initial window cannot be split "
            f"into {n_folds} folds"
        )

    block = remaining // n_folds
    bounds: list[tuple[int, int, int]] = []
    for k in range(n_folds):
        train_end = min_initial_train + k * block
        val_end = n_rows if k == n_folds - 1 else train_end + block
        bounds.append((0, train_end, val_end))
    return bounds


def _assert_fold_is_clean(
    dates: pd.Series, train_end: int, val_end: int, fold: int
) -> None:
    """The OOF invariant: no training date may reach the fold's validation block."""
    train_dates = dates.iloc[:train_end]
    val_dates = dates.iloc[train_end:val_end]

    if train_dates.max() >= val_dates.min():
        raise ValueError(
            f"OOF VIOLATION in fold {fold}: training data ends "
            f"{train_dates.max():%Y-%m-%d} but the held-out block starts "
            f"{val_dates.min():%Y-%m-%d}. A date's OOF prediction must come from a model "
            "that never saw that date or any later date."
        )
    if len(set(train_dates) & set(val_dates)) > 0:
        raise ValueError(f"OOF VIOLATION in fold {fold}: train/val date overlap")


def run_fold(
    values: np.ndarray,
    dates: pd.Series,
    train_end: int,
    val_end: int,
    fold: int,
    config: LSTMConfig,
    seed: int = GLOBAL_SEED,
    verbose: int = 0,
) -> tuple[np.ndarray, np.ndarray, FoldResult]:
    """Train on [0, train_end) and predict [train_end, val_end).

    Returns:
        (target positions predicted, predictions in original units, fold log).
    """
    _assert_fold_is_clean(dates, train_end, val_end, fold)

    # Fresh weights and identical seed per fold: fold-to-fold differences must come from
    # the data, not from initialisation drift.
    set_global_seed(seed)

    # Scaler fit on THIS fold's training slice only (Leakage Rule 3).
    scaler = TargetScaler("minmax").fit(values[:train_end])
    scaled = scaler.transform(values)

    train_targets = np.arange(config.window, train_end, dtype="int64")
    X_train, y_train, _ = build_sequences(scaled, config.window, train_targets)

    # Chronological tail for early stopping -- still training data, never the official
    # val split. Keras' validation_split would take the tail too, but doing it explicitly
    # keeps the boundary visible and loggable.
    n_es = max(1, int(len(X_train) * EARLY_STOPPING_TAIL))
    X_fit, y_fit = X_train[:-n_es], y_train[:-n_es]
    X_es, y_es = X_train[-n_es:], y_train[-n_es:]

    model = build_lstm(config)
    history = model.fit(
        X_fit,
        y_fit,
        validation_data=(X_es, y_es),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=[early_stopping_callback(config.patience)],
        # shuffle=False. Minibatch shuffling of pre-built windows would NOT be leakage
        # (each sample is a self-contained window strictly preceding its target, and the
        # early-stopping tail is split off chronologically beforehand), so this was tested
        # rather than assumed: on fold 3, shuffling raised OOF MAE from 288k to 309k.
        # Keeping the chronological presentation order, which also matches how the folds
        # themselves advance.
        shuffle=False,
        verbose=verbose,
    )

    val_targets = np.arange(train_end, val_end, dtype="int64")
    X_val, _, positions = build_sequences(scaled, config.window, val_targets)
    scaled_pred = model.predict(X_val, verbose=0).ravel()
    predictions = scaler.inverse_transform(scaled_pred)

    y_true_block = values[positions]
    val_losses = [float(v) for v in history.history["val_loss"]]
    train_losses = [float(v) for v in history.history["loss"]]

    result = FoldResult(
        fold=fold,
        train_start=dates.iloc[0],
        train_end=dates.iloc[train_end - 1],
        val_start=dates.iloc[train_end],
        val_end=dates.iloc[val_end - 1],
        n_train_rows=train_end,
        n_train_sequences=len(X_train),
        n_val_rows=val_end - train_end,
        epochs_run=len(train_losses),
        best_epoch=int(np.argmin(val_losses)) + 1,
        final_train_loss=train_losses[-1],
        final_val_loss=val_losses[-1],
        mae=mae(y_true_block, predictions),
        rmse=rmse(y_true_block, predictions),
        loss_curve=train_losses,
        val_loss_curve=val_losses,
    )
    return positions, predictions, result


def generate_oof(
    features_path: Path | str = FEATURES_DAILY_FILE,
    config: LSTMConfig | None = None,
    n_folds: int = N_FOLDS,
    min_initial_train: int = MIN_INITIAL_TRAIN,
    seed: int = GLOBAL_SEED,
    verbose: bool = True,
) -> tuple[pd.DataFrame, list[FoldResult]]:
    """Produce the assembled OOF prediction table over the training period.

    Returns:
        (OOF table with columns [date, y_true, y_pred_oof, fold, has_oof, exclusion_reason],
         per-fold logs).
    """
    config = config or LSTMConfig()

    df = pd.read_parquet(features_path).sort_values("date").reset_index(drop=True)
    bounds = chronological_split(df["date"])
    trimmed, _ = apply_warmup_policy(df, bounds)
    train_df = trimmed[split_masks(trimmed, bounds)["train"]].reset_index(drop=True)

    values = train_df[TARGET_COL].to_numpy(dtype="float64")
    dates = train_df["date"]

    boundaries = fold_boundaries(len(values), n_folds, min_initial_train)

    if verbose:
        print("=" * 78)
        print("LSTM OUT-OF-FOLD GENERATION — expanding-window walk-forward")
        print("=" * 78)
        print(f"config      : {config.describe()}")
        print(f"seed        : {seed}")
        print(
            f"train period: {dates.iloc[0]:%Y-%m-%d} -> {dates.iloc[-1]:%Y-%m-%d} "
            f"({len(values)} rows, post warm-up)"
        )
        print(f"folds       : {n_folds}, expanding; min initial train {min_initial_train}")
        print()

    oof = np.full(len(values), np.nan, dtype="float64")
    fold_ids = np.full(len(values), -1, dtype="int64")
    results: list[FoldResult] = []

    for k, (_, train_end, val_end) in enumerate(boundaries, start=1):
        positions, predictions, result = run_fold(
            values, dates, train_end, val_end, k, config, seed
        )
        oof[positions] = predictions
        fold_ids[positions] = k
        results.append(result)
        if verbose:
            print(result.describe(), flush=True)

    has_oof = ~np.isnan(oof)
    table = pd.DataFrame(
        {
            "date": dates,
            "y_true": values,
            "y_pred_oof": oof,
            "fold": fold_ids,
            "has_oof": has_oof,
            "exclusion_reason": np.where(has_oof, "", NO_OOF_REASON),
        }
    )

    if verbose:
        n_excluded = int((~has_oof).sum())
        print()
        print(f"OOF coverage : {int(has_oof.sum())}/{len(values)} training rows")
        print(
            f"excluded     : {n_excluded} rows "
            f"({dates.iloc[0]:%Y-%m-%d} -> {dates.iloc[n_excluded - 1]:%Y-%m-%d})"
        )
        print(f"  reason     : {NO_OOF_REASON}")
        print(
            f"overall OOF  : MAE {mae(values[has_oof], oof[has_oof]):,.0f}  "
            f"RMSE {rmse(values[has_oof], oof[has_oof]):,.0f}"
        )

    return table, results


def main() -> None:
    table, results = generate_oof()
    OOF_PREDICTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(OOF_PREDICTIONS_FILE, index=False)
    print(f"\nOOF predictions saved: {OOF_PREDICTIONS_FILE}")

    curves = pd.DataFrame(
        [
            {
                "fold": r.fold,
                "epochs_run": r.epochs_run,
                "best_epoch": r.best_epoch,
                "final_train_loss": r.final_train_loss,
                "final_val_loss": r.final_val_loss,
                "mae": r.mae,
                "rmse": r.rmse,
            }
            for r in results
        ]
    )
    curves.to_parquet(PROCESSED_DIR / "lstm_oof_fold_log.parquet", index=False)
    print(f"fold log saved       : {PROCESSED_DIR / 'lstm_oof_fold_log.parquet'}")


if __name__ == "__main__":
    main()
