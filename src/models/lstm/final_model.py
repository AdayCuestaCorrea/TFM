"""Final Stage 1 LSTM: trained on the full training set, predicting val and test.

EARLY-STOPPING SLICE IS NOT THE OFFICIAL VAL SPLIT. The stopping criterion uses a
chronological tail of the TRAINING period (the last `EARLY_STOPPING_TAIL` of training
sequences, ~15%). The official validation split from src/utils/splits.py is never touched
during training.

This distinction is easy to blur and consequential if blurred: Keras' `validation_data` is
conventionally "the validation set", so wiring the official val split in there would make
early stopping select the epoch that best fits val -- turning val into a tuning set and
destroying its status as an out-of-sample estimate for the Phase 3/4/5 comparison table.
The reported val metrics would then be optimistic by an unknown margin, and the LSTM would
appear to beat SARIMAX partly because SARIMAX never got that advantage.

PREDICTION PROTOCOL matches oof.py and Phase 3's SARIMAX: one-step-ahead with observed
history. The window for day t is actual demand through t-1, never the model's own earlier
predictions. Weights are fit on training data only, so no val/test observation influences
a parameter.
"""

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.metrics import all_metrics
from src.features.build_features import FEATURES_DAILY_FILE
from src.models.lstm.model import LSTMConfig, build_lstm, early_stopping_callback
from src.models.lstm.oof import EARLY_STOPPING_TAIL, TARGET_COL
from src.models.lstm.scaling import FeatureScaler, TargetScaler
from src.models.lstm.sequence_builder import build_exog_matrix, build_sequences
from src.utils.paths import MODELS_DIR, PROCESSED_DIR
from src.utils.seed import GLOBAL_SEED, set_global_seed
from src.utils.splits import apply_warmup_policy, chronological_split, split_masks

VAL_TEST_PREDICTIONS_FILE: Path = PROCESSED_DIR / "lstm_val_test_predictions.parquet"
FINAL_MODEL_FILE: Path = MODELS_DIR / "lstm_stage1_final.keras"
FINAL_SCALER_FILE: Path = MODELS_DIR / "lstm_stage1_scaler.joblib"


@dataclass
class FinalModelResult:
    """Training log and metrics for the final model."""

    config: LSTMConfig
    n_train_rows: int
    n_train_sequences: int
    n_early_stopping_sequences: int
    early_stopping_start: pd.Timestamp
    early_stopping_end: pd.Timestamp
    epochs_run: int
    best_epoch: int
    metrics: dict[str, dict[str, float]]
    loss_curve: list[float]
    val_loss_curve: list[float]


def train_final_model(
    features_path: Path | str = FEATURES_DAILY_FILE,
    config: LSTMConfig | None = None,
    seed: int = GLOBAL_SEED,
    verbose: bool = True,
    max_train_rows: int | None = None,
    exog_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, FinalModelResult, TargetScaler, object]:
    """Train on the post-warm-up training set and predict val + test.

    Args:
        exog_columns: If given, additionally feed each target day's own `feat_`-prefixed
            feature vector to a second model input -- the Phase 8 informational-parity
            DIAGNOSTIC control (`src/evaluation/lstm_parity_control.py`). `None` (the
            default) is byte-identical to the Phase 4 univariate path and is pinned by
            `test_exog_columns_none_reproduces_the_phase4_training_set`. Persists no model.

        max_train_rows: If given, train on only the LAST `max_train_rows` rows of the
            training block — a TRAILING window anchored at `train_end`, growing backwards.
            Used by the learning-curve diagnostic (`lstm_learning_curve.py`) to vary
            training size while holding the evaluation set fixed. `None` (the default) is
            byte-identical to the original behaviour: it trains on the whole 889-row
            post-warm-up block, and a test pins that.

            The trailing anchor is deliberately CONSERVATIVE for a sample-size experiment:
            the smallest fractions train on the rows closest in time to validation, giving
            small samples a recency advantage. This biases the experiment AGAINST finding
            a sample-size effect, so any effect that survives is more credible, and a null
            result is correspondingly weaker evidence than it would first appear.

    Returns:
        (predictions table, training log, fitted scaler, trained Keras model).
    """
    config = config or LSTMConfig()
    if exog_columns is not None and config.n_exog == 0:
        config = replace(config, n_exog=len(exog_columns))
    set_global_seed(seed)

    df = pd.read_parquet(features_path).sort_values("date").reset_index(drop=True)
    bounds = chronological_split(df["date"])
    trimmed, _ = apply_warmup_policy(df, bounds)
    trimmed = trimmed.reset_index(drop=True)
    masks = split_masks(trimmed, bounds)

    values = trimmed[TARGET_COL].to_numpy(dtype="float64")
    dates = trimmed["date"]
    n_train = int(masks["train"].sum())

    # Trailing window: 0 keeps the whole training block (the default, byte-identical path).
    train_start = 0 if max_train_rows is None else max(0, n_train - int(max_train_rows))

    # Scaler fit on the (trailing) training block ONLY (Leakage Rule 3).
    scaler = TargetScaler("minmax").fit(values[train_start:n_train])
    scaled = scaler.transform(values)

    train_targets = np.arange(train_start + config.window, n_train, dtype="int64")
    X_train, y_train, train_positions = build_sequences(scaled, config.window, train_targets)

    # Optional day-t exogenous vector (Phase 8 parity control). Scaled on train rows only.
    exog_all = None
    if exog_columns is not None:
        if not all(c.startswith("feat_") for c in exog_columns):
            raise ValueError(
                "train_final_model: exog_columns must all carry the 'feat_' prefix "
                "(the Phase 2/3 leakage-clean feature matrix); got a non-feat column."
            )
        feat_values = trimmed[exog_columns].to_numpy(dtype="float64")
        feat_scaler = FeatureScaler().fit(feat_values[train_start:n_train])
        feat_scaled = feat_scaler.transform(feat_values)
        exog_all = build_exog_matrix(feat_scaled, np.arange(len(feat_scaled)))

    n_es = max(1, int(len(X_train) * EARLY_STOPPING_TAIL))
    X_fit, y_fit = X_train[:-n_es], y_train[:-n_es]
    X_es, y_es = X_train[-n_es:], y_train[-n_es:]
    es_positions = train_positions[-n_es:]

    model = build_lstm(config)
    if exog_all is not None:
        E_train = exog_all[train_positions]
        fit_inputs = [X_fit, E_train[:-n_es]]
        es_inputs = [X_es, E_train[-n_es:]]
    else:
        fit_inputs, es_inputs = X_fit, X_es
    history = model.fit(
        fit_inputs,
        y_fit,
        validation_data=(es_inputs, y_es),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=[early_stopping_callback(config.patience)],
        shuffle=False,
        verbose=1 if verbose else 0,
    )

    # Predict every position that has a full window, across all three splits.
    all_targets = np.arange(config.window, len(values), dtype="int64")
    X_all, _, positions = build_sequences(scaled, config.window, all_targets)
    predict_inputs = X_all if exog_all is None else [X_all, exog_all[positions]]
    preds = scaler.inverse_transform(model.predict(predict_inputs, verbose=0).ravel())

    predictions = pd.DataFrame(
        {
            "date": dates.iloc[positions].to_numpy(),
            "y_true": values[positions],
            "y_pred_lstm": preds,
        }
    )
    pred_masks = split_masks(predictions, bounds)
    predictions["split"] = np.select(
        [pred_masks["train"], pred_masks["val"], pred_masks["test"]],
        ["train", "val", "test"],
        default="unassigned",
    )

    metrics = {
        split: all_metrics(
            predictions.loc[predictions["split"] == split, "y_true"],
            predictions.loc[predictions["split"] == split, "y_pred_lstm"],
        )
        for split in ("train", "val", "test")
    }

    val_losses = [float(v) for v in history.history["val_loss"]]
    result = FinalModelResult(
        config=config,
        n_train_rows=n_train - train_start,
        n_train_sequences=len(X_train),
        n_early_stopping_sequences=n_es,
        early_stopping_start=dates.iloc[es_positions[0]],
        early_stopping_end=dates.iloc[es_positions[-1]],
        epochs_run=len(val_losses),
        best_epoch=int(np.argmin(val_losses)) + 1,
        metrics=metrics,
        loss_curve=[float(v) for v in history.history["loss"]],
        val_loss_curve=val_losses,
    )
    return predictions, result, scaler, model


def main() -> None:
    predictions, result, scaler, model = train_final_model()

    print("\n" + "=" * 78)
    print("FINAL LSTM — trained on the full post-warm-up training set")
    print("=" * 78)
    print(f"config             : {result.config.describe()}")
    print(f"training rows      : {result.n_train_rows}")
    print(f"training sequences : {result.n_train_sequences}")
    print(
        f"early-stop slice   : {result.n_early_stopping_sequences} sequences "
        f"({result.early_stopping_start:%Y-%m-%d} -> "
        f"{result.early_stopping_end:%Y-%m-%d})"
    )
    print("                     ^ tail of TRAINING, NOT the official val split")
    print(f"epochs run         : {result.epochs_run} (best {result.best_epoch})")

    print("\n" + "-" * 78)
    print(f"{'split':<8} {'n':>5} {'MAE':>12} {'RMSE':>12} {'MAPE':>8} {'R2':>9}")
    print("-" * 78)
    for split in ("train", "val", "test"):
        m = result.metrics[split]
        note = "  (in-sample, reference only)" if split == "train" else ""
        print(
            f"{split:<8} {int(m['n']):>5} {m['MAE']:>12,.0f} {m['RMSE']:>12,.0f} "
            f"{m['MAPE']:>7.2f}% {m['R2']:>9.4f}{note}"
        )
    print("-" * 78)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    val_test = predictions[predictions["split"].isin(["val", "test"])].reset_index(drop=True)
    val_test.to_parquet(VAL_TEST_PREDICTIONS_FILE, index=False)
    model.save(FINAL_MODEL_FILE)
    scaler.save(FINAL_SCALER_FILE)

    print(f"\nval/test predictions : {VAL_TEST_PREDICTIONS_FILE}")
    print(f"model                : {FINAL_MODEL_FILE}")
    print(f"scaler               : {FINAL_SCALER_FILE}")


if __name__ == "__main__":
    main()
