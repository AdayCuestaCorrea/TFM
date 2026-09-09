"""Stage 1 LSTM architecture and training configuration.

ARCHITECTURE: a single LSTM layer (32 units by default) -> dropout -> Dense(1, linear).

Deliberately small. With ~860 usable training rows the binding constraint is sample size,
not capacity: a deeper or wider network would fit the training period more tightly without
generalising, and the residual hybrid's premise is that stage 1 captures the smooth
temporal structure while stage 2 handles what stage 1 cannot. An over-parameterised stage 1
that memorises holiday dips would leave stage 2 with residuals that look like noise on
training data and like structure at inference -- exactly the failure mode the OOF
requirement exists to prevent. 64 units is available via `units` and should only be adopted
if validation loss justifies it.

OUTPUT ACTIVATION IS LINEAR, NOT RELU. This is a regression head. ReLU would clip negative
values, and while demand itself is always positive, the SCALED target can legitimately sit
at or below zero: MinMax maps the training minimum to exactly 0.0, and any validation or
test day below the training minimum scales negative. A ReLU head could not represent those
days at all, and would silently floor them.
"""

from dataclasses import dataclass

DEFAULT_UNITS = 32
DEFAULT_DROPOUT = 0.2
DEFAULT_LEARNING_RATE = 1e-3
# 500, not 200. At 200 the cap was BINDING rather than early stopping: fold 3 of the first
# OOF run finished at epoch 200 with best_epoch=200, i.e. validation loss was still falling
# when training was cut off. A binding cap silently turns "train to convergence" into
# "train for a fixed budget", and the budget differs per fold because the folds differ in
# size -- so the folds were not comparable to each other either. With 500 the early
# stopping callback governs, which is what the patience setting is for.
DEFAULT_EPOCHS = 500
DEFAULT_BATCH_SIZE = 32
# Patience 10 follows the reference paper's convention; it is a reasonable default
# independent of architecture size, and with restore_best_weights it costs only time.
DEFAULT_PATIENCE = 10


@dataclass(frozen=True)
class LSTMConfig:
    """Everything that defines a training run, in one place for logging/reproducibility."""

    window: int = 28
    units: int = DEFAULT_UNITS
    dropout: float = DEFAULT_DROPOUT
    learning_rate: float = DEFAULT_LEARNING_RATE
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    patience: int = DEFAULT_PATIENCE
    # Number of day-t exogenous features concatenated after the LSTM branch. 0 (the
    # default) keeps the univariate Phase 4 architecture exactly. A positive value is used
    # only by the Phase 8 informational-parity DIAGNOSTIC control
    # (`src/evaluation/lstm_parity_control.py`), which persists no model artifact.
    n_exog: int = 0

    def describe(self) -> str:
        exog = f", n_exog={self.n_exog}" if self.n_exog else ""
        return (
            f"LSTM(units={self.units}, dropout={self.dropout}, window={self.window}{exog}) "
            f"Adam(lr={self.learning_rate}) MSE, batch={self.batch_size}, "
            f"max_epochs={self.epochs}, early_stopping(patience={self.patience})"
        )


def build_lstm(config: LSTMConfig):
    """Construct and compile the Stage 1 model.

    TensorFlow is imported inside the function so that modules importing this one for its
    config dataclass alone do not pay the multi-second import cost.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models

    if config.n_exog == 0:
        model = models.Sequential(
            [
                layers.Input(shape=(config.window, 1)),
                layers.LSTM(config.units),
                layers.Dropout(config.dropout),
                layers.Dense(1, activation="linear"),  # module docstring: never ReLU here
            ]
        )
    else:
        # Phase 8 informational-parity control: the target window feeds the LSTM branch,
        # and row t's own exogenous vector is concatenated afterwards -- the minimal change
        # that lets the LSTM see the calendar of the day being predicted (the window spans
        # [t-W, t) and excludes row t by construction). DIAGNOSTIC only; persists nothing.
        seq_in = layers.Input(shape=(config.window, 1), name="target_window")
        exog_in = layers.Input(shape=(config.n_exog,), name="exog")
        recurrent = layers.LSTM(config.units)(seq_in)
        merged = layers.Concatenate()([recurrent, exog_in])
        dropped = layers.Dropout(config.dropout)(merged)
        out = layers.Dense(1, activation="linear")(dropped)
        model = models.Model(inputs=[seq_in, exog_in], outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


def early_stopping_callback(patience: int = DEFAULT_PATIENCE):
    """Early stopping on validation loss, restoring the best weights.

    restore_best_weights=True matters: without it the returned model is the one from
    `patience` epochs past the optimum, which is measurably worse and varies run to run.
    """
    from tensorflow.keras.callbacks import EarlyStopping

    return EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        mode="min",
        verbose=0,
    )
