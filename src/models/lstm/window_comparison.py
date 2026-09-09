"""Empirical justification for the W=28 default: compare W in [14, 28, 60].

Scope, stated plainly: ONE final-model run per W with early stopping, scored on the
official validation split. This is a documented comparison, not a hyperparameter search --
no repeated seeds, no joint tuning of units/dropout/learning rate, and no OOF regeneration
per W (which would triple an already long runtime for a decision this coarse).

Read the result accordingly. A single run per W carries real run-to-run variance, so treat
a small gap between two windows as a tie rather than evidence, and prefer the shorter
window on a tie: it consumes less history as warm-up and yields more training sequences.
Test-split numbers are deliberately NOT reported here -- choosing W by test performance
would consume the test set as a tuning signal.
"""

from pathlib import Path

import pandas as pd

from src.models.lstm.final_model import train_final_model
from src.models.lstm.model import LSTMConfig
from src.utils.paths import PROCESSED_DIR
from src.utils.seed import GLOBAL_SEED

WINDOWS: list[int] = [14, 28, 60]
WINDOW_COMPARISON_FILE: Path = PROCESSED_DIR / "lstm_window_comparison.parquet"


def compare_windows(
    windows: list[int] | None = None, seed: int = GLOBAL_SEED, verbose: bool = True
) -> pd.DataFrame:
    """Train one final model per window length and score it on the validation split."""
    windows = windows or WINDOWS
    rows = []

    for window in windows:
        if verbose:
            print(f"\n=== W = {window} ===", flush=True)
        config = LSTMConfig(window=window)
        _, result, _, _ = train_final_model(config=config, seed=seed, verbose=False)

        val = result.metrics["val"]
        rows.append(
            {
                "window": window,
                "train_sequences": result.n_train_sequences,
                "epochs_run": result.epochs_run,
                "best_epoch": result.best_epoch,
                "val_MAE": val["MAE"],
                "val_RMSE": val["RMSE"],
                "val_MAPE": val["MAPE"],
                "val_R2": val["R2"],
            }
        )
        if verbose:
            print(
                f"W={window}: val MAE {val['MAE']:,.0f}  RMSE {val['RMSE']:,.0f}  "
                f"MAPE {val['MAPE']:.2f}%  R2 {val['R2']:.4f}",
                flush=True,
            )

    return pd.DataFrame(rows)


def main() -> None:
    table = compare_windows()

    print("\n" + "=" * 78)
    print("WINDOW LENGTH COMPARISON — validation split, one run per W")
    print("=" * 78)
    display = table.copy()
    for col in ("val_MAE", "val_RMSE"):
        display[col] = display[col].map(lambda v: f"{v:,.0f}")
    display["val_MAPE"] = display["val_MAPE"].map(lambda v: f"{v:.2f}%")
    display["val_R2"] = display["val_R2"].map(lambda v: f"{v:.4f}")
    print(display.to_string(index=False))

    best = table.loc[table["val_MAE"].idxmin()]
    print(f"\nlowest val MAE: W={int(best['window'])}")
    print("Single run per W — treat small gaps as ties and prefer the shorter window.")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    table.to_parquet(WINDOW_COMPARISON_FILE, index=False)
    print(f"saved: {WINDOW_COMPARISON_FILE}")


if __name__ == "__main__":
    main()
