"""Global random seed control.

Per the CLAUDE.md reproducibility convention the seed is defined ONCE, here, and applied
from every training entry point. Do not redefine it locally in a model module: a second
literal 42 somewhere else is how two runs silently stop being comparable.
"""

import os
import random

import numpy as np

GLOBAL_SEED = 42


def set_global_seed(seed: int = GLOBAL_SEED, deterministic_ops: bool = True) -> None:
    """Seed `random`, `numpy`, and TensorFlow, and optionally force deterministic kernels.

    Args:
        seed: The seed value. Defaults to GLOBAL_SEED.
        deterministic_ops: If True, request deterministic GPU/CPU kernels from TensorFlow.
            Costs some speed but makes repeated runs bit-identical, which is what the
            reproducibility convention actually requires -- seeding the RNGs alone does
            not, because non-deterministic reductions still vary run to run.
    """
    # Must be set before TensorFlow initialises its kernels to have any effect.
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic_ops:
        os.environ["TF_DETERMINISTIC_OPS"] = "1"

    random.seed(seed)
    np.random.seed(seed)

    # Imported lazily so that modules which only need numpy seeding do not pay the
    # multi-second TensorFlow import cost.
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    if deterministic_ops:
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            # Not available on every backend/build; seeding above still applies.
            pass
