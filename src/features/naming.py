"""Shared naming conventions for engineered features.

Every column produced by a module in src/features/ carries the FEATURE_PREFIX. This is
not cosmetic: it lets downstream code select the model-ready matrix mechanically
(`[c for c in df.columns if c.startswith(FEATURE_PREFIX)]`) instead of maintaining a
hand-written column list that silently drifts out of date, and it makes an unprefixed
column -- i.e. a raw same-day value -- impossible to feed a model by accident.
"""

import re

FEATURE_PREFIX = "feat_"


def slugify(value: str) -> str:
    """Normalise a category label into a safe snake_case column-name fragment.

    Accented Spanish category labels ('Festivo de la Comunidad de Madrid') become stable
    ASCII-ish identifiers so that one-hot column names never depend on source casing or
    punctuation.
    """
    text = value.strip().lower()
    text = re.sub(r"[^0-9a-záéíóúüñ]+", "_", text)
    text = (
        text.replace("á", "a")
        .replace("é", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ú", "u")
        .replace("ü", "u")
        .replace("ñ", "n")
    )
    return re.sub(r"_+", "_", text).strip("_")
