"""Canonical project paths.

Centralised so that no module in src/ hard-codes an absolute path: every loader
resolves its default input relative to the repository root, which keeps the code
portable and lets tests point at fixtures by passing an explicit path.
"""

from pathlib import Path

# paths.py -> src/utils -> src -> <project root>
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR: Path = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"

CRTM_FILE: Path = RAW_DIR / "CRTM_Evolucion_demanda_diaria.xlsx"
WEATHER_FILE: Path = RAW_DIR / "open-meteo-40.39N3.68W666m.csv"
CALENDAR_FILE: Path = RAW_DIR / "300082-1-calendario_laboral-csv.csv"

UNIFIED_DAILY_FILE: Path = INTERIM_DIR / "unified_daily.parquet"

# Trained model artifacts (Keras models, fitted scalers). Kept outside data/ because they
# are code-adjacent build products, not data.
MODELS_DIR: Path = PROJECT_ROOT / "models"
