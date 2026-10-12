"""Configuración central. Todo se controla por variables de entorno.

- Sin GCS_BUCKET: modo local (MLflow en SQLite y artefactos en ./.marketml).
- Con GCS_BUCKET: la base de MLflow y los artefactos viven en Cloud Storage.
"""
import os
from pathlib import Path

GCS_BUCKET = os.getenv("GCS_BUCKET", "").strip()
LOCAL_DIR = Path(os.getenv("MARKETML_HOME", Path.cwd() / ".marketml")).resolve()
LOCAL_DIR.mkdir(parents=True, exist_ok=True)

# Base de datos de MLflow (runs, métricas, Model Registry). En la nube se
# guarda como un fichero SQLite en GCS que el job de entrenamiento descarga,
# actualiza y vuelve a subir (un único escritor).
MLFLOW_DB_LOCAL = LOCAL_DIR / "mlflow.db"
MLFLOW_DB_BLOB = "mlflow/mlflow.db"
ARTIFACT_ROOT = (
    f"gs://{GCS_BUCKET}/mlflow/artifacts" if GCS_BUCKET else (LOCAL_DIR / "artifacts").as_uri()
)

# Salidas de inferencia batch (las lee la API).
ALLOCATION_LATEST = "predictions/allocation/latest.json"

DATA_START = os.getenv("DATA_START", "2006-01-01")

FORECASTER_NAME = "vix-spy-forecaster"
ALLOCATOR_NAME = "etf-allocator"
FORECASTER_EXPERIMENT = "vix-spy-forecast"
ALLOCATOR_EXPERIMENT = "etf-allocation"

# Horizontes de predicción en días hábiles.
HORIZONS = {"1d": 1, "1w": 5, "1m": 21}

# Champion/challenger: el retador sustituye al campeón si su métrica de
# validación no empeora más que esta tolerancia.
FORECASTER_TOLERANCE = 0.01  # en "skill" (fracción de mejora sobre el baseline)
ALLOCATOR_TOLERANCE = 0.05  # en Sharpe de la estrategia que pasó el filtro

MLFLOW_UI_URL = os.getenv("MLFLOW_UI_URL", "")
