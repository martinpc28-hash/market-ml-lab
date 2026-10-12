#!/bin/sh
# Cada instancia nueva descarga la versión más reciente de la base de MLflow.
# Cloud Run escala a cero, así que tras un rato sin uso se vuelve a cargar.
# La cuenta de servicio solo tiene permiso de lectura sobre el bucket.
set -e
python - <<'PY'
import os
from google.cloud import storage
blob = storage.Client().bucket(os.environ["GCS_BUCKET"]).get_blob("mlflow/mlflow.db")
if blob:
    blob.download_to_filename("/tmp/mlflow.db")
    print("mlflow.db descargada, generation", blob.generation)
else:
    print("Aún no existe mlflow.db; se arranca vacía")
PY
exec mlflow server \
  --backend-store-uri sqlite:////tmp/mlflow.db \
  --default-artifact-root "gs://${GCS_BUCKET}/mlflow/artifacts" \
  --no-serve-artifacts \
  --host 0.0.0.0 --port "${PORT:-8080}" \
  --allowed-hosts "*" \
  --workers 1
