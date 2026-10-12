"""Almacenamiento: base de MLflow sincronizada con GCS, JSON de predicciones y
lógica champion/challenger del Model Registry."""
import json
import logging
import sqlite3
import tempfile
from pathlib import Path

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from . import config as cfg

log = logging.getLogger(__name__)


def _bucket():
    from google.cloud import storage

    return storage.Client().bucket(cfg.GCS_BUCKET)


class MlflowStore:
    """Copia local del SQLite de MLflow, con descarga y subida a GCS.

    La subida usa if_generation_match: si otro proceso modificó la base
    mientras entrenábamos, la subida falla en vez de pisar sus cambios.
    """

    def __init__(self, versioned: bool = False):
        # versioned=True (API): cada versión de la base se descarga a un fichero
        # propio, para no sustituir un SQLite con conexiones abiertas.
        self.versioned = versioned
        self.generation = None
        self.local_path = cfg.MLFLOW_DB_LOCAL

    def pull(self) -> int | None:
        if not cfg.GCS_BUCKET:
            return None
        blob = _bucket().get_blob(cfg.MLFLOW_DB_BLOB)
        if blob is None:
            log.info("No existe base de MLflow en GCS; se creará una nueva")
            self.generation = 0
            self.local_path.unlink(missing_ok=True)
        else:
            if self.versioned:
                self.local_path = cfg.LOCAL_DIR / f"mlflow-{blob.generation}.db"
            tmp = self.local_path.with_suffix(".download")
            blob.download_to_filename(str(tmp), if_generation_match=blob.generation)
            tmp.replace(self.local_path)
            self.generation = blob.generation
            log.info("Base de MLflow descargada (generation=%s)", blob.generation)
        return self.generation

    def remote_generation(self) -> int | None:
        if not cfg.GCS_BUCKET:
            return None
        blob = _bucket().get_blob(cfg.MLFLOW_DB_BLOB)
        return blob.generation if blob else 0

    def push(self):
        if not cfg.GCS_BUCKET:
            return
        # Copia consistente con la API de backup de SQLite antes de subir.
        with tempfile.TemporaryDirectory() as d:
            snap = Path(d) / "mlflow.db"
            src = sqlite3.connect(self.local_path)
            dst = sqlite3.connect(snap)
            with dst:
                src.backup(dst)
            src.close()
            dst.close()
            blob = _bucket().blob(cfg.MLFLOW_DB_BLOB)
            blob.upload_from_filename(str(snap), if_generation_match=self.generation)
            self.generation = blob.generation
        log.info("Base de MLflow subida a gs://%s/%s", cfg.GCS_BUCKET, cfg.MLFLOW_DB_BLOB)

    def configure(self):
        uri = f"sqlite:///{self.local_path.as_posix()}"
        mlflow.set_tracking_uri(uri)
        mlflow.set_registry_uri(uri)


def ensure_experiment(name: str):
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=f"{cfg.ARTIFACT_ROOT}/{name}")
    mlflow.set_experiment(name)


def write_json(relpath: str, obj) -> str:
    data = json.dumps(obj, ensure_ascii=False, default=str)
    if cfg.GCS_BUCKET:
        _bucket().blob(relpath).upload_from_string(data, content_type="application/json")
        return f"gs://{cfg.GCS_BUCKET}/{relpath}"
    path = cfg.LOCAL_DIR / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data, encoding="utf-8")
    return str(path)


def read_json(relpath: str):
    if cfg.GCS_BUCKET:
        blob = _bucket().get_blob(relpath)
        return json.loads(blob.download_as_text()) if blob else None
    path = cfg.LOCAL_DIR / relpath
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def register_with_champion_challenger(model_uri: str, name: str, run_id: str, metric: str,
                                      value: float, tolerance: float) -> dict:
    """Registra una versión nueva como 'challenger' y la promueve a 'champion'
    si su métrica de validación no es peor que la del campeón actual menos la tolerancia."""
    client = MlflowClient()
    mv = mlflow.register_model(model_uri, name)
    client.set_registered_model_alias(name, "challenger", mv.version)

    try:
        champ = client.get_model_version_by_alias(name, "champion")
        champ_value = client.get_run(champ.run_id).data.metrics.get(metric)
    except MlflowException:
        champ, champ_value = None, None

    if champ is None:
        promote, reason = True, "no había campeón"
    elif champ_value is None:
        promote, reason = True, f"el campeón v{champ.version} no tiene la métrica {metric} (criterio nuevo)"
    elif value >= champ_value - tolerance:
        promote = True
        reason = f"{metric}={value:.4f} vs campeón v{champ.version}={champ_value:.4f} (tolerancia {tolerance})"
    else:
        promote = False
        reason = f"{metric}={value:.4f} peor que campeón v{champ.version}={champ_value:.4f}"

    if promote:
        client.set_registered_model_alias(name, "champion", mv.version)
    client.set_model_version_tag(name, mv.version, "promoted", str(promote).lower())
    client.set_model_version_tag(name, mv.version, "decision", reason)
    log.info("%s v%s: %s -> %s", name, mv.version, reason, "PROMOVIDO" if promote else "se queda como retador")
    return {"name": name, "version": int(mv.version), "promoted": promote, "reason": reason}
