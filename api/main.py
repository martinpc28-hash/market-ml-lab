"""API de inferencia + web.

- /api/forecast           Inferencia ONLINE: descarga datos de mercado, calcula
                          features y ejecuta el modelo campeón del Model Registry.
- /api/forecast/scenario  Igual, pero cambiando variables a mano ("¿y si el VIX sube a 35?").
- /api/allocation         Inferencia BATCH: lee la cartera que calculó el job diario.
- /api/models             Versiones, alias y métricas del Model Registry.
"""
import json
import logging
import math
import threading
import time
from datetime import date, timedelta
from pathlib import Path

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field

from marketml import config as cfg
from marketml.data import DataError, download_prices
from marketml.features import MARKET_FEATURES, market_features
from marketml.forecaster import TICKERS
from marketml.store import MlflowStore, read_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("marketml.api")

STATIC = Path(__file__).parent / "static"
REGISTRY_CHECK_SECONDS = 300
MARKET_CACHE_SECONDS = 1800

app = FastAPI(
    title="Market ML Lab",
    description="Predicción de VIX/SPY y asignación de ETFs con modelos registrados en MLflow.",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class ModelCache:
    """Mantiene en memoria el modelo campeón y lo recarga si el job publicó una base de MLflow nueva."""

    def __init__(self):
        self.store = MlflowStore(versioned=True)
        self.lock = threading.Lock()
        self.generation = None
        self.checked_at = 0.0
        self.forecaster = None
        self.forecaster_info = None
        self.forecast_backtest = None
        self.artifacts = {}
        self.market = None
        self.market_at = 0.0

    def _reload(self):
        self.generation = self.store.pull()
        self.store.configure()
        client = MlflowClient()
        mv = client.get_model_version_by_alias(cfg.FORECASTER_NAME, "champion")
        self.forecaster = mlflow.pyfunc.load_model(f"models:/{cfg.FORECASTER_NAME}@champion")
        meta = self.forecaster.unwrap_python_model().metadata
        self.forecaster_info = {"name": cfg.FORECASTER_NAME, "version": int(mv.version), **meta}
        self.forecast_backtest = self.artifact_json(mv.run_id, "backtest/forecast_oof.json")
        log.info("Modelo %s v%s cargado", cfg.FORECASTER_NAME, mv.version)

    def artifact_json(self, run_id: str, path: str):
        key = (run_id, path)
        if key not in self.artifacts:
            local = MlflowClient().download_artifacts(run_id, path)
            self.artifacts[key] = json.loads(Path(local).read_text(encoding="utf-8"))
        return self.artifacts[key]

    def ensure_loaded(self):
        with self.lock:
            now = time.time()
            if self.forecaster is not None and now - self.checked_at < REGISTRY_CHECK_SECONDS:
                return
            self.checked_at = now
            if self.forecaster is None or self.store.remote_generation() != self.generation:
                try:
                    self._reload()
                except MlflowException as e:
                    raise HTTPException(503, f"Aún no hay modelo campeón registrado: {e.message}") from e

    def latest_features(self) -> pd.DataFrame:
        with self.lock:
            if self.market is None or time.time() - self.market_at > MARKET_CACHE_SECONDS:
                start = (date.today() - timedelta(days=500)).isoformat()
                try:
                    px = download_prices(TICKERS, start, allow_fred_fallback=True)
                except DataError as e:
                    raise HTTPException(503, f"No se pudieron descargar datos de mercado: {e}") from e
                self.market = (market_features(px).tail(1), px.tail(1), px.attrs.get("sources", {}))
                self.market_at = time.time()
            return self.market


cache = ModelCache()


def _forecast_payload(features: pd.DataFrame, prices: pd.DataFrame | None) -> dict:
    pred = cache.forecaster.predict(features).iloc[0]
    horizons = []
    for name, days in cfg.HORIZONS.items():
        horizons.append({
            "horizon": name,
            "days": days,
            "vix": {k: round(float(pred[f"vix_{name}{s}"]), 2) for k, s in (("pred", ""), ("p10", "_p10"), ("p90", "_p90"))},
            "spy_return_pct": {k: round(float(pred[f"spy_{name}{s}"]), 3) for k, s in (("pred", ""), ("p10", "_p10"), ("p90", "_p90"))},
        })
    info = cache.forecaster_info
    return {
        "as_of": features.index[-1].date().isoformat(),
        "current": {
            "vix": round(float(features["vix"].iloc[-1]), 2),
            "spy": round(float(prices["SPY"].iloc[-1]), 2) if prices is not None else None,
        },
        "forecast": horizons,
        "features": {k: round(float(v), 5) for k, v in features.iloc[-1].items()},
        "model": {
            "name": info["name"],
            "version": info["version"],
            "run_id": info["run_id"],
            "trained_until": info["train_end"],
            "val_skill_mean": info["val_skill_mean"],
        },
    }


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/forecast", summary="Predicción de VIX y SPY a 1 día, 1 semana y 1 mes")
def forecast():
    cache.ensure_loaded()
    feats, prices, sources = cache.latest_features()
    return {**_forecast_payload(feats, prices), "data_sources": sources}


class Scenario(BaseModel):
    overrides: dict[str, float] = Field(
        default_factory=dict,
        description="Variables a modificar sobre los datos de hoy, p. ej. {\"vix\": 35, \"term_structure\": 1.1}",
        examples=[{"vix": 35, "term_structure": 1.1}],
    )


@app.post("/api/forecast/scenario", summary="Predicción con variables modificadas (what-if)")
def scenario(body: Scenario):
    unknown = set(body.overrides) - set(MARKET_FEATURES)
    if unknown:
        raise HTTPException(422, f"Variables desconocidas: {sorted(unknown)}. Válidas: {MARKET_FEATURES}")
    if any(not math.isfinite(v) for v in body.overrides.values()):
        raise HTTPException(422, "Los valores deben ser números finitos")
    if "vix" in body.overrides and not 5 <= body.overrides["vix"] <= 150:
        raise HTTPException(422, "vix debe estar entre 5 y 150")
    cache.ensure_loaded()
    feats, prices, _ = cache.latest_features()
    feats = feats.copy()
    for k, v in body.overrides.items():
        feats[k] = v
    out = _forecast_payload(feats, prices)
    out["overrides"] = body.overrides
    return out


@app.get("/api/forecast/explain", summary="Cómo funciona el modelo: métricas, importancia de variables")
def forecast_explain():
    cache.ensure_loaded()
    info = cache.forecaster_info
    return {k: info[k] for k in ("name", "version", "run_id", "train_start", "train_end", "n_rows",
                                 "params", "metrics", "val_skill_mean", "importance", "feature_info")}


@app.get("/api/forecast/backtest", summary="Predicciones fuera de muestra frente a la realidad")
def forecast_backtest(target: str = "vix_1m", last: int = 250):
    cache.ensure_loaded()
    series = cache.forecast_backtest.get(target)
    if series is None:
        raise HTTPException(404, f"target válido: {list(cache.forecast_backtest)}")
    return {"target": target, "points": series[-max(1, min(last, 500)):]}


@app.get("/api/allocation", summary="Cartera de ETFs recomendada (inferencia batch diaria)")
def allocation():
    data = read_json(cfg.ALLOCATION_LATEST)
    if data is None:
        raise HTTPException(503, "El job de entrenamiento aún no ha publicado ninguna cartera")
    return data


@app.get("/api/allocation/backtest", summary="Backtest walk-forward del modelo de asignación campeón")
def allocation_backtest():
    cache.ensure_loaded()
    client = MlflowClient()
    try:
        mv = client.get_model_version_by_alias(cfg.ALLOCATOR_NAME, "champion")
    except MlflowException as e:
        raise HTTPException(503, "No hay modelo de asignación campeón") from e
    return cache.artifact_json(mv.run_id, "backtest/allocation_backtest.json")


@app.get("/api/models", summary="Model Registry: versiones, alias y decisión champion/challenger")
def models():
    cache.ensure_loaded()
    client = MlflowClient()
    out = {"mlflow_ui": cfg.MLFLOW_UI_URL or None, "models": []}
    key_metric = {cfg.FORECASTER_NAME: "val_skill_mean", cfg.ALLOCATOR_NAME: "strategy_sharpe"}
    for name, metric in key_metric.items():
        versions = sorted(client.search_model_versions(f"name='{name}'"), key=lambda v: -int(v.version))
        rm = client.get_registered_model(name)
        aliases = {v: a for a, v in rm.aliases.items()}
        out["models"].append({
            "name": name,
            "key_metric": metric,
            "versions": [{
                "version": int(v.version),
                "alias": aliases.get(v.version),
                "created": pd.Timestamp(v.creation_timestamp, unit="ms").isoformat(),
                "run_id": v.run_id,
                metric: client.get_run(v.run_id).data.metrics.get(metric),
                "promoted": v.tags.get("promoted"),
                "decision": v.tags.get("decision"),
                "strategy": client.get_run(v.run_id).data.tags.get("production_strategy"),
            } for v in versions[:20]],
        })
    return out
