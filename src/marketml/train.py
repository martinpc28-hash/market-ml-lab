"""Punto de entrada del entrenamiento (Cloud Run Job o local).

    python -m marketml.train

1. Descarga la base de MLflow de GCS.
2. Descarga precios y entrena los dos modelos, registrando todo en MLflow.
3. Champion/challenger en el Model Registry.
4. Inferencia batch de la cartera con la estrategia campeona (usa las predicciones del modelo 1).
5. Sube la base de MLflow actualizada a GCS.
"""
import logging
import os
import sys

import mlflow

from . import allocator, config as cfg, forecaster
from .data import download_prices
from .features import market_features
from .store import MlflowStore


def champion_signals_today(px):
    """Predicciones del modelo 1 CAMPEÓN para la última fecha: variables para el modelo 2."""
    feats = market_features(px[forecaster.TICKERS]).tail(1)
    model = mlflow.pyfunc.load_model(f"models:/{cfg.FORECASTER_NAME}@champion")
    return forecaster.signal_features_from_prediction(model.predict(feats), feats["vix"])


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("marketml.train")
    trigger = os.getenv("TRIGGER", "manual")

    store = MlflowStore()
    store.pull()
    store.configure()

    tickers = sorted(set(forecaster.TICKERS) | set(allocator.TICKERS))
    px = download_prices(tickers, cfg.DATA_START)
    sources = px.attrs.get("sources", {})
    log.info("Precios: %d días, %s -> %s, fuentes %s", len(px), px.index[0].date(), px.index[-1].date(), sources)

    # El modelo 1 entrega sus predicciones fuera de muestra al modelo 2 (meta-labeling).
    fc_summary, vix_signals = forecaster.train(px, trigger=trigger, sources=sources)
    results = {
        "forecaster": fc_summary,
        "allocator": allocator.train(px, vix_signals, trigger=trigger, sources=sources),
    }
    allocation = allocator.batch_predict(px, champion_signals_today(px))
    store.push()

    log.info("Resultado: %s", results)
    log.info("Cartera publicada para %s", allocation["as_of"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
