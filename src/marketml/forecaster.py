"""Modelo 1: predicción del VIX y de la rentabilidad del SPY a 1 día, 1 semana y 1 mes.

Cómo se usa el ML aquí
----------------------
Para cada día t se calculan ~18 variables de mercado (features.py) y 6 objetivos:

    vix_h = log(VIX[t+h] / VIX[t])   ->  cambio del VIX en h días
    spy_h = log(SPY[t+h] / SPY[t])   ->  rentabilidad del SPY en h días
    con h en {1, 5, 21} días hábiles

Por cada objetivo se entrenan 3 modelos HistGradientBoosting (árboles potenciados):
  - uno con pérdida absoluta (predicción central, la mediana);
  - dos de regresión cuantílica (percentiles 10 y 90) que forman un intervalo del 80 %.

La validación es walk-forward: siempre se entrena con el pasado y se evalúa con
el futuro, dejando un hueco de h días para que los objetivos no se solapen.
Cada modelo se compara con un baseline ingenuo: para el VIX, "mañana igual que
hoy"; para el SPY, la rentabilidad mediana histórica. El "skill" mide cuánto
mejora el modelo al baseline (0 = igual, 0.10 = 10 % menos error).
"""
import logging
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.model_selection import TimeSeriesSplit  # noqa: E402

from . import config as cfg  # noqa: E402
from .features import MARKET_FEATURE_INFO, MARKET_FEATURES, market_features  # noqa: E402
from .store import ensure_experiment, register_with_champion_challenger  # noqa: E402

log = logging.getLogger(__name__)

TICKERS = ["SPY", "^VIX", "^VIX3M"]
TARGETS = [f"{asset}_{h}" for asset in ("vix", "spy") for h in cfg.HORIZONS]
QUANTILE_HEADS = {"pred": ("absolute_error", None), "p10": ("quantile", 0.1), "p90": ("quantile", 0.9)}

# Muy regularizado a propósito: con datos financieros ruidosos, árboles poco
# profundos y hojas grandes generalizan mejor (probado con walk-forward; ver README).
# Señales que el modelo 1 entrega al modelo 2 (cartera), en unidades logarítmicas.
SIGNAL_FEATURE_INFO = {
    "vix_fc_1m": "Cambio previsto del VIX a 1 mes según el modelo 1",
    "vix_fc_1m_p90": "Escenario alto (p90) del VIX a 1 mes según el modelo 1",
    "spy_fc_1m_p10": "Escenario malo (p10) del SPY a 1 mes según el modelo 1",
}
SIGNAL_FEATURES = list(SIGNAL_FEATURE_INFO)

DEFAULT_PARAMS = {
    "max_iter": 100,
    "learning_rate": 0.02,
    "max_depth": 2,
    "min_samples_leaf": 500,
    "l2_regularization": 10.0,
}


def build_dataset(px: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = market_features(px)
    Y = pd.DataFrame(index=px.index)
    for name, h in cfg.HORIZONS.items():
        Y[f"vix_{name}"] = np.log(px["^VIX"].shift(-h) / px["^VIX"])
        Y[f"spy_{name}"] = np.log(px["SPY"].shift(-h) / px["SPY"])
    return X, Y.reindex(X.index)


def _model(head: str, params: dict) -> HistGradientBoostingRegressor:
    loss, q = QUANTILE_HEADS[head]
    kw = {"quantile": q} if q is not None else {}
    return HistGradientBoostingRegressor(loss=loss, random_state=42, **params, **kw)


def _to_human(target: str, z: np.ndarray, vix_now: np.ndarray) -> np.ndarray:
    """Convierte la salida del modelo (cambio logarítmico) a unidades legibles."""
    if target.startswith("vix"):
        return vix_now * np.exp(z)  # nivel del VIX
    return (np.exp(z) - 1) * 100  # rentabilidad en %


def signal_features_from_oof(oofs: dict) -> pd.DataFrame:
    """Predicciones fuera de muestra del walk-forward -> variables para el modelo de cartera."""
    return pd.DataFrame({
        "vix_fc_1m": oofs["vix_1m"]["pred"],
        "vix_fc_1m_p90": oofs["vix_1m"]["p90"],
        "spy_fc_1m_p10": oofs["spy_1m"]["p10"],
    })


def signal_features_from_prediction(pred: pd.DataFrame, vix_now: pd.Series) -> pd.DataFrame:
    """Salida del modelo campeón (unidades legibles) -> las mismas variables en log."""
    return pd.DataFrame({
        "vix_fc_1m": np.log(pred["vix_1m"] / vix_now),
        "vix_fc_1m_p90": np.log(pred["vix_1m_p90"] / vix_now),
        "spy_fc_1m_p10": np.log1p(pred["spy_1m_p10"] / 100),
    }, index=pred.index)


class MarketForecaster(mlflow.pyfunc.PythonModel):
    """Modelo empaquetado para MLflow: entra una tabla de features y salen
    las 6 predicciones con su intervalo del 80 %, en unidades legibles."""

    def __init__(self, models: dict, metadata: dict):
        self.models = models  # {"vix_1d": {"pred": m, "p10": m, "p90": m}, ...}
        self.metadata = metadata

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        X = model_input[MARKET_FEATURES].astype(float)
        vix_now = X["vix"].to_numpy()
        out = {}
        for target, heads in self.models.items():
            raw = {h: m.predict(X) for h, m in heads.items()}
            lo = np.minimum.reduce([raw["p10"], raw["p90"], raw["pred"]])
            hi = np.maximum.reduce([raw["p10"], raw["p90"], raw["pred"]])
            out[target] = _to_human(target, raw["pred"], vix_now)
            out[f"{target}_p10"] = _to_human(target, lo, vix_now)
            out[f"{target}_p90"] = _to_human(target, hi, vix_now)
        return pd.DataFrame(out, index=model_input.index)


def _metrics(y, pred, p10, p90, base) -> dict:
    err, err_b = np.abs(y - pred), np.abs(y - base)
    return {
        "mae": float(err.mean()),
        "mae_baseline": float(err_b.mean()),
        "skill": float(1 - err.mean() / err_b.mean()),
        "hit_rate": float((np.sign(pred) == np.sign(y)).mean()),
        # Para el SPY el baseline siempre dice "sube" (mediana > 0): acierta tantas
        # veces como sube el mercado. Para el VIX el baseline (sin cambio) no tiene dirección.
        "hit_rate_baseline": float((np.sign(base) == np.sign(y)).mean()),
        "coverage_80": float(((y >= p10) & (y <= p90)).mean()),
    }


def walk_forward(X: pd.DataFrame, y: pd.Series, h: int, target: str, params: dict, n_splits: int):
    ok = y.notna()
    Xv, yv = X[ok], y[ok]
    oof = pd.DataFrame(index=yv.index, columns=["pred", "p10", "p90", "base"], dtype=float)
    last = None
    for tr, te in TimeSeriesSplit(n_splits=n_splits, gap=h).split(Xv):
        for head in QUANTILE_HEADS:
            m = _model(head, params).fit(Xv.iloc[tr], yv.iloc[tr])
            oof.iloc[te, oof.columns.get_loc(head)] = m.predict(Xv.iloc[te])
            if head == "pred":
                last = (m, Xv.iloc[te], yv.iloc[te])
        oof.iloc[te, oof.columns.get_loc("base")] = 0.0 if target.startswith("vix") else yv.iloc[tr].median()
    oof = oof.dropna()
    oof["y"] = yv.loc[oof.index]
    return oof, last


def _importance(fitted) -> dict:
    m, Xte, yte = fitted
    pi = permutation_importance(m, Xte, yte, scoring="neg_mean_absolute_error", n_repeats=5, random_state=0)
    imp = pd.Series(pi.importances_mean, index=Xte.columns).clip(lower=0)
    imp = imp / imp.sum() if imp.sum() > 0 else imp
    return {k: round(float(v), 4) for k, v in imp.sort_values(ascending=False).items()}


def _backtest_series(oof: pd.DataFrame, X: pd.DataFrame, target: str, h: int, px_index, n_last=500) -> list:
    """Predicción vs realidad en unidades legibles, para la web."""
    o = oof.tail(n_last)
    vix_now = X.loc[o.index, "vix"].to_numpy()
    target_dates = [px_index[px_index.get_loc(d) + h] for d in o.index]
    conv = lambda z: _to_human(target, z.to_numpy(), vix_now)  # noqa: E731
    rows = zip(o.index, target_dates, conv(o["y"]), conv(o["pred"]), conv(o["p10"]), conv(o["p90"]))
    return [
        {"origin": d.date().isoformat(), "date": td.date().isoformat(), "actual": round(a, 3),
         "pred": round(p, 3), "p10": round(lo, 3), "p90": round(hi, 3)}
        for d, td, a, p, lo, hi in rows
    ]


def _plot(series: dict, path: Path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=False)
    for ax, target in zip(axes.ravel(), TARGETS):
        s = pd.DataFrame(series[target]).tail(250)
        d = pd.to_datetime(s["date"])
        ax.fill_between(d, s["p10"], s["p90"], alpha=0.25, label="intervalo 80 %")
        ax.plot(d, s["actual"], lw=1, color="black", label="real")
        ax.plot(d, s["pred"], lw=1, label="predicción")
        ax.set_title(target)
        ax.tick_params(axis="x", labelrotation=30, labelsize=7)
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def train(px: pd.DataFrame, params: dict | None = None, n_splits: int = 5, trigger: str = "manual",
          sources: dict | None = None) -> dict:
    params = {**DEFAULT_PARAMS, **(params or {})}
    X, Y = build_dataset(px[TICKERS])
    ensure_experiment(cfg.FORECASTER_EXPERIMENT)

    with mlflow.start_run(run_name=f"forecaster-{X.index[-1].date()}") as run:
        mlflow.set_tags({f"data_source.{t.lstrip('^')}": s for t, s in (sources or {}).items()})
        mlflow.set_tags({"trigger": trigger, "model_family": "HistGradientBoosting + quantile"})
        mlflow.log_params({**params, "n_splits": n_splits, "n_features": len(MARKET_FEATURES),
                           "train_start": X.index[0].date(), "train_end": X.index[-1].date(),
                           "n_rows": len(X)})

        metrics, importance, series, models, oofs = {}, {}, {}, {}, {}
        for target in TARGETS:
            h = cfg.HORIZONS[target.split("_")[1]]
            oof, last = walk_forward(X, Y[target], h, target, params, n_splits)
            oofs[target] = oof
            m = _metrics(oof["y"], oof["pred"], oof["p10"], oof["p90"], oof["base"])
            metrics[target] = m
            mlflow.log_metrics({f"{target}_{k}": v for k, v in m.items()})
            importance[target] = _importance(last)
            series[target] = _backtest_series(oof, X, target, h, px.index)

            ok = Y[target].notna()
            models[target] = {head: _model(head, params).fit(X[ok], Y[target][ok]) for head in QUANTILE_HEADS}
            log.info("%s skill=%.3f hit=%.3f cov=%.3f", target, m["skill"], m["hit_rate"], m["coverage_80"])

        skill_mean = float(np.mean([m["skill"] for m in metrics.values()]))
        mlflow.log_metric("val_skill_mean", skill_mean)

        metadata = {
            "run_id": run.info.run_id,
            "train_start": X.index[0].date().isoformat(),
            "train_end": X.index[-1].date().isoformat(),
            "n_rows": len(X),
            "params": params,
            "metrics": metrics,
            "val_skill_mean": skill_mean,
            "importance": importance,
            "feature_info": MARKET_FEATURE_INFO,
        }
        mlflow.log_dict(metadata, "metadata.json")
        mlflow.log_dict(series, "backtest/forecast_oof.json")
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "pred_vs_real.png"
            _plot(series, p)
            mlflow.log_artifact(str(p), "plots")

        model = MarketForecaster(models, metadata)
        example = X.tail(5)
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=model,
            input_example=example,
            signature=infer_signature(example, model.predict(None, example)),
            code_paths=[str(Path(__file__).parent)],
        )

    decision = register_with_champion_challenger(
        info.model_uri, cfg.FORECASTER_NAME, run.info.run_id, "val_skill_mean", skill_mean,
        cfg.FORECASTER_TOLERANCE,
    )
    summary = {"run_id": run.info.run_id, "val_skill_mean": skill_mean, **decision}
    return summary, signal_features_from_oof(oofs)
