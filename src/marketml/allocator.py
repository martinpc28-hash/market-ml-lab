"""Modelo 2: cartera de 12 ETFs (inferencia batch). Cuatro estrategias compiten y
solo la que se gana el puesto llega a producción.

Estrategias, de la más simple a la más compleja
-----------------------------------------------
1. equal_weight  Los 12 ETFs a partes iguales (equivale, en media, a elegir al azar).
2. momentum      Los 4 ETFs con mejor rentabilidad a 12 meses sin el último mes.
3. meta          Meta-labeling: el momentum propone 6 candidatos y un clasificador
                 estima la probabilidad de que cada uno rinda por encima de la mediana
                 el mes siguiente (es decir, de que NO sea un falso positivo). Se
                 quedan los 4 más probables. Entre sus variables están las
                 predicciones del modelo 1 (VIX/SPY), que es el que sí tiene poder
                 predictivo.
4. ml            Ranking: un regresor predice el puesto (percentil 0-1) de cada ETF
                 entre los 12 y se compran los 4 mejores.

Filtro antes de producción (regla de parsimonia)
-----------------------------------------------
Las cuatro se evalúan en el mismo backtest walk-forward (top 4 equiponderado,
rebalanceo cada 21 días hábiles). Se parte de la más simple y una más compleja
solo la sustituye si mejora su Sharpe en más de MARGIN. Así el ML tiene que
ganarse el puesto frente a reglas sencillas, no basta con empatar.

Inferencia batch: el job diario calcula la cartera con la estrategia campeona y
la guarda en GCS. La API solo la lee.
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
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import TimeSeriesSplit  # noqa: E402

from . import config as cfg  # noqa: E402
from .features import ASSET_FEATURE_INFO, ASSET_FEATURES, asset_features, market_features  # noqa: E402
from .forecaster import SIGNAL_FEATURE_INFO, SIGNAL_FEATURES  # noqa: E402
from .store import ensure_experiment, register_with_champion_challenger, write_json  # noqa: E402

log = logging.getLogger(__name__)

ASSET_CLASSES = {0: "Acciones", 1: "Bonos", 2: "Materias primas", 3: "Inmobiliario"}
UNIVERSE = {
    "SPY": ("S&P 500", 0), "QQQ": ("Nasdaq 100", 0), "IWM": ("Russell 2000", 0),
    "EFA": ("Desarrollados ex-EE. UU.", 0), "EEM": ("Emergentes", 0),
    "TLT": ("Tesoro EE. UU. 20+ años", 1), "IEF": ("Tesoro EE. UU. 7-10 años", 1),
    "LQD": ("Corporativos grado inversión", 1), "HYG": ("High yield", 1),
    "GLD": ("Oro", 2), "DBC": ("Materias primas", 2), "VNQ": ("REITs EE. UU.", 3),
}
TICKERS = list(UNIVERSE)
HORIZON = 21
TOP_K = 4
N_CANDIDATES = 6
MARGIN = 0.05  # mejora mínima de Sharpe para que una estrategia más compleja sustituya a la simple

LADDER = ["equal_weight", "momentum", "meta", "ml"]
STRATEGIES = {
    "equal_weight": "Los 12 ETFs a partes iguales",
    "momentum": "Regla de momentum: los 4 ETFs con mejor rentabilidad a 12 meses (sin el último mes)",
    "meta": "Momentum + ML: de los 6 mejores por momentum, los 4 con más probabilidad de acierto según el clasificador",
    "ml": "Ranking ML: los 4 ETFs con mejor puesto previsto por el modelo",
}
SHORT = {"equal_weight": "equiponderada", "momentum": "momentum", "meta": "momentum + ML", "ml": "ranking ML"}

RANK_FEATURES = ASSET_FEATURES
META_FEATURES = ASSET_FEATURES + ["mom_rank"] + SIGNAL_FEATURES

# Elegidos con los experimentos walk-forward de experiments/.
RANK_PARAMS = {"max_iter": 150, "learning_rate": 0.03, "max_depth": 2, "min_samples_leaf": 1000,
               "l2_regularization": 10.0}
META_PARAMS = {"max_iter": 200, "learning_rate": 0.05, "max_depth": 3, "min_samples_leaf": 150,
               "l2_regularization": 1.0}


def build_dataset(px: pd.DataFrame, signals: pd.DataFrame | None = None) -> pd.DataFrame:
    """Panel (date, ticker). `signals`: variables del modelo 1 por fecha (puede faltar)."""
    mkt = market_features(px[["SPY", "^VIX", "^VIX3M"]])
    panel = asset_features(px, TICKERS, {t: c for t, (_, c) in UNIVERSE.items()}, mkt)
    fwd = np.log(px[TICKERS].shift(-HORIZON) / px[TICKERS]).stack().rename("fwd_ret")
    fwd.index.names = ["date", "ticker"]
    data = panel.join(fwd)
    sig = signals if signals is not None else pd.DataFrame(columns=SIGNAL_FEATURES, index=pd.DatetimeIndex([]),
                                                            dtype=float)
    data = data.join(sig[SIGNAL_FEATURES].rename_axis("date"), on="date")

    by_date = data.groupby(level="date")
    data["mom_rank"] = by_date["mom_12_1"].rank(ascending=False, method="first")
    # Objetivo del ranking: percentil (0-1) del ETF entre los 12 por rentabilidad del mes siguiente.
    data["target"] = by_date["fwd_ret"].rank(pct=True)
    # Etiqueta del meta-labeling: ¿rinde por encima de la mediana? (0 = falso positivo si era candidato)
    data["label"] = (data["fwd_ret"] > by_date["fwd_ret"].transform("median")).astype(float)
    data.loc[data["fwd_ret"].isna(), ["target", "label"]] = np.nan
    return data


def _ranker(params=RANK_PARAMS):
    return HistGradientBoostingRegressor(categorical_features=["asset_class_code"], random_state=42, **params)


def _meta(params=META_PARAMS):
    return HistGradientBoostingClassifier(categorical_features=["asset_class_code"], random_state=42, **params)


def _usable(cols: list[str], frame: pd.DataFrame) -> list[str]:
    # Las señales del modelo 1 solo existen desde que hay predicciones fuera de muestra
    # (~2011): en los primeros tramos del walk-forward esas columnas están vacías.
    return [c for c in cols if frame[c].notna().any()]


def _top(scores: pd.Series, k: int = TOP_K) -> pd.Index:
    # method="first": desempate determinista si hay puntuaciones iguales
    return scores[scores.rank(ascending=False, method="first") <= k].index


class AllocationStrategy(mlflow.pyfunc.PythonModel):
    """Lo que se registra en MLflow: la estrategia ganadora. Entra el panel de un día
    (una fila por ETF) y salen los pesos de la cartera."""

    def __init__(self, mode: str, ranker, meta, meta_features: list[str], metadata: dict):
        self.mode, self.ranker, self.meta, self.metadata = mode, ranker, meta, metadata
        self.meta_features = meta_features

    def scores(self, day: pd.DataFrame) -> dict:
        day = day.assign(mom_rank=day["mom_12_1"].rank(ascending=False, method="first"))
        cand = day[day["mom_rank"] <= N_CANDIDATES]
        return {
            "ml": pd.Series(self.ranker.predict(day[RANK_FEATURES]), index=day.index),
            "meta": pd.Series(self.meta.predict_proba(cand[self.meta_features])[:, 1], index=cand.index),
        }

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.Series:
        if self.mode == "equal_weight":
            return pd.Series(1 / len(model_input), index=model_input.index)
        if self.mode == "momentum":
            picks = _top(model_input["mom_12_1"])
        else:
            picks = _top(self.scores(model_input)[self.mode])
        return pd.Series(np.where(model_input.index.isin(picks), 1 / TOP_K, 0.0), index=model_input.index)


def walk_forward(data: pd.DataFrame, n_splits: int) -> pd.DataFrame:
    d = data.dropna(subset=["target"])
    dates = d.index.get_level_values("date").unique().sort_values()
    oof = []
    for tr, te in TimeSeriesSplit(n_splits=n_splits, gap=HORIZON).split(dates):
        dd = d.index.get_level_values("date")
        train, test = d[dd.isin(dates[tr])], d[dd.isin(dates[te])].copy()
        test["score"] = _ranker().fit(train[RANK_FEATURES], train["target"]).predict(test[RANK_FEATURES])
        cand_tr, cand_te = train[train["mom_rank"] <= N_CANDIDATES], test["mom_rank"] <= N_CANDIDATES
        feats = _usable(META_FEATURES, cand_tr)
        meta = _meta().fit(cand_tr[feats], cand_tr["label"])
        test.loc[cand_te, "meta_p"] = meta.predict_proba(test.loc[cand_te, feats])[:, 1]
        oof.append(test)
    return pd.concat(oof)


def _perf(rets: pd.Series) -> dict:
    eq = (1 + rets).cumprod()
    per_year = 252 / HORIZON
    return {
        "cagr": float(eq.iloc[-1] ** (per_year / len(rets)) - 1),
        "vol": float(rets.std() * np.sqrt(per_year)),
        "sharpe": float(rets.mean() / rets.std() * np.sqrt(per_year)),
        "max_drawdown": float((eq / eq.cummax() - 1).min()),
    }


def evaluate(oof: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.Series]:
    rebal = oof.index.get_level_values("date").unique().sort_values()[::HORIZON]
    rows, ics, hits = [], {}, {"momentum": [], "meta": []}
    for d in rebal:
        g = oof.xs(d, level="date")
        simple = np.exp(g["fwd_ret"]) - 1
        picks = {"ml": _top(g["score"]), "momentum": _top(-g["mom_rank"]), "meta": _top(g["meta_p"].dropna())}
        bottom = g["score"].rank(ascending=True, method="first") <= TOP_K
        rows.append({
            "date": d,
            **{k: float(simple[v].mean()) for k, v in picks.items()},
            "equal_weight": float(simple.mean()),
            "spy": float(simple["SPY"]),
            "ml_spread": float(simple[picks["ml"]].mean() - simple[bottom].mean()),
        })
        ics[d] = g["score"].corr(g["fwd_ret"], method="spearman")
        hits["momentum"] += list(g.loc[picks["momentum"], "label"])
        hits["meta"] += list(g.loc[picks["meta"], "label"])
    bt = pd.DataFrame(rows).set_index("date")
    ic = pd.Series(ics)
    ic_by_year = ic.groupby(ic.index.year).mean()
    cand = oof.dropna(subset=["meta_p"])

    metrics = {
        "ic_mean": float(ic.mean()),
        "ic_years_positive": float((ic_by_year > 0).mean()),
        "ml_spread_annual": float(bt["ml_spread"].mean() * 252 / HORIZON),
        "meta_auc": float(roc_auc_score(cand["label"], cand["meta_p"])),
        "momentum_precision": float(np.mean(hits["momentum"])),
        "meta_precision": float(np.mean(hits["meta"])),
    }
    for col in ("ml", "meta", "momentum", "equal_weight", "spy"):
        metrics.update({f"{col}_{k}": v for k, v in _perf(bt[col]).items()})
    return metrics, bt.drop(columns="ml_spread"), ic_by_year


def choose_strategy(m: dict) -> tuple[str, str]:
    """Regla de parsimonia: se parte de la más simple; una más compleja solo la
    sustituye si su Sharpe la supera en más de MARGIN."""
    chosen, notes = LADDER[0], []
    for s in LADDER[1:]:
        if s == "ml" and m["ml_spread_annual"] <= 0:
            notes.append(f"{SHORT[s]} descartado: sus 4 favoritos no rinden más que sus 4 últimos")
            continue
        if m[f"{s}_sharpe"] > m[f"{chosen}_sharpe"] + MARGIN:
            notes.append(f"{SHORT[s]} ({m[f'{s}_sharpe']:.2f}) supera a {SHORT[chosen]} ({m[f'{chosen}_sharpe']:.2f})")
            chosen = s
        else:
            notes.append(f"{SHORT[s]} ({m[f'{s}_sharpe']:.2f}) no supera a {SHORT[chosen]} "
                         f"({m[f'{chosen}_sharpe']:.2f}) en más de {MARGIN}")
    reason = f"En producción: {SHORT[chosen]}. " + "; ".join(notes) + "."
    return chosen, reason


def _plot(bt: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = {"ml": "Ranking ML", "meta": "Momentum + ML (meta-labeling)", "momentum": "Momentum 12-1",
              "equal_weight": "Equiponderada 12 ETFs", "spy": "SPY"}
    for col, label in labels.items():
        ax.plot(bt.index, (1 + bt[col]).cumprod(), label=label)
    ax.set_yscale("log")
    ax.set_title("Backtest walk-forward (top 4, rebalanceo cada 21 días hábiles)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def train(px: pd.DataFrame, signals: pd.DataFrame | None = None, n_splits: int = 5, trigger: str = "manual",
          sources: dict | None = None) -> dict:
    """`signals`: predicciones fuera de muestra del modelo 1 (forecaster.signal_features_from_oof)."""
    data = build_dataset(px, signals)
    ensure_experiment(cfg.ALLOCATOR_EXPERIMENT)
    dates = data.index.get_level_values("date")

    with mlflow.start_run(run_name=f"allocator-{dates.max().date()}") as run:
        mlflow.set_tags({f"data_source.{t.lstrip('^')}": s for t, s in (sources or {}).items()})
        mlflow.set_tags({"trigger": trigger, "model_family": "HistGradientBoosting ranking + meta-labeling"})
        mlflow.log_params({**{f"rank_{k}": v for k, v in RANK_PARAMS.items()},
                           **{f"meta_{k}": v for k, v in META_PARAMS.items()},
                           "n_splits": n_splits, "universe": ",".join(TICKERS), "top_k": TOP_K,
                           "n_candidates": N_CANDIDATES, "margin": MARGIN, "horizon_days": HORIZON,
                           "train_start": dates.min().date(), "train_end": dates.max().date(),
                           "n_rows": len(data), "uses_vix_model_signals": signals is not None})

        oof = walk_forward(data, n_splits)
        metrics, bt, ic_by_year = evaluate(oof)
        strategy, reason = choose_strategy(metrics)
        metrics["strategy_sharpe"] = metrics[f"{strategy}_sharpe"]
        mlflow.log_metrics(metrics)
        mlflow.set_tags({"production_strategy": strategy, "gate_reason": reason})
        log.info("Sharpe EW=%.2f momentum=%.2f meta=%.2f ml=%.2f SPY=%.2f | meta AUC=%.3f precision %.3f->%.3f",
                 metrics["equal_weight_sharpe"], metrics["momentum_sharpe"], metrics["meta_sharpe"],
                 metrics["ml_sharpe"], metrics["spy_sharpe"], metrics["meta_auc"],
                 metrics["momentum_precision"], metrics["meta_precision"])
        log.info(reason)

        curve = (1 + bt).cumprod().reset_index()
        curve["date"] = curve["date"].dt.date.astype(str)
        ic_years = {str(y): round(float(v), 4) for y, v in ic_by_year.items()}
        mlflow.log_dict({"metrics": metrics, "strategy": strategy, "reason": reason, "ic_by_year": ic_years,
                         "curve": curve.round(4).to_dict(orient="records")},
                        "backtest/allocation_backtest.json")
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "equity_curve.png"
            _plot(bt, p)
            mlflow.log_artifact(str(p), "plots")

        rows = data.dropna(subset=["target"])
        ranker = _ranker().fit(rows[RANK_FEATURES], rows["target"])
        cand = rows[rows["mom_rank"] <= N_CANDIDATES]
        meta_feats = _usable(META_FEATURES, cand)
        meta = _meta().fit(cand[meta_feats], cand["label"])
        packaged = AllocationStrategy(strategy, ranker, meta, meta_feats, {
            "strategy": strategy, "reason": reason, "metrics": metrics, "ic_by_year": ic_years,
            "run_id": run.info.run_id})
        example = rows.xs(rows.index.get_level_values("date").max(), level="date")[
            sorted(set(RANK_FEATURES) | set(META_FEATURES) - {"mom_rank"})]
        info = mlflow.pyfunc.log_model(
            name="model", python_model=packaged, input_example=example,
            signature=infer_signature(example, packaged.predict(None, example)),
            code_paths=[str(Path(__file__).parent)],
        )

    decision = register_with_champion_challenger(
        info.model_uri, cfg.ALLOCATOR_NAME, run.info.run_id, "strategy_sharpe", metrics["strategy_sharpe"],
        cfg.ALLOCATOR_TOLERANCE,
    )
    return {"run_id": run.info.run_id, "strategy": strategy, "strategy_sharpe": metrics["strategy_sharpe"],
            **decision}


def batch_predict(px: pd.DataFrame, signals_today: pd.DataFrame | None = None) -> dict:
    """Carga la estrategia CAMPEONA del registry y calcula la cartera de hoy.
    `signals_today`: predicciones del modelo 1 campeón para la última fecha."""
    client = mlflow.tracking.MlflowClient()
    champ = client.get_model_version_by_alias(cfg.ALLOCATOR_NAME, "champion")
    strat = mlflow.pyfunc.load_model(f"models:/{cfg.ALLOCATOR_NAME}@champion").unwrap_python_model()

    data = build_dataset(px, signals_today)
    as_of = data.index.get_level_values("date").max()
    today = data.xs(as_of, level="date")
    weights = strat.predict(None, today)
    sc = strat.scores(today)

    rows = []
    order = today.assign(w=weights).sort_values(["w", "mom_12_1"], ascending=False).index
    for rank, t in enumerate(order, start=1):
        name, cls = UNIVERSE[t]
        rows.append({
            "ticker": t, "name": name, "asset_class": ASSET_CLASSES[cls], "rank": rank,
            "weight": round(float(weights[t]), 4),
            "mom_12_1_pct": round(float(today.loc[t, "mom_12_1"] * 100), 2),
            "momentum_candidate": bool(t in sc["meta"].index),
            "meta_prob_pct": round(float(sc["meta"][t] * 100), 1) if t in sc["meta"].index else None,
            "ml_score": round(float(sc["ml"][t] * 100), 1),
            "vol_63_pct": round(float(today.loc[t, "vol_63"] * 100), 2),
        })

    sig = today[SIGNAL_FEATURES].iloc[0]
    result = {
        "as_of": as_of.date().isoformat(),
        "strategy": strat.mode,
        "strategy_description": STRATEGIES[strat.mode],
        "reason": strat.metadata["reason"],
        "model": {"name": cfg.ALLOCATOR_NAME, "version": int(champ.version), "run_id": champ.run_id},
        "method": f"Top {TOP_K} ETFs equiponderados, rebalanceo mensual" if strat.mode != "equal_weight"
                  else "12 ETFs a partes iguales",
        "vix_model_signals": {k: (None if pd.isna(v) else round(float(v), 4)) for k, v in sig.items()},
        "feature_info": {**ASSET_FEATURE_INFO, **SIGNAL_FEATURE_INFO},
        "allocations": rows,
    }
    write_json(cfg.ALLOCATION_LATEST, result)
    write_json(f"predictions/allocation/history/{result['as_of']}.json", result)
    log.info("Cartera %s (%s): %s", result["as_of"], strat.mode, [r["ticker"] for r in rows if r["weight"]])
    return result
