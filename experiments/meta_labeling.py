"""Experimento: momentum + ML como filtro de falsos positivos (meta-labeling).

1. Regla primaria: candidatos = los N_CAND ETFs con mejor momentum 12-1.
2. Etiqueta: 1 si el candidato rinde por encima de la mediana de los 12 el mes siguiente.
   Un candidato con etiqueta 0 es un falso positivo del momentum.
3. Clasificador (solo sobre candidatos) que estima P(acierto).
4. Se compara la cartera de momentum pura con varias formas de usar P(acierto).
"""
import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from marketml import allocator as A, forecaster as F
from marketml.data import download_prices

logging.disable(logging.WARNING)
N_CAND, K = 6, 4

px = download_prices(sorted(set(F.TICKERS) | set(A.TICKERS)), "2006-01-01")
data = A.build_dataset(px).dropna(subset=["fwd_ret"])
data["simple"] = np.exp(data["fwd_ret"]) - 1
by_date = data.groupby(level="date")
data["mom_rank"] = by_date["mom_12_1"].rank(ascending=False, method="first")
data["label"] = (data["fwd_ret"] > by_date["fwd_ret"].transform("median")).astype(int)
FEATS = A.ASSET_FEATURES + ["mom_rank"]
cand = data[data["mom_rank"] <= N_CAND]
dates = data.index.get_level_values("date").unique().sort_values()

PARAMS = {
    "muy_reg": dict(max_iter=150, learning_rate=0.03, max_depth=2, min_samples_leaf=500, l2_regularization=10.0),
    "medio": dict(max_iter=200, learning_rate=0.05, max_depth=3, min_samples_leaf=150, l2_regularization=1.0),
}


def oof_proba(params):
    out = []
    cd = cand.index.get_level_values("date")
    for tr, te in TimeSeriesSplit(n_splits=5, gap=A.HORIZON).split(dates):
        trn, tst = cand[cd.isin(dates[tr])], cand[cd.isin(dates[te])]
        m = HistGradientBoostingClassifier(categorical_features=["asset_class_code"], random_state=42, **params)
        m.fit(trn[FEATS], trn["label"])
        out.append(pd.Series(m.predict_proba(tst[FEATS])[:, 1], index=tst.index))
    return pd.concat(out)


def perf(r):
    eq = (1 + r).cumprod()
    return {"sharpe": r.mean() / r.std() * np.sqrt(12), "cagr": eq.iloc[-1] ** (12 / len(r)) - 1,
            "vol": r.std() * np.sqrt(12), "maxdd": (eq / eq.cummax() - 1).min()}


def backtest(proba):
    rebal = proba.index.get_level_values("date").unique().sort_values()[::A.HORIZON]
    strat = {k: [] for k in ("momentum top4", "meta: top4 por P entre 6 candidatos",
                             "meta: momentum top4, quita P<0.5 (a equiponderada)",
                             "meta: momentum top4, quita P<0.5 (a IEF)", "equiponderada", "SPY")}
    precision = {"momentum": [], "filtrado": []}
    for d in rebal:
        g = data.xs(d, level="date")
        p = proba.xs(d, level="date")
        top4 = g["mom_rank"].nsmallest(K).index
        ew = g["simple"].mean()
        strat["momentum top4"].append(g.loc[top4, "simple"].mean())
        strat["meta: top4 por P entre 6 candidatos"].append(g.loc[p.nlargest(K).index, "simple"].mean())
        keep = [t for t in top4 if p[t] >= 0.5]
        n_drop = K - len(keep)
        filled = g.loc[keep, "simple"].sum()
        strat["meta: momentum top4, quita P<0.5 (a equiponderada)"].append((filled + n_drop * ew) / K)
        strat["meta: momentum top4, quita P<0.5 (a IEF)"].append((filled + n_drop * g.loc["IEF", "simple"]) / K)
        strat["equiponderada"].append(ew)
        strat["SPY"].append(g.loc["SPY", "simple"])
        precision["momentum"] += list(g.loc[top4, "label"])
        precision["filtrado"] += list(g.loc[keep, "label"])
    res = pd.DataFrame({k: perf(pd.Series(v)) for k, v in strat.items()}).T
    prec = {k: (np.mean(v), len(v)) for k, v in precision.items()}
    return res, prec


pd.set_option("display.width", 200)
for name, params in PARAMS.items():
    proba = oof_proba(params)
    y = cand.loc[proba.index, "label"]
    print(f"\n=== Clasificador '{name}'  AUC fuera de muestra: {roc_auc_score(y, proba):.3f} "
          f"(0.5 = azar) | aciertos base de los candidatos: {y.mean():.1%}")
    res, prec = backtest(proba)
    print(res.round(3).to_string())
    print(f"Precisión (candidato rinde > mediana): momentum top4 {prec['momentum'][0]:.1%} "
          f"({prec['momentum'][1]} posiciones) -> tras el filtro {prec['filtrado'][0]:.1%} "
          f"({prec['filtrado'][1]} posiciones)")
