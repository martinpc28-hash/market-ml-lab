"""Experimento: usar el modelo del VIX (que sí tiene poder predictivo) para mejorar la cartera.

Las predicciones del VIX/SPY que se usan son las fuera de muestra (OOF) del
walk-forward del modelo 1, así que no hay fuga de información. Los umbrales de
"estrés" se calculan con percentiles EXPANDING (solo con el pasado).

A) Como variables extra del modelo de ranking y del meta-labeling.
B) Como freno de riesgo sobre las carteras: si el modelo prevé estrés, se pasa
   a activos defensivos.
"""
import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit

from marketml import allocator as A, forecaster as F
from marketml.data import download_prices

logging.disable(logging.WARNING)
pd.set_option("display.width", 200)
DEFENSIVE = ["IEF", "TLT", "GLD", "LQD"]

px = download_prices(sorted(set(F.TICKERS) | set(A.TICKERS)), "2006-01-01")

# --- Predicciones OOF del modelo 1
X, Y = F.build_dataset(px[F.TICKERS])
fc = {}
for target in ("vix_1m", "spy_1m"):
    oof, _ = F.walk_forward(X, Y[target], 21, target, F.DEFAULT_PARAMS, 5)
    fc[target] = oof
fcf = pd.DataFrame({
    "vix_fc_1m": fc["vix_1m"]["pred"],          # cambio log previsto del VIX a 1 mes
    "vix_fc_1m_p90": fc["vix_1m"]["p90"],       # escenario alto del VIX
    "spy_fc_1m_p10": fc["spy_1m"]["p10"],       # escenario malo del SPY
})
# Señal de estrés: el p10 previsto del SPY está en su 20 % peor histórico (expanding, solo pasado)
thr = fcf["spy_fc_1m_p10"].expanding(min_periods=250).quantile(0.20).shift(1)
fcf["stress"] = (fcf["spy_fc_1m_p10"] < thr).astype(float)
print(f"Predicciones OOF del VIX desde {fcf.index[0].date()}; días en estrés: {fcf['stress'].mean():.0%}")

data = A.build_dataset(px).dropna(subset=["fwd_ret"])
data = data.join(fcf, on="date")
data = data[data["vix_fc_1m"].notna()]
data["simple"] = np.exp(data["fwd_ret"]) - 1
by_date = data.groupby(level="date")
data["mom_rank"] = by_date["mom_12_1"].rank(ascending=False, method="first")
data["label"] = (data["fwd_ret"] > by_date["fwd_ret"].transform("median")).astype(int)
dates = data.index.get_level_values("date").unique().sort_values()
FC = ["vix_fc_1m", "vix_fc_1m_p90", "spy_fc_1m_p10"]
P_RANK = dict(max_iter=150, learning_rate=0.03, max_depth=2, min_samples_leaf=1000, l2_regularization=10.0)
P_META = dict(max_iter=200, learning_rate=0.05, max_depth=3, min_samples_leaf=150, l2_regularization=1.0)


def oof(frame, feats, target, clf, params):
    out = []
    d = frame.index.get_level_values("date")
    for tr, te in TimeSeriesSplit(n_splits=4, gap=A.HORIZON).split(dates):
        trn, tst = frame[d.isin(dates[tr])], frame[d.isin(dates[te])]
        M = HistGradientBoostingClassifier if clf else HistGradientBoostingRegressor
        m = M(categorical_features=["asset_class_code"], random_state=42, **params).fit(trn[feats], trn[target])
        p = m.predict_proba(tst[feats])[:, 1] if clf else m.predict(tst[feats])
        out.append(pd.Series(p, index=tst.index))
    return pd.concat(out)


cand = data[data["mom_rank"] <= 6]
scores = {
    "ml": oof(data, A.ASSET_FEATURES, "target", False, P_RANK),
    "ml+vix": oof(data, A.ASSET_FEATURES + FC, "target", False, P_RANK),
    "meta": oof(cand, A.ASSET_FEATURES + ["mom_rank"], "label", True, P_META),
    "meta+vix": oof(cand, A.ASSET_FEATURES + ["mom_rank"] + FC, "label", True, P_META),
}
common = scores["ml"].index.get_level_values("date").unique().sort_values()
rebal = common[::A.HORIZON]


def picks(name, g, d):
    if name == "momentum":
        return g["mom_rank"].nsmallest(4).index
    if name in ("ml", "ml+vix"):
        s = scores[name].xs(d, level="date")
        return s[s.rank(ascending=False, method="first") <= 4].index
    s = scores[name].xs(d, level="date")  # meta: top 4 por P(acierto) entre 6 candidatos
    return s.nlargest(4).index


def run(name, overlay):
    rets = []
    for d in rebal:
        g = data.xs(d, level="date")
        if overlay and g["stress"].iloc[0] == 1:
            r = g.loc[DEFENSIVE, "simple"].mean()
        else:
            r = g.loc[picks(name, g, d), "simple"].mean()
        rets.append(r)
    r = pd.Series(rets, index=rebal)
    eq = (1 + r).cumprod()
    return {"sharpe": r.mean() / r.std() * np.sqrt(12), "cagr": eq.iloc[-1] ** (12 / len(r)) - 1,
            "vol": r.std() * np.sqrt(12), "maxdd": (eq / eq.cummax() - 1).min()}


rows = {}
for name in ("momentum", "ml", "ml+vix", "meta", "meta+vix"):
    rows[name] = run(name, False)
    rows[name + " | freno VIX"] = run(name, True)
ew = []
for d in rebal:
    ew.append(data.xs(d, level="date")["simple"].mean())
ew = pd.Series(ew)
eq = (1 + ew).cumprod()
rows["equiponderada"] = {"sharpe": ew.mean() / ew.std() * np.sqrt(12), "cagr": eq.iloc[-1] ** (12 / len(ew)) - 1,
                         "vol": ew.std() * np.sqrt(12), "maxdd": (eq / eq.cummax() - 1).min()}
print(f"Periodo: {rebal[0].date()} -> {rebal[-1].date()} ({len(rebal)} rebalanceos)\n")
print(pd.DataFrame(rows).T.round(3).sort_values("sharpe", ascending=False).to_string())
