"""Experimento: variantes del modelo de cartera frente a reglas simples (walk-forward)."""
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit

from marketml import allocator as A, forecaster as F
from marketml.data import download_prices

logging.disable(logging.WARNING)
px = download_prices(sorted(set(F.TICKERS) | set(A.TICKERS)), "2006-01-01")
data = A.build_dataset(px).dropna(subset=["target"])
data["rank_target"] = data.groupby(level="date")["fwd_ret"].rank(pct=True)
data["simple"] = np.exp(data["fwd_ret"]) - 1
dates = data.index.get_level_values("date").unique().sort_values()

SMALL = ["mom_21", "mom_63", "mom_126", "mom_12_1", "vol_63", "dd_252", "ma200_dist", "asset_class_code", "vix", "term_structure"]
PARAMS = {
    "muy_reg": dict(max_iter=150, learning_rate=0.03, max_depth=2, min_samples_leaf=1000, l2_regularization=10.0),
    "medio": dict(max_iter=200, learning_rate=0.05, max_depth=3, min_samples_leaf=200, l2_regularization=1.0),
    "suelto": dict(max_iter=300, learning_rate=0.05, max_depth=4, min_samples_leaf=50, l2_regularization=0.0),
}

def oof_scores(target, feats, params):
    out = []
    for tr, te in TimeSeriesSplit(n_splits=5, gap=A.HORIZON).split(dates):
        d = data.index.get_level_values("date")
        trn, tst = data[d.isin(dates[tr])], data[d.isin(dates[te])]
        m = HistGradientBoostingRegressor(categorical_features=["asset_class_code"], random_state=42, **params)
        m.fit(trn[feats], trn[target])
        out.append(pd.Series(m.predict(tst[feats]), index=tst.index))
    return pd.concat(out)

test_dates = None
def evaluate(score):
    global test_dates
    df = data.loc[score.index].assign(score=score)
    rebal = df.index.get_level_values("date").unique().sort_values()[::A.HORIZON]
    rets, spread = [], []
    for dt in rebal:
        g = df.xs(dt, level="date")
        r = g["score"].rank(ascending=False, method="first")
        rets.append(g.loc[r <= 4, "simple"].mean())
        spread.append(g.loc[r <= 4, "simple"].mean() - g.loc[r >= 9, "simple"].mean())
    rets, spread = pd.Series(rets, index=rebal), pd.Series(spread, index=rebal)
    ic = df.loc[df.index.get_level_values("date").isin(rebal)].groupby(level="date").apply(
        lambda g: g["score"].corr(g["fwd_ret"], method="spearman"))
    ic_y = ic.groupby(ic.index.year).mean()
    eq = (1 + rets).cumprod()
    return {"sharpe": rets.mean() / rets.std() * np.sqrt(12), "cagr": eq.iloc[-1] ** (12 / len(rets)) - 1,
            "maxdd": (eq / eq.cummax() - 1).min(), "spread_anual": spread.mean() * 12,
            "ic": ic.mean(), "años_ic>0": (ic_y > 0).mean()}

rows = {}
base_idx = oof_scores("target", A.ASSET_FEATURES, PARAMS["muy_reg"]).index
rows["REGLA momentum 12-1"] = evaluate(data.loc[base_idx, "mom_12_1"])
rows["REGLA momentum 6m"] = evaluate(data.loc[base_idx, "mom_126"])
rows["REGLA equiponderada (= azar)"] = evaluate(pd.Series(np.random.default_rng(0).random(len(base_idx)), index=base_idx))
for tgt in ("target", "rank_target"):
    for fname, feats in (("todas", A.ASSET_FEATURES), ("reducidas", SMALL)):
        for pname, p in PARAMS.items():
            rows[f"ML {tgt:11s} {fname:9s} {pname}"] = evaluate(oof_scores(tgt, feats, p))
res = pd.DataFrame(rows).T
pd.set_option("display.width", 200)
print(res.round(3).sort_values("sharpe", ascending=False).to_string())
