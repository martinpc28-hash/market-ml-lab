"""Ingeniería de variables (features).

Regla de oro: la fila de la fecha t solo usa información disponible al cierre
de t. Los objetivos (lo que se predice) miran hacia delante y se construyen
aparte, en cada modelo. tests/test_features.py comprueba que no hay fuga de
información futura.
"""
import numpy as np
import pandas as pd

ANNUAL = np.sqrt(252)

# Descripción en lenguaje claro de cada variable (la API y la web la muestran).
MARKET_FEATURE_INFO = {
    "vix": "Nivel actual del VIX (volatilidad implícita del S&P 500 a 30 días)",
    "vix_chg_1d": "Cambio logarítmico del VIX en 1 día",
    "vix_chg_5d": "Cambio logarítmico del VIX en 1 semana",
    "vix_chg_21d": "Cambio logarítmico del VIX en 1 mes",
    "vix_z_63": "Z-score del VIX frente a su media de 3 meses (¿está anormalmente alto?)",
    "vix_ma_ratio_21": "VIX dividido por su media móvil de 21 días",
    "term_structure": "VIX / VIX3M. Por encima de 1 = backwardation (estrés agudo)",
    "spy_ret_1d": "Rentabilidad del SPY en 1 día",
    "spy_ret_5d": "Rentabilidad del SPY en 1 semana",
    "spy_ret_21d": "Rentabilidad del SPY en 1 mes",
    "spy_ret_63d": "Rentabilidad del SPY en 3 meses",
    "rv_5": "Volatilidad realizada del SPY a 5 días (anualizada, en puntos como el VIX)",
    "rv_21": "Volatilidad realizada del SPY a 21 días",
    "rv_63": "Volatilidad realizada del SPY a 63 días",
    "vrp": "Prima de riesgo de volatilidad: VIX menos volatilidad realizada a 21 días",
    "spy_drawdown_252": "Caída del SPY desde su máximo de 12 meses",
    "spy_ma50_dist": "Distancia del SPY a su media móvil de 50 días",
    "spy_ma200_dist": "Distancia del SPY a su media móvil de 200 días",
}
MARKET_FEATURES = list(MARKET_FEATURE_INFO)


def market_features(px: pd.DataFrame) -> pd.DataFrame:
    """px: cierres con columnas SPY, ^VIX, ^VIX3M."""
    spy, vix, vix3m = px["SPY"], px["^VIX"], px["^VIX3M"]
    lvix = np.log(vix)
    r = np.log(spy).diff()

    f = pd.DataFrame(index=px.index)
    f["vix"] = vix
    for n in (1, 5, 21):
        f[f"vix_chg_{n}d"] = lvix.diff(n)
    f["vix_z_63"] = (vix - vix.rolling(63).mean()) / vix.rolling(63).std()
    f["vix_ma_ratio_21"] = vix / vix.rolling(21).mean()
    f["term_structure"] = vix / vix3m
    for n in (1, 5, 21, 63):
        f[f"spy_ret_{n}d"] = np.log(spy).diff(n)
    for n in (5, 21, 63):
        f[f"rv_{n}"] = r.rolling(n).std() * ANNUAL * 100
    f["vrp"] = vix - f["rv_21"]
    f["spy_drawdown_252"] = spy / spy.rolling(252).max() - 1
    f["spy_ma50_dist"] = spy / spy.rolling(50).mean() - 1
    f["spy_ma200_dist"] = spy / spy.rolling(200).mean() - 1
    return f[MARKET_FEATURES].dropna()


ASSET_FEATURE_INFO = {
    "mom_21": "Momentum a 1 mes del ETF",
    "mom_63": "Momentum a 3 meses",
    "mom_126": "Momentum a 6 meses",
    "mom_12_1": "Momentum a 12 meses excluyendo el último mes (clásico en la literatura)",
    "vol_21": "Volatilidad realizada a 1 mes",
    "vol_63": "Volatilidad realizada a 3 meses",
    "dd_252": "Caída desde el máximo de 12 meses",
    "ma200_dist": "Distancia a la media móvil de 200 días",
    "mom_63_rank": "Ranking del momentum a 3 meses frente al resto de ETFs (0-1)",
    "mom_12_1_rank": "Ranking del momentum 12-1 frente al resto de ETFs (0-1)",
    "vol_63_rank": "Ranking de volatilidad frente al resto de ETFs (0-1)",
    "asset_class_code": "Clase de activo (acciones, bonos, materias primas, inmobiliario)",
    "vix": "Nivel del VIX (contexto de mercado)",
    "term_structure": "VIX / VIX3M (contexto de estrés)",
    "spy_ma200_dist": "Tendencia del mercado de acciones (contexto)",
}
ASSET_FEATURES = list(ASSET_FEATURE_INFO)


def asset_features(px: pd.DataFrame, tickers: list[str], asset_class: dict[str, int],
                   mkt: pd.DataFrame) -> pd.DataFrame:
    """Panel largo con índice (date, ticker): una fila por ETF y día."""
    frames = []
    for t in tickers:
        c = px[t]
        lc = np.log(c)
        r = lc.diff()
        f = pd.DataFrame(index=px.index)
        for n in (21, 63, 126):
            f[f"mom_{n}"] = lc.diff(n)
        f["mom_12_1"] = lc.shift(21) - lc.shift(252)
        f["vol_21"] = r.rolling(21).std() * ANNUAL
        f["vol_63"] = r.rolling(63).std() * ANNUAL
        f["dd_252"] = c / c.rolling(252).max() - 1
        f["ma200_dist"] = c / c.rolling(200).mean() - 1
        f["asset_class_code"] = asset_class[t]
        f["ticker"] = t
        frames.append(f)
    panel = pd.concat(frames).rename_axis("date").reset_index().set_index(["date", "ticker"]).sort_index()

    by_date = panel.groupby(level="date")
    for col in ("mom_63", "mom_12_1", "vol_63"):
        panel[f"{col}_rank"] = by_date[col].rank(pct=True)

    ctx = mkt[["vix", "term_structure", "spy_ma200_dist"]]
    panel = panel.join(ctx, on="date")
    return panel[ASSET_FEATURES].dropna()
