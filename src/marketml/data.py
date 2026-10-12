"""Descarga de precios de cierre.

- VIX y VIX3M: Cboe, que es quien calcula el índice (CSV público, con el cierre del mismo día).
  FRED republica esos mismos cierres con un día de retraso y se usa para:
    * completar la historia anterior al inicio del CSV de Cboe (el VIX3M de Cboe empieza en 2009-09;
      el de FRED, VXVCLS, en 2007-12, lo que conserva la crisis de 2008 en el entrenamiento);
    * sustituir a Cboe si su descarga falla.
  Yahoo queda como último recurso.
- SPY y ETFs: Yahoo Finance (precios ajustados por dividendos). FRED no publica
  ETFs; para el SPY se usa el índice SP500 de FRED solo como respaldo en la API.
"""
import io
import logging
import time

import pandas as pd
import requests
import yfinance as yf

log = logging.getLogger(__name__)

CBOE_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{}_History.csv"
CBOE_PRIMARY = {"^VIX": "VIX", "^VIX3M": "VIX3M"}
FRED_PRIMARY = {"^VIX": "VIXCLS", "^VIX3M": "VXVCLS"}
# SP500 es el índice (sin dividendos y solo 10 años de historia): vale como
# respaldo para calcular las features de hoy, no para entrenar.
FRED_FALLBACK = {"SPY": "SP500"}


class DataError(RuntimeError):
    pass


def _yahoo(tickers: list[str], start: str) -> pd.DataFrame:
    last_err = None
    for attempt in range(3):
        try:
            raw = yf.download(tickers, start=start, auto_adjust=True, progress=False, threads=False)
            close = raw["Close"]
            missing = [t for t in tickers if t not in close.columns or close[t].dropna().empty]
            if missing:
                raise DataError(f"Yahoo no devolvió datos para {missing}")
            return close[tickers]
        except Exception as e:  # noqa: BLE001 - yfinance lanza tipos muy variados
            last_err = e
            log.warning("Yahoo intento %d falló: %s", attempt + 1, e)
            time.sleep(2 * (attempt + 1))
    raise DataError(f"Yahoo Finance no disponible: {last_err}")


def _fred(series_id: str, start: str) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start}"
    last_err = None
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            s = pd.read_csv(io.StringIO(r.text), index_col=0, parse_dates=True, na_values=".").iloc[:, 0]
            s = s.astype(float).dropna()  # FRED marca los festivos con "."
            if s.empty:
                raise DataError(f"FRED {series_id} vacío")
            return s
        except (requests.RequestException, ValueError, DataError) as e:
            last_err = e
            log.warning("FRED %s intento %d falló: %s", series_id, attempt + 1, e)
            time.sleep(2 * (attempt + 1))
    raise DataError(f"FRED {series_id} no disponible: {last_err}")


def _cboe(symbol: str, start: str) -> pd.Series:
    url = CBOE_URL.format(symbol)
    last_err = None
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0 (market-ml-lab)"})
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            s = pd.Series(df["CLOSE"].astype(float).to_numpy(),
                          index=pd.to_datetime(df["DATE"], format="%m/%d/%Y"), name=symbol)
            s = s[s.index >= start].dropna()
            s = s[s > 0]
            if s.empty:
                raise DataError(f"Cboe {symbol} vacío")
            return s
        except (requests.RequestException, KeyError, ValueError, DataError) as e:
            last_err = e
            log.warning("Cboe %s intento %d falló: %s", symbol, attempt + 1, e)
            time.sleep(2 * (attempt + 1))
    raise DataError(f"Cboe {symbol} no disponible: {last_err}")


def _vix_series(ticker: str, start: str) -> tuple[pd.Series, str]:
    """Cboe como fuente principal; FRED para la historia anterior y como respaldo."""
    try:
        cboe = _cboe(CBOE_PRIMARY[ticker], start)
    except DataError as e:
        log.warning("%s; se usa FRED para %s", e, ticker)
        return _fred(FRED_PRIMARY[ticker], start), f"FRED:{FRED_PRIMARY[ticker]} (respaldo)"

    first = cboe.index.min()
    if first <= pd.Timestamp(start) + pd.Timedelta(days=7):
        return cboe, f"Cboe:{CBOE_PRIMARY[ticker]}"
    try:
        fred = _fred(FRED_PRIMARY[ticker], start)
    except DataError as e:
        log.warning("%s; %s solo tendrá historia desde %s", e, ticker, first.date())
        return cboe, f"Cboe:{CBOE_PRIMARY[ticker]}"

    overlap = cboe.index.intersection(fred.index)
    gap = (cboe[overlap] - fred[overlap]).abs().mean() if len(overlap) else float("nan")
    log.info("%s: Cboe desde %s; antes se completa con FRED %s (dif. media en solape %.4f)",
             ticker, first.date(), FRED_PRIMARY[ticker], gap)
    spliced = pd.concat([fred[fred.index < first], cboe])
    return spliced, f"Cboe:{CBOE_PRIMARY[ticker]} (+FRED:{FRED_PRIMARY[ticker]} antes de {first.date()})"


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df.sort_index()


def download_prices(tickers: list[str], start: str, allow_fred_fallback: bool = False) -> pd.DataFrame:
    """Devuelve un DataFrame de cierres (índice = fecha, columnas = tickers) sin huecos.

    px.attrs["sources"] indica de dónde salió cada serie.
    """
    sources = {}
    direct_cols = {}
    for t in [t for t in tickers if t in CBOE_PRIMARY]:
        try:
            direct_cols[t], sources[t] = _vix_series(t, start)
        except DataError as e:
            log.warning("%s; se usará Yahoo para %s", e, t)

    yahoo_tickers = [t for t in tickers if t not in direct_cols]
    yahoo = None
    if yahoo_tickers:
        try:
            yahoo = _normalize(_yahoo(yahoo_tickers, start))
            sources.update({t: "Yahoo" for t in yahoo_tickers})
        except DataError:
            if not allow_fred_fallback or not set(yahoo_tickers) <= set(FRED_FALLBACK):
                raise
            log.warning("Usando FRED como respaldo para %s", yahoo_tickers)
            direct_cols.update({t: _fred(FRED_FALLBACK[t], start) for t in yahoo_tickers})
            sources.update({t: f"FRED:{FRED_FALLBACK[t]}" for t in yahoo_tickers})

    direct = _normalize(pd.DataFrame(direct_cols)) if direct_cols else None
    if yahoo is not None and direct is not None:
        # Calendario de negociación = el de Yahoo; Cboe/FRED se alinean a esas fechas.
        px = yahoo.join(direct, how="left")
    else:
        px = yahoo if yahoo is not None else direct
    px = px[tickers]

    # FRED publica con un día de retraso (si actúa de respaldo): no rellenar el último día con el dato
    # de ayer, sino cortar en la última fecha en que TODAS las series tienen dato.
    last_common = min(px[c].last_valid_index() for c in px.columns)
    px = px.loc[:last_common].ffill(limit=3).dropna()
    if len(px) < 300:
        raise DataError(f"Muy pocos datos tras limpiar: {len(px)} filas")
    px.attrs["sources"] = sources
    return px
