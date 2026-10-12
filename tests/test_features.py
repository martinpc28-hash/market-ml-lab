"""El error más caro en ML financiero es usar información del futuro sin darse
cuenta (lookahead). Estos tests lo comprueban sin necesidad de red."""
import numpy as np
import pandas as pd

from marketml import allocator, forecaster
from marketml.features import MARKET_FEATURES, market_features


def fake_prices(n=700, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    cols = ["SPY", "^VIX", "^VIX3M", *[t for t in allocator.TICKERS if t != "SPY"]]
    rets = rng.normal(0.0003, 0.01, size=(n, len(cols)))
    px = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=idx, columns=cols)
    px["^VIX"] = 15 * np.exp(np.cumsum(rng.normal(0, 0.05, n)) * 0.2)
    px["^VIX3M"] = px["^VIX"] * 1.1
    return px


def test_market_features_do_not_look_ahead():
    px = fake_prices()
    cut = px.index[500]
    base = market_features(px).loc[:cut]
    shocked = px.copy()
    shocked.loc[shocked.index > cut] *= 3  # cambiar radicalmente el futuro
    after = market_features(shocked).loc[:cut]
    pd.testing.assert_frame_equal(base, after)


def test_asset_features_do_not_look_ahead():
    px = fake_prices()
    cut = px.index[550]
    base = allocator.build_dataset(px).loc[:cut, allocator.ASSET_FEATURES]
    shocked = px.copy()
    shocked.loc[shocked.index > cut] *= 0.5
    after = allocator.build_dataset(shocked).loc[:cut, allocator.ASSET_FEATURES]
    pd.testing.assert_frame_equal(base, after)


def test_targets_look_forward_by_horizon():
    px = fake_prices()
    X, Y = forecaster.build_dataset(px)
    t = X.index[100]
    i = px.index.get_loc(t)
    assert np.isclose(Y.loc[t, "spy_1w"], np.log(px["SPY"].iloc[i + 5] / px["SPY"].iloc[i]))
    assert np.isclose(Y.loc[t, "vix_1m"], np.log(px["^VIX"].iloc[i + 21] / px["^VIX"].iloc[i]))
    assert list(X.columns) == MARKET_FEATURES


def test_forecaster_outputs_ordered_intervals():
    px = fake_prices()
    X, Y = forecaster.build_dataset(px)
    params = {**forecaster.DEFAULT_PARAMS, "max_iter": 20, "min_samples_leaf": 20}
    models = {}
    for target in forecaster.TARGETS:
        ok = Y[target].notna()
        models[target] = {h: forecaster._model(h, params).fit(X[ok], Y[target][ok])
                          for h in forecaster.QUANTILE_HEADS}
    out = forecaster.MarketForecaster(models, {}).predict(None, X.tail(10))
    assert out.shape == (10, 18)
    for target in forecaster.TARGETS:
        assert (out[f"{target}_p10"] <= out[target]).all()
        assert (out[target] <= out[f"{target}_p90"]).all()


def _gate(ew=0.8, momentum=1.0, meta=0.8, ml=0.7, spread=0.05):
    return {"equal_weight_sharpe": ew, "momentum_sharpe": momentum, "meta_sharpe": meta,
            "ml_sharpe": ml, "ml_spread_annual": spread}


def test_gate_keeps_simple_rule_when_ml_is_worse():
    strategy, reason = allocator.choose_strategy(_gate())
    assert strategy == "momentum"
    assert "momentum" in reason


def test_gate_requires_margin_not_a_tie():
    # Meta-labeling apenas mejor que el momentum: no basta con empatar, debe superar el margen.
    strategy, _ = allocator.choose_strategy(_gate(meta=1.0 + allocator.MARGIN / 2))
    assert strategy == "momentum"


def test_gate_promotes_meta_when_clearly_better():
    strategy, _ = allocator.choose_strategy(_gate(meta=1.2))
    assert strategy == "meta"


def test_gate_rejects_ml_that_does_not_rank():
    # Aunque gane en Sharpe, si sus favoritos no baten a sus últimos, no está ordenando.
    strategy, _ = allocator.choose_strategy(_gate(ml=1.5, spread=-0.01))
    assert strategy == "momentum"


def test_gate_falls_back_to_equal_weight():
    strategy, _ = allocator.choose_strategy(_gate(ew=1.0, momentum=0.9, meta=0.9, ml=0.9))
    assert strategy == "equal_weight"
