"""Signal-ranking tests: risk-adjusted momentum blend + weight renormalization."""

import numpy as np
import pandas as pd
import pytest

from src.market_data import PriceHistory
from src.signals import blended_risk_adjusted_momentum, momentum_window, rank_candidates


def make_history(n_days: int = 70) -> PriceHistory:
    """Deterministic panel: STDY grinds up quietly, SPIK gets the same total
    return with violent chop, SPY drifts flat, DOWN bleeds."""
    idx = pd.bdate_range("2026-03-02", periods=n_days)
    steady = 100 * (1.002 ** np.arange(n_days))
    # Alternate +3% / -2.5239% so each pair nets the same as two +0.2% days
    pair = [1.03, (1.002 ** 2) / 1.03]
    spike = 100 * np.cumprod([pair[i % 2] for i in range(n_days)])
    spy = 100 * (1.0002 ** np.arange(n_days))
    down = 100 * (0.998 ** np.arange(n_days))
    return PriceHistory(
        prices=pd.DataFrame({"STDY": steady, "SPIK": spike, "SPY": spy, "DOWN": down}, index=idx)
    )


EMPTY_CONGRESS = pd.DataFrame(
    columns=["ticker", "net_dollar", "n_buyers", "n_sellers", "n_politicians", "latest"]
)


def test_momentum_window_too_short_returns_none():
    h = make_history(30)
    assert momentum_window(h, 63) is None
    assert momentum_window(h, 20) is not None


def test_risk_adjusted_momentum_prefers_low_vol_path():
    h = make_history(70)
    vol = pd.Series(
        {t: float(h.returns[t].std() * np.sqrt(252)) for t in h.prices.columns}
    )
    ram = blended_risk_adjusted_momentum(h, vol)
    # Same-ish total return, but the quiet compounder must dominate the chopper
    assert ram["STDY"] > ram["SPIK"] * 2


def test_rank_prefers_steady_over_spiky():
    h = make_history(70)
    ranked = rank_candidates(h, EMPTY_CONGRESS)
    order = [c.ticker for c in ranked]
    assert order.index("STDY") < order.index("SPIK")
    assert order.index("STDY") < order.index("DOWN")


def test_rank_excludes_spy():
    h = make_history(70)
    ranked = rank_candidates(h, EMPTY_CONGRESS)
    assert "SPY" not in [c.ticker for c in ranked]


def test_weight_renormalization_with_dead_signals():
    # congress/social/analyst all zero -> active weights renormalize to 1.0,
    # so scores keep full magnitude instead of being deflated ~2x.
    h = make_history(70)
    ranked = rank_candidates(h, EMPTY_CONGRESS)
    top = ranked[0]
    # With deflated weights (bug), the top z-combo rarely clears 0.4.
    assert top.score > 0.4


def test_short_history_falls_back_to_20d():
    h = make_history(40)  # too short for the 63d leg
    ranked = rank_candidates(h, EMPTY_CONGRESS)
    assert [c.ticker for c in ranked]  # no crash, produces a ranking


def test_too_short_for_any_momentum_raises():
    h = make_history(15)
    vol = pd.Series({t: 0.2 for t in h.prices.columns})
    with pytest.raises(ValueError):
        blended_risk_adjusted_momentum(h, vol)
