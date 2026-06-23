"""
US equity market data + cross-stock relationships.

Pulls historical OHLCV from yfinance and computes the relationships an analyst
actually uses: returns, rolling correlations, beta vs SPY, lead/lag (Granger-lite),
and sector co-movement.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf


SPY = "SPY"


@dataclass
class PriceHistory:
    """Adjusted-close panel indexed by date, columns = tickers."""

    prices: pd.DataFrame

    @property
    def returns(self) -> pd.DataFrame:
        return self.prices.pct_change().dropna(how="all")

    @property
    def log_returns(self) -> pd.DataFrame:
        return np.log(self.prices / self.prices.shift(1)).dropna(how="all")


def fetch_prices(
    tickers: Iterable[str],
    lookback_days: int = 365,
    end: date | None = None,
) -> PriceHistory:
    end = end or date.today()
    start = end - timedelta(days=lookback_days)
    tickers = sorted(set(t.upper() for t in tickers))
    raw = yf.download(
        tickers,
        start=start.isoformat(),
        end=end.isoformat(),
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})
    return PriceHistory(prices=prices.dropna(how="all"))


def correlation_matrix(history: PriceHistory, window: int | None = None) -> pd.DataFrame:
    """Full-window or trailing-window return correlations."""
    rets = history.returns
    if window is None:
        return rets.corr()
    return rets.tail(window).corr()


def rolling_correlation(history: PriceHistory, a: str, b: str, window: int = 30) -> pd.Series:
    """Trailing correlation between two tickers — useful for spotting regime shifts."""
    rets = history.returns[[a, b]].dropna()
    return rets[a].rolling(window).corr(rets[b])


def beta_vs_market(history: PriceHistory, ticker: str, market: str = SPY) -> float:
    """OLS beta of ticker returns vs market returns."""
    rets = history.returns[[ticker, market]].dropna()
    cov = rets.cov().iloc[0, 1]
    var_m = rets[market].var()
    return float(cov / var_m) if var_m else float("nan")


def lead_lag_correlation(
    history: PriceHistory, leader: str, follower: str, max_lag: int = 5
) -> pd.Series:
    """
    Correlate leader_returns(t-k) with follower_returns(t) for k in [0..max_lag].
    A peak at k>0 suggests `leader` moves before `follower` — useful for finding
    supplier->customer or sector-leader relationships.
    """
    rets = history.returns[[leader, follower]].dropna()
    out = {}
    for k in range(0, max_lag + 1):
        out[k] = rets[leader].shift(k).corr(rets[follower])
    return pd.Series(out, name=f"{leader}->{follower}")


def most_correlated(history: PriceHistory, ticker: str, top_n: int = 10) -> pd.Series:
    """Top-N tickers in the panel most correlated with `ticker` (excluding self)."""
    corr = history.returns.corr()[ticker].drop(labels=[ticker])
    return corr.abs().sort_values(ascending=False).head(top_n)


def annualized_vol(history: PriceHistory, ticker: str) -> float:
    rets = history.returns[ticker].dropna()
    return float(rets.std() * np.sqrt(252))


def cluster_momentum(
    history: PriceHistory, ticker: str, top_k: int = 5, mom_window: int = 20
) -> float:
    """
    Weighted-average momentum of `ticker`'s top-K most correlated peers.

    A ticker whose correlation cluster is strongly trending up will continue
    to be supported by the cluster; a ticker whose peers are rolling over is
    a higher-risk hold even if its own chart looks fine.

    Weights = correlation^2 (heavier weight on tighter co-movers).
    """
    if ticker not in history.prices.columns:
        return 0.0
    rets = history.returns
    if ticker not in rets:
        return 0.0
    corrs = rets.corr()[ticker].drop(labels=[ticker]).dropna()
    if corrs.empty:
        return 0.0
    top = corrs.abs().sort_values(ascending=False).head(top_k).index
    weights = (corrs.loc[top] ** 2)
    if weights.sum() == 0:
        return 0.0
    p = history.prices
    valid = [t for t in top if t in p.columns and len(p[t].dropna()) > mom_window]
    if not valid:
        return 0.0
    peer_mom = pd.Series(
        {t: float(p[t].iloc[-1] / p[t].iloc[-mom_window - 1] - 1) for t in valid}
    )
    w = weights.reindex(valid).fillna(0.0)
    if w.sum() == 0:
        return 0.0
    return float((peer_mom * w).sum() / w.sum())


def summary_stats(history: PriceHistory) -> pd.DataFrame:
    """One-row-per-ticker summary an analyst would glance at first."""
    rets = history.returns
    out = pd.DataFrame(
        {
            "last_price": history.prices.iloc[-1],
            "ytd_return": history.prices.iloc[-1] / history.prices.iloc[0] - 1,
            "ann_vol": rets.std() * np.sqrt(252),
            "sharpe_naive": (rets.mean() / rets.std()) * np.sqrt(252),
        }
    )
    if SPY in rets.columns:
        out["beta_vs_spy"] = [beta_vs_market(history, t) if t != SPY else 1.0 for t in out.index]
    return out.sort_values("sharpe_naive", ascending=False)
