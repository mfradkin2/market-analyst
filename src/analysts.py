"""
Wall Street analyst aggregates.

Two sources, blended:
  1. yfinance (Yahoo) — mean price target + average rating (1=strong buy, 5=sell)
  2. Finnhub (FINNHUB_KEY env var, free tier) — price target consensus,
     recommendation trends (counts per category, last month), and recent
     upgrade/downgrade events.

We do NOT scrape Seeking Alpha / TipRanks / Zacks — paywalled + anti-bot.

Per-ticker derived signals (combined):
  - upside_pct: (mean_target - last_price) / last_price  [from Yahoo + Finnhub avg]
  - rating_score: ∈ [-1, +1], positive = buy
  - upgrade_net_30d: count of upgrades − downgrades in last 30 days (Finnhub)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable

import pandas as pd
import requests
import yfinance as yf


FINNHUB_TARGET_URL = "https://finnhub.io/api/v1/stock/price-target"
FINNHUB_REC_URL = "https://finnhub.io/api/v1/stock/recommendation"
FINNHUB_UPDOWN_URL = "https://finnhub.io/api/v1/stock/upgrade-downgrade"


log = logging.getLogger(__name__)


@dataclass
class AnalystView:
    ticker: str
    last_price: float | None
    mean_target: float | None
    n_opinions: int | None
    recommendation_mean: float | None  # 1 strong buy → 5 sell
    recommendation_key: str | None
    # Finnhub augmentation
    finnhub_target_mean: float | None = None
    finnhub_n_targets: int | None = None
    finnhub_strong_buy: int | None = None
    finnhub_buy: int | None = None
    finnhub_hold: int | None = None
    finnhub_sell: int | None = None
    finnhub_strong_sell: int | None = None
    upgrades_30d: int = 0
    downgrades_30d: int = 0

    @property
    def upside_pct(self) -> float | None:
        targets = [t for t in (self.mean_target, self.finnhub_target_mean) if t]
        if self.last_price and targets:
            blended = sum(targets) / len(targets)
            return (blended / self.last_price) - 1
        return None

    @property
    def rating_score(self) -> float | None:
        """
        Blended rating score in [-1, +1] (positive = buy).
        Combines yfinance recommendationMean and Finnhub category counts.
        """
        scores: list[float] = []
        if self.recommendation_mean is not None:
            scores.append((3.0 - self.recommendation_mean) / 2.0)
        # Finnhub: weighted score from buy/hold/sell counts
        if any(
            v is not None for v in (
                self.finnhub_strong_buy, self.finnhub_buy, self.finnhub_hold,
                self.finnhub_sell, self.finnhub_strong_sell,
            )
        ):
            sb = self.finnhub_strong_buy or 0
            b = self.finnhub_buy or 0
            h = self.finnhub_hold or 0
            s = self.finnhub_sell or 0
            ss = self.finnhub_strong_sell or 0
            total = sb + b + h + s + ss
            if total > 0:
                weighted = (sb * 1.0 + b * 0.5 + h * 0.0 + s * -0.5 + ss * -1.0) / total
                scores.append(weighted)
        if not scores:
            return None
        return sum(scores) / len(scores)

    @property
    def upgrade_net_30d(self) -> int:
        return self.upgrades_30d - self.downgrades_30d


def _safe_get(info: dict, *keys) -> float | None:
    for k in keys:
        v = info.get(k)
        if v is not None and v != "" and not (isinstance(v, float) and (v != v)):  # NaN check
            try:
                return float(v)
            except (TypeError, ValueError):
                return None
    return None


def _finnhub_get(url: str, params: dict, timeout: int = 15) -> dict | list | None:
    key = os.environ.get("FINNHUB_KEY")
    if not key:
        return None
    try:
        r = requests.get(url, params={**params, "token": key}, timeout=timeout)
        if r.status_code == 429:
            log.warning("finnhub rate-limited on %s", url)
            return None
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.warning("finnhub request failed (%s): %s", url, e)
        return None


def _fetch_finnhub_for(view: AnalystView) -> None:
    """Mutate `view` in-place with Finnhub fields (price target, rec trends, upgrades)."""
    sym = view.ticker

    target = _finnhub_get(FINNHUB_TARGET_URL, {"symbol": sym})
    if isinstance(target, dict):
        view.finnhub_target_mean = _safe_get(target, "targetMean")
        view.finnhub_n_targets = int(target.get("numberOfAnalysts") or 0) or None

    rec = _finnhub_get(FINNHUB_REC_URL, {"symbol": sym})
    if isinstance(rec, list) and rec:
        latest = rec[0]  # most recent month first
        view.finnhub_strong_buy = latest.get("strongBuy")
        view.finnhub_buy = latest.get("buy")
        view.finnhub_hold = latest.get("hold")
        view.finnhub_sell = latest.get("sell")
        view.finnhub_strong_sell = latest.get("strongSell")

    cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    updown = _finnhub_get(FINNHUB_UPDOWN_URL, {"symbol": sym, "from": cutoff, "to": today})
    if isinstance(updown, list):
        ups = 0
        downs = 0
        for ev in updown:
            action = (ev.get("action") or "").lower()
            if action in ("up", "upgrade"):
                ups += 1
            elif action in ("down", "downgrade"):
                downs += 1
        view.upgrades_30d = ups
        view.downgrades_30d = downs


def fetch_one(ticker: str, with_finnhub: bool = True) -> AnalystView:
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
    except Exception as e:
        log.warning("yfinance .info failed for %s: %s", ticker, e)
        info = {}

    view = AnalystView(
        ticker=ticker,
        last_price=_safe_get(info, "currentPrice", "regularMarketPrice", "previousClose"),
        mean_target=_safe_get(info, "targetMeanPrice"),
        n_opinions=int(info["numberOfAnalystOpinions"]) if info.get("numberOfAnalystOpinions") else None,
        recommendation_mean=_safe_get(info, "recommendationMean"),
        recommendation_key=info.get("recommendationKey"),
    )
    if with_finnhub:
        _fetch_finnhub_for(view)
    return view


def fetch_many(tickers: Iterable[str], with_finnhub: bool = True) -> pd.DataFrame:
    """Returns one row per ticker with analyst aggregates and derived signals."""
    rows = []
    for t in tickers:
        v = fetch_one(t, with_finnhub=with_finnhub)
        rows.append(
            {
                "ticker": v.ticker,
                "last_price": v.last_price,
                "mean_target": v.mean_target,
                "finnhub_target_mean": v.finnhub_target_mean,
                "n_opinions": v.n_opinions,
                "recommendation_mean": v.recommendation_mean,
                "recommendation_key": v.recommendation_key,
                "upgrades_30d": v.upgrades_30d,
                "downgrades_30d": v.downgrades_30d,
                "upgrade_net_30d": v.upgrade_net_30d,
                "upside_pct": v.upside_pct,
                "rating_score": v.rating_score,
            }
        )
    return pd.DataFrame(rows).set_index("ticker")
