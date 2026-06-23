"""
News + social sentiment for the analyst pipeline.

Three pluggable backends, picked by env vars (cheapest configured wins):

  1. StockTwits  — free, no auth, cashtag-organized. Default.
  2. X (Twitter) — paid (Basic tier minimum). Set X_BEARER_TOKEN.
  3. Finnhub     — free with key, stock-tagged news headlines. Set FINNHUB_KEY.

For each ticker we surface two numbers an analyst would actually use:
  - mention_volume: how many recent messages mention it (volume is often
    a stronger signal than raw direction — surprise drives volume).
  - sentiment_avg:  mean sentiment in [-1, 1].

We do NOT pretend free-tier X access exists for tweet search — as of 2026
the cheapest tier that allows /2/tweets/search/recent is X Basic (~$200/mo).
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable

import pandas as pd
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

try:
    # curl_cffi mimics a real browser's TLS fingerprint, which is required to
    # get past Cloudflare's bot challenge on the StockTwits API.
    from curl_cffi import requests as cffi_requests  # type: ignore
    _HAS_CFFI = True
except ImportError:
    cffi_requests = None
    _HAS_CFFI = False


log = logging.getLogger(__name__)

STOCKTWITS_URL = "https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
X_SEARCH_URL = "https://api.x.com/2/tweets/search/recent"
FINNHUB_NEWS_URL = "https://finnhub.io/api/v1/company-news"

_vader = SentimentIntensityAnalyzer()


@dataclass
class Mention:
    ticker: str
    source: str  # "stocktwits" | "x" | "finnhub"
    text: str
    created_at: datetime
    sentiment: float  # [-1, 1]


def _score_text(text: str) -> float:
    if not text:
        return 0.0
    return float(_vader.polarity_scores(text)["compound"])


def fetch_stocktwits(ticker: str, timeout: int = 15) -> list[Mention]:
    """
    StockTwits stream for one cashtag. ~30 most recent messages.

    StockTwits sits behind Cloudflare bot detection — plain `requests` gets
    a 403 challenge page. We use curl_cffi when available to mimic a real
    browser TLS handshake. Without it, this endpoint won't work.
    """
    url = STOCKTWITS_URL.format(ticker=ticker.upper())
    if _HAS_CFFI:
        r = cffi_requests.get(url, impersonate="chrome120", timeout=timeout)
    else:
        log.warning("curl_cffi not installed; StockTwits will likely 403")
        r = requests.get(url, headers={"User-Agent": "market-analyst/0.1"}, timeout=timeout)
    if r.status_code == 429:
        log.warning("stocktwits rate-limited on %s", ticker)
        return []
    if r.status_code in (403, 404):
        return []  # blocked or unknown symbol
    r.raise_for_status()
    payload = r.json()
    out: list[Mention] = []
    for msg in payload.get("messages", []):
        body = msg.get("body") or ""
        st_label = (
            (msg.get("entities") or {}).get("sentiment") or {}
        ).get("basic")
        # Prefer self-tagged sentiment when present (Bullish/Bearish);
        # fall back to VADER on the text.
        if st_label == "Bullish":
            score = 0.6
        elif st_label == "Bearish":
            score = -0.6
        else:
            score = _score_text(body)
        try:
            created = datetime.fromisoformat(msg["created_at"].replace("Z", "+00:00"))
        except (KeyError, ValueError):
            created = datetime.now(timezone.utc)
        out.append(
            Mention(
                ticker=ticker.upper(),
                source="stocktwits",
                text=body,
                created_at=created,
                sentiment=score,
            )
        )
    return out


def fetch_x(ticker: str, bearer: str, max_results: int = 30, timeout: int = 20) -> list[Mention]:
    """
    X (Twitter) recent search for $TICKER mentions.
    Requires X API v2 Basic+ access; free tier won't work for search reads.
    """
    params = {
        "query": f"${ticker.upper()} -is:retweet lang:en",
        "max_results": min(max(max_results, 10), 100),
        "tweet.fields": "created_at,public_metrics,lang",
    }
    r = requests.get(
        X_SEARCH_URL,
        headers={"Authorization": f"Bearer {bearer}"},
        params=params,
        timeout=timeout,
    )
    if r.status_code in (401, 403):
        log.error("X API auth failed (%s) — check X_BEARER_TOKEN and tier", r.status_code)
        return []
    if r.status_code == 429:
        log.warning("X rate-limited on %s", ticker)
        return []
    r.raise_for_status()
    out: list[Mention] = []
    for t in r.json().get("data", []):
        body = t.get("text") or ""
        try:
            created = datetime.fromisoformat(t["created_at"].replace("Z", "+00:00"))
        except (KeyError, ValueError):
            created = datetime.now(timezone.utc)
        out.append(
            Mention(
                ticker=ticker.upper(),
                source="x",
                text=body,
                created_at=created,
                sentiment=_score_text(body),
            )
        )
    return out


def fetch_finnhub(ticker: str, key: str, days: int = 5, timeout: int = 20) -> list[Mention]:
    """Finnhub stock-tagged company news. Free tier, generous rate limits."""
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=days)
    r = requests.get(
        FINNHUB_NEWS_URL,
        params={
            "symbol": ticker.upper(),
            "from": start.isoformat(),
            "to": end.isoformat(),
            "token": key,
        },
        timeout=timeout,
    )
    if r.status_code == 429:
        log.warning("finnhub rate-limited on %s", ticker)
        return []
    r.raise_for_status()
    out: list[Mention] = []
    for item in r.json():
        headline = item.get("headline") or ""
        summary = item.get("summary") or ""
        body = f"{headline}. {summary}".strip()
        try:
            created = datetime.fromtimestamp(item["datetime"], tz=timezone.utc)
        except (KeyError, ValueError, OSError):
            created = datetime.now(timezone.utc)
        out.append(
            Mention(
                ticker=ticker.upper(),
                source="finnhub",
                text=body,
                created_at=created,
                sentiment=_score_text(body),
            )
        )
    return out


def fetch_mentions(tickers: Iterable[str], polite_delay: float = 0.2) -> list[Mention]:
    """
    Pull mentions for each ticker from whichever backends are configured.
    StockTwits is always used unless DISABLE_STOCKTWITS=1.

    Order of preference for sentiment: explicit (StockTwits Bull/Bear) >
    Finnhub headlines > VADER on X text.
    """
    x_token = os.environ.get("X_BEARER_TOKEN")
    finnhub_key = os.environ.get("FINNHUB_KEY")
    use_stocktwits = os.environ.get("DISABLE_STOCKTWITS") != "1"

    out: list[Mention] = []
    for t in tickers:
        if use_stocktwits:
            try:
                out.extend(fetch_stocktwits(t))
            except Exception as e:
                log.warning("stocktwits failed for %s: %s", t, e)
            time.sleep(polite_delay)
        if x_token:
            try:
                out.extend(fetch_x(t, x_token))
            except Exception as e:
                log.warning("x failed for %s: %s", t, e)
            time.sleep(polite_delay)
        if finnhub_key:
            try:
                out.extend(fetch_finnhub(t, finnhub_key))
            except Exception as e:
                log.warning("finnhub failed for %s: %s", t, e)
            time.sleep(polite_delay)
    return out


def aggregate(mentions: list[Mention], lookback_hours: int = 72) -> pd.DataFrame:
    """
    Per-ticker rollup:
      - mention_volume: count of mentions in lookback window
      - sentiment_avg:  mean compound sentiment
      - sentiment_net:  bullish_count - bearish_count (using a ±0.2 threshold)
      - sources:        comma-joined source list
      - latest:         most recent mention timestamp
    """
    if not mentions:
        return pd.DataFrame(
            columns=["ticker", "mention_volume", "sentiment_avg", "sentiment_net", "sources", "latest"]
        )
    cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    df = pd.DataFrame([m.__dict__ for m in mentions])
    df = df[df["created_at"] >= cutoff]
    if df.empty:
        return pd.DataFrame(
            columns=["ticker", "mention_volume", "sentiment_avg", "sentiment_net", "sources", "latest"]
        )

    def _net(s: pd.Series) -> int:
        return int((s > 0.2).sum() - (s < -0.2).sum())

    g = df.groupby("ticker").agg(
        mention_volume=("text", "count"),
        sentiment_avg=("sentiment", "mean"),
        sentiment_net=("sentiment", _net),
        sources=("source", lambda s: ",".join(sorted(set(s)))),
        latest=("created_at", "max"),
    ).reset_index()
    return g.sort_values("mention_volume", ascending=False)
