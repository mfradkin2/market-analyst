"""
Congressional trade disclosures (STOCK Act, Periodic Transaction Reports).

This is fully public data — not insider information. Filings lag the actual
trade by up to 45 days, so treat this as a slow signal, not a fast one.

Data source — pluggable:
  1. LOCAL_PATH (env CONGRESS_LOCAL_JSON): point at a JSON file you maintain
     (download from quiverquant.com export, capitoltrades.com, etc.).
  2. QUIVER_TOKEN (env QUIVER_TOKEN): free-tier Quiver Quantitative API.
  3. (deprecated) house-stock-watcher S3 — was free, returns 403 as of 2025.

If none are configured, fetch_all() returns an empty DataFrame and the
analyst pipeline continues without the congress signal.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests


QUIVER_CONGRESS_URL = "https://api.quiverquant.com/beta/bulk/congresstrading"


@dataclass(frozen=True)
class Trade:
    politician: str
    chamber: str  # "house" | "senate"
    ticker: str
    transaction_type: str  # "purchase" | "sale" | "exchange" | ...
    transaction_date: date
    disclosure_date: date | None
    amount_range: str  # e.g. "$1,001 - $15,000"
    amount_mid: float  # midpoint of the disclosed range


_AMOUNT_BUCKETS = {
    "$1,001 - $15,000": 8_000,
    "$15,001 - $50,000": 32_500,
    "$50,001 - $100,000": 75_000,
    "$100,001 - $250,000": 175_000,
    "$250,001 - $500,000": 375_000,
    "$500,001 - $1,000,000": 750_000,
    "$1,000,001 - $5,000,000": 3_000_000,
    "$5,000,001 - $25,000,000": 15_000_000,
    "$25,000,001 - $50,000,000": 37_500_000,
    "$50,000,001 -": 50_000_000,
}


def _midpoint(label: str | None) -> float:
    if not label:
        return 0.0
    label = label.strip()
    return float(_AMOUNT_BUCKETS.get(label, 0.0))


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _row_to_trade(row: dict, chamber_hint: str | None = None) -> Trade | None:
    """Normalize one disclosure record from any source into a Trade."""
    ticker = (row.get("ticker") or row.get("Ticker") or "").upper().strip()
    if not ticker or ticker == "--":
        return None
    td = _parse_date(row.get("transaction_date") or row.get("TransactionDate") or row.get("Traded"))
    if td is None:
        return None
    politician = (
        row.get("representative")
        or row.get("senator")
        or row.get("Representative")
        or row.get("Senator")
        or row.get("Name")
        or ""
    ).strip()
    chamber = chamber_hint or (row.get("chamber") or "").lower() or (
        "senate" if "senator" in row or "Senator" in row else "house"
    )
    ttype = (row.get("type") or row.get("Transaction") or "").lower()
    amount_label = row.get("amount") or row.get("Range") or ""
    return Trade(
        politician=politician,
        chamber=chamber,
        ticker=ticker,
        transaction_type=ttype,
        transaction_date=td,
        disclosure_date=_parse_date(row.get("disclosure_date") or row.get("Filed")),
        amount_range=amount_label,
        amount_mid=_midpoint(amount_label),
    )


def fetch_local(path: str | Path) -> list[Trade]:
    """Read disclosures from a local JSON file (list of records)."""
    data = json.loads(Path(path).read_text())
    out: list[Trade] = []
    for row in data:
        t = _row_to_trade(row)
        if t:
            out.append(t)
    return out


def fetch_quiver(token: str, timeout: int = 60) -> list[Trade]:
    """Pull congress trades from Quiver Quantitative (requires API token)."""
    r = requests.get(
        QUIVER_CONGRESS_URL,
        headers={"Authorization": f"Token {token}", "Accept": "application/json"},
        timeout=timeout,
    )
    r.raise_for_status()
    return [t for row in r.json() if (t := _row_to_trade(row))]


def fetch_all() -> pd.DataFrame:
    """
    Combined congressional trades from whichever source is configured.

    Resolution order:
      1. CONGRESS_LOCAL_JSON env var → local file
      2. QUIVER_TOKEN env var → Quiver Quantitative API
      3. empty DataFrame (pipeline continues without the signal)
    """
    local = os.environ.get("CONGRESS_LOCAL_JSON")
    if local:
        trades = fetch_local(local)
    elif token := os.environ.get("QUIVER_TOKEN"):
        trades = fetch_quiver(token)
    else:
        trades = []
    return pd.DataFrame([t.__dict__ for t in trades])


def filter_trades(
    df: pd.DataFrame,
    politicians: list[str] | None = None,
    since: date | None = None,
    only_buys: bool = False,
) -> pd.DataFrame:
    out = df.copy()
    if politicians:
        wanted = {p.lower() for p in politicians}
        out = out[out["politician"].str.lower().isin(wanted)]
    if since:
        out = out[out["transaction_date"] >= since]
    if only_buys:
        out = out[out["transaction_type"].str.contains("purchase", na=False)]
    return out.sort_values("transaction_date", ascending=False)


def recent_signals(
    df: pd.DataFrame,
    days: int = 60,
    min_amount: float = 15_000,
) -> pd.DataFrame:
    """
    Aggregate recent trades into a per-ticker signal:
      - net_dollar = buys - sells (range midpoints)
      - n_distinct_politicians
      - latest_trade_date
    A ticker with multiple distinct politicians buying recently is a stronger
    signal than one congressperson repeatedly trading the same name.
    """
    cutoff = date.today() - timedelta(days=days)
    recent = df[df["transaction_date"] >= cutoff].copy()
    recent = recent[recent["amount_mid"] >= min_amount]
    if recent.empty:
        return pd.DataFrame(
            columns=["ticker", "net_dollar", "n_buyers", "n_sellers", "n_politicians", "latest"]
        )

    recent["signed_amount"] = recent.apply(
        lambda r: r["amount_mid"]
        if "purchase" in r["transaction_type"]
        else -r["amount_mid"]
        if "sale" in r["transaction_type"]
        else 0.0,
        axis=1,
    )
    grouped = recent.groupby("ticker").agg(
        net_dollar=("signed_amount", "sum"),
        n_buyers=(
            "transaction_type",
            lambda s: s.str.contains("purchase", na=False).sum(),
        ),
        n_sellers=(
            "transaction_type",
            lambda s: s.str.contains("sale", na=False).sum(),
        ),
        n_politicians=("politician", "nunique"),
        latest=("transaction_date", "max"),
    )
    return grouped.sort_values("net_dollar", ascending=False).reset_index()
