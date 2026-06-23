"""
Daily analyst loop.

Pipeline:
  1. Pull recent congressional disclosures, aggregate the last ~60 days.
  2. Build a universe = (default megacaps) ∪ (tickers congress has touched).
  3. Pull price history for the universe.
  4. Compute summary stats + correlation map.
  5. Rank candidates by combined market + congress score.
  6. Sketch trades into the paper book (or, if RH_LIVE + --live, the live book).

This is intentionally readable, not "production trading infra". Use it as a
research scaffold, watch how the picks behave in paper for at least a few
weeks before considering live execution.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from src import analysts, congress_trades, executor, market_data, news, signals
from src.portfolio import Order, PaperBook


DEFAULT_UNIVERSE = [
    "SPY", "QQQ", "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META",
    "TSLA", "AMD", "AVGO", "JPM", "XOM", "UNH", "LLY", "V", "MA",
    "HD", "COST", "WMT", "PG", "BAC", "CRM", "ORCL", "ADBE", "NFLX",
]

# Politicians whose disclosures get extra weight. Edit freely.
WATCHLIST = [
    "Nancy Pelosi", "Paul Pelosi",
    "Dan Crenshaw", "Tommy Tuberville", "Ro Khanna",
]


def build_universe(congress_df: pd.DataFrame, base: list[str] = DEFAULT_UNIVERSE) -> list[str]:
    cutoff = date.today() - timedelta(days=90)
    recent_tickers = (
        congress_df[congress_df["transaction_date"] >= cutoff]["ticker"].dropna().unique().tolist()
    )
    universe = sorted(set(base) | {t for t in recent_tickers if t.isalpha() and len(t) <= 5})
    return universe


def run(args: argparse.Namespace) -> None:
    log = logging.getLogger("analyst")
    log.info("fetching congressional disclosures...")
    try:
        congress_df = congress_trades.fetch_all()
    except Exception as e:
        log.warning("congress fetch failed (%s) — continuing with empty signal", e)
        congress_df = pd.DataFrame(columns=["ticker", "transaction_date", "amount_mid", "transaction_type", "politician"])

    if not congress_df.empty:
        watchlist_df = congress_trades.filter_trades(congress_df, politicians=WATCHLIST)
        log.info("watchlist trades in dataset: %d", len(watchlist_df))
        cs = congress_trades.recent_signals(congress_df, days=args.congress_days)
    else:
        cs = pd.DataFrame(columns=["ticker", "net_dollar", "n_buyers", "n_sellers", "n_politicians", "latest"])

    universe = build_universe(congress_df) if not congress_df.empty else DEFAULT_UNIVERSE
    log.info("universe size: %d", len(universe))

    log.info("fetching prices...")
    history = market_data.fetch_prices(universe, lookback_days=args.lookback_days)
    log.info("price panel: %s", history.prices.shape)

    log.info("market summary (top by sharpe):")
    print(market_data.summary_stats(history).head(10).round(3))

    if args.skip_social:
        log.info("social signal disabled by --skip-social")
        social_df = pd.DataFrame()
    else:
        log.info("fetching social/news mentions...")
        try:
            mentions = news.fetch_mentions(universe)
            social_df = news.aggregate(mentions, lookback_hours=args.social_hours)
            log.info("social aggregated: %d tickers with mentions", len(social_df))
        except Exception as e:
            log.warning("social fetch failed (%s) — continuing without it", e)
            social_df = pd.DataFrame()

    if args.skip_analysts:
        log.info("analyst fetch disabled by --skip-analysts")
        analyst_df = pd.DataFrame()
    else:
        log.info("fetching analyst data (yfinance)...")
        try:
            analyst_df = analysts.fetch_many(universe)
            log.info("analyst rows: %d", len(analyst_df))
        except Exception as e:
            log.warning("analyst fetch failed (%s) — continuing without it", e)
            analyst_df = pd.DataFrame()

    log.info("ranking candidates...")
    ranked = signals.rank_candidates(
        history, cs, social_signals=social_df,
        analyst_data=analyst_df if not analyst_df.empty else None,
        universe=universe,
    )
    print("\nTop candidates:")
    for c in ranked[:10]:
        print(
            f"  {c.ticker:6s} score={c.score:+.2f} | mom20={c.momentum_20d:+.1%}"
            f" vol={c.vol_ann:.0%} beta={c.beta:.2f} | {c.rationale}"
        )

    book = PaperBook(Path(args.book), starting_cash=args.starting_cash)
    prices_now = {t: float(history.prices[t].iloc[-1]) for t in history.prices.columns}
    log.info("current equity: $%.2f (cash $%.2f)", book.equity(prices_now), book.cash)

    if args.simulate_buys:
        budget_each = (book.equity(prices_now) * 0.10)  # 10% per pick, naive
        for c in ranked[: args.simulate_buys]:
            if c.score <= 0:
                continue
            px = prices_now.get(c.ticker)
            if not px:
                continue
            qty = int(budget_each // px)
            if qty <= 0:
                continue
            if c.ticker in book.positions:
                continue
            try:
                book.submit(Order(c.ticker, "buy", qty, px, note=f"signal score {c.score:.2f}"))
                log.info("paper-bought %d %s @ $%.2f", qty, c.ticker, px)
            except ValueError as e:
                log.warning("skip %s: %s", c.ticker, e)

    snap = book.snapshot(prices_now)
    print("\nPortfolio snapshot:")
    print(f"  cash: ${snap['cash']:,.2f}   equity: ${snap['equity']:,.2f}")
    for row in snap["positions"]:
        print(
            f"  {row['ticker']:6s} qty={row['qty']:.0f} avg=${row['avg_cost']:.2f}"
            f" last=${row['last']:.2f} upnl=${row['upnl']:+,.2f}"
        )

    if args.plan:
        positions_payload: dict = {}
        if args.live_positions:
            raw = json.loads(Path(args.live_positions).read_text())
            # Accept both old shape (ticker -> float) and new shape (ticker -> dict)
            for t, v in raw.items():
                if isinstance(v, dict):
                    positions_payload[t] = v
                else:
                    positions_payload[t] = {"qty": 0.0, "avg_cost": 0.0, "mkt_value": float(v)}

        guardrails = executor.Guardrails(
            whitelist=DEFAULT_UNIVERSE + list(history.prices.columns),
        )
        quotes_payload = {
            t: {
                "last": float(history.prices[t].iloc[-1]),
                "ask": float(history.prices[t].iloc[-1]),
                "bid": float(history.prices[t].iloc[-1]),
            }
            for t in history.prices.columns
        }
        plan = executor.plan_trades(
            ranked=ranked,
            account_number=args.account,
            account_equity=args.account_equity,
            cash_available=args.cash_available,
            quotes=quotes_payload,
            positions=positions_payload,
            intraday_pnl_pct=args.intraday_pnl_pct,
            guardrails=guardrails,
        )
        out_path = Path(args.plan_out)
        plan.write(out_path)
        log.info("wrote trade plan to %s (%d orders, %d skipped)", out_path, len(plan.orders), len(plan.skipped))
        print(f"\nTrade plan ({len(plan.orders)} orders):")
        for o in plan.orders:
            tag = "SELL" if o.side == "sell" else "BUY "
            print(
                f"  {tag} {o.symbol:6s} ~${o.dollar_amount:>7.2f}"
                f"  {o.type:6s} ~{o.quantity_estimate} sh  | {o.rationale}"
            )
        if plan.circuit_breaker_tripped:
            print("\n*** CIRCUIT BREAKER TRIPPED — no orders will be placed ***")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", default="paper_book.json", help="path to paper book JSON")
    ap.add_argument("--starting-cash", type=float, default=10_000.0)
    ap.add_argument("--lookback-days", type=int, default=365)
    ap.add_argument("--congress-days", type=int, default=60)
    ap.add_argument("--simulate-buys", type=int, default=0, help="paper-buy top N picks")
    ap.add_argument("--skip-social", action="store_true", help="skip news/social fetch (useful offline)")
    ap.add_argument("--skip-analysts", action="store_true", help="skip yfinance analyst fetch")
    ap.add_argument("--social-hours", type=int, default=72, help="lookback window for social mentions")
    ap.add_argument("--live", action="store_true", help="reserved for live Robinhood path (not wired in main)")
    ap.add_argument("--plan", action="store_true", help="emit a trade_plan.json from the ranked candidates")
    ap.add_argument("--plan-out", default="trade_plan.json")
    ap.add_argument(
        "--account",
        default=os.environ.get("RH_ACCOUNT", ""),
        help="Robinhood account number (or set RH_ACCOUNT env var)",
    )
    ap.add_argument("--account-equity", type=float, default=997.52, help="current account equity in USD")
    ap.add_argument("--cash-available", type=float, default=0.0, help="buying power in USD")
    ap.add_argument("--intraday-pnl-pct", type=float, default=0.0, help="today's PnL as a decimal, e.g. -0.012 for -1.2%%")
    ap.add_argument("--live-positions", default="", help="path to JSON {ticker: market_value} of current holdings")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
