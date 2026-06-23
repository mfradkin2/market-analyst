# market-analyst

A readable Python scaffold for combining US equity market structure with
publicly disclosed congressional trades into a ranked candidate list.

## What this is — and isn't

- **It is** a research tool that pulls free public data (yfinance for prices,
  house-stock-watcher for STOCK Act disclosures) and ranks tickers using
  a transparent score you can read top-to-bottom.
- **It is not** an edge. Congressional disclosures lag the actual trade by up
  to 45 days. Pelosi-tracking has been hyped but post-cost returns are mixed.
  Use this as a thematic overlay, not a money printer.
- **It is not** insider trading. Filings under the STOCK Act are public
  information once disclosed.

## Install

```bash
cd ~/market-analyst
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run (paper mode)

```bash
python main.py --simulate-buys 5
```

The first run writes `paper_book.json` in the working directory with $10k
starting cash. Re-run daily — the book persists.

## Layout

| file | purpose |
|---|---|
| `src/market_data.py` | yfinance pulls, correlations, beta, lead/lag |
| `src/congress_trades.py` | House + Senate disclosure feeds, aggregation |
| `src/news.py` | StockTwits / X / Finnhub mentions + VADER sentiment |
| `src/signals.py` | combined ranking (momentum + congress + social + low-vol) |
| `src/portfolio.py` | local paper-trading book, persisted to JSON |
| `src/robinhood_client.py` | optional live path via `robin-stocks` |
| `main.py` | daily orchestrator |

## Going live with Robinhood (read this twice)

Robinhood has **no official Python API**. The community library
`robin-stocks` reverse-engineers the mobile app's HTTP endpoints. Risks:

1. Robinhood can change endpoints, ban headless logins, or freeze your
   account at any time.
2. You hand the library your password + MFA seed. Treat your machine like a
   secret.
3. There is no support if something goes wrong with an order.

To enable:

```bash
pip install robin-stocks pyotp
export RH_USERNAME=...
export RH_PASSWORD=...
export RH_MFA_TOTP=...        # TOTP seed from authenticator setup
export RH_LIVE=1              # gate 1
# and pass confirm=True at the call site — gate 2
```

The `RobinhoodClient.submit` method has **two independent gates** before
sending an order. The `--live` flag in `main.py` is intentionally not wired
to the broker call — wire it yourself when you're sure.

## Suggested workflow

1. Run in paper mode daily for **at least a month**.
2. Compare the paper book's equity curve to a simple SPY buy-and-hold over
   the same period. If you can't beat SPY in paper, you won't beat it live.
3. Inspect each candidate's `rationale` before trusting the score.
4. Only then consider plumbing the live path — and start with tiny size.

## Data sources

- Prices: [yfinance](https://github.com/ranaroussi/yfinance) (free, rate-limited)
- Congressional trades: **pluggable**, see below. The historically-popular
  `house-stock-watcher` S3 bucket returns 403 now, so the module no longer
  hard-codes it.

### Wiring up congressional trades

Pick one:

**Option A — local JSON (recommended for tinkering):**
Download a recent trades dump from a source you trust (e.g.,
[capitoltrades.com](https://www.capitoltrades.com), a CSV export from
QuiverQuant's web UI, or a community GitHub mirror), normalize it to a list
of records with at least `ticker`, `transaction_date`, `type`, `amount`,
and politician name, and save as `trades.json`.

```bash
export CONGRESS_LOCAL_JSON=/path/to/trades.json
python main.py --simulate-buys 5
```

**Option B — Quiver Quantitative API:**
Register at [quiverquant.com](https://www.quiverquant.com) and grab a token.

```bash
export QUIVER_TOKEN=...
python main.py --simulate-buys 5
```

If neither is set, the analyst still runs — you just get a market-only score
with the congress component zeroed out.

### Wiring up news / social mentions

Sentiment + mention volume from three pluggable sources. Picked by env vars
(any combination is fine — they stack):

**StockTwits — free, default.**
No env var needed. Cashtag-organized retail chatter, self-tagged
bullish/bearish by users. Best free signal for stock-specific buzz.
Disable with `DISABLE_STOCKTWITS=1` if you don't want it.

> Note: StockTwits sits behind Cloudflare bot detection — plain `requests`
> gets a 403 challenge page. The module uses `curl_cffi` to mimic a real
> browser's TLS fingerprint, which gets through. `curl_cffi` is in
> requirements.txt; if you skip installing it, StockTwits silently returns
> empty.

**X (Twitter) — paid only. Be honest about this.**
X's free API tier does NOT include `/2/tweets/search/recent`. The cheapest
tier that does is **X Basic ≈ $200/mo, capped at 10k tweets/month**. If you
have a token:

```bash
export X_BEARER_TOKEN=...
python main.py --simulate-buys 5
```

Scraping X without API access (Nitter, headless browsers) is fragile, against
their TOS, and gets your IP/account flagged. Not implemented here.

**Finnhub — free with key, stock-tagged news headlines.**
Register at [finnhub.io](https://finnhub.io) for a free key (60 calls/min).

```bash
export FINNHUB_KEY=...
python main.py --simulate-buys 5
```

### How the social signal scores

For each ticker:
- `mention_volume` — count of messages in last 72h (tunable via `--social-hours`)
- `sentiment_avg` — mean VADER compound score in [-1, 1], with self-tagged
  StockTwits Bullish/Bearish overriding VADER when present

The ranking uses `z(volume) * sign(sentiment)` — meaning volume spikes only
help a name when the chatter is positive, and hurt when it's negative. Pure
noise (high volume, neutral sentiment) contributes zero. Skip the whole
thing with `--skip-social` if you're offline.
