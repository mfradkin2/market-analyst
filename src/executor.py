"""
Turn ranked candidates + current portfolio into a TradePlan with guardrails.

This module is intentionally MCP-unaware. It writes a JSON trade plan that
the Claude session (or scheduled remote agent) then reads and executes via
the robinhood-trading MCP tools. Decoupling means:
  - Python is fully testable without any broker connection
  - Plans are inspectable, reviewable, and diff-able run-over-run
  - Execution is always a separate, human/agent-confirmed step
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .signals import Candidate


log = logging.getLogger(__name__)


@dataclass
class Guardrails:
    max_position_pct: float = 0.18
    max_total_positions: int = 8
    min_trade_dollars: float = 20.0
    min_candidate_score: float = 0.0
    daily_loss_circuit_breaker_pct: float = -0.03
    limit_price_buffer: float = 0.001  # marketable limit = ask * (1 + buffer)
    whitelist: list[str] = field(default_factory=list)
    allow_sells: bool = True
    # Sell-side rules
    exit_score_threshold: float = -0.25       # full sell if score drops below this
    stop_loss_pct: float = -0.08              # full sell if unrealized PnL <= this
    profit_take_pct: float = 0.25             # trim 50% on profit-takes
    profit_take_weakening_score: float = 0.10 # only trim if score has weakened to <= this
    overcap_trim_buffer: float = 0.005        # trim back to (cap - 0.5%) so we don't toggle
    # Cash-account T+1: same-session sell proceeds aren't usable for buys. When True,
    # the BUY pass only uses pre-existing buying_power (cash_available), ignoring
    # cash freed by sells in this run. Set False for margin accounts.
    cash_account_t_plus_1: bool = True


@dataclass
class PlannedOrder:
    symbol: str
    side: str  # "buy" | "sell"
    dollar_amount: float
    limit_price: float
    quantity_estimate: float
    type: str  # "limit" or "market" (fractional sells require market)
    time_in_force: str
    rationale: str
    quantity: float | None = None  # exact share count, for full-position sells


@dataclass
class TradePlan:
    generated_at: str
    account_number: str
    account_equity: float
    cash_available: float
    intraday_pnl_pct: float
    circuit_breaker_tripped: bool
    orders: list[PlannedOrder]
    skipped: list[dict]  # ticker + reason

    def to_json(self) -> str:
        return json.dumps(
            {
                "generated_at": self.generated_at,
                "account_number": self.account_number,
                "account_equity": self.account_equity,
                "cash_available": self.cash_available,
                "intraday_pnl_pct": self.intraday_pnl_pct,
                "circuit_breaker_tripped": self.circuit_breaker_tripped,
                "orders": [asdict(o) for o in self.orders],
                "skipped": self.skipped,
            },
            indent=2,
        )

    def write(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())


def _round_price(p: float) -> float:
    """Robinhood requires <= 2 decimals for prices >= $1, 4 below."""
    return round(p, 2) if p >= 1 else round(p, 4)


def plan_trades(
    ranked: list[Candidate],
    *,
    account_number: str,
    account_equity: float,
    cash_available: float,
    quotes: dict[str, dict[str, float]],
    positions: dict[str, dict[str, float]],  # ticker -> {qty, avg_cost, mkt_value}
    intraday_pnl_pct: float,
    guardrails: Guardrails,
) -> TradePlan:
    """
    positions[ticker] = {"qty": float, "avg_cost": float, "mkt_value": float}
    quotes[ticker]    = {"last": float, "ask": float, "bid": float}
    """
    now = datetime.now(timezone.utc).isoformat()
    orders: list[PlannedOrder] = []
    skipped: list[dict] = []

    if intraday_pnl_pct <= guardrails.daily_loss_circuit_breaker_pct:
        log.warning(
            "circuit breaker tripped: intraday PnL %.2f%% <= %.2f%%",
            intraday_pnl_pct * 100, guardrails.daily_loss_circuit_breaker_pct * 100,
        )
        return TradePlan(
            generated_at=now,
            account_number=account_number,
            account_equity=account_equity,
            cash_available=cash_available,
            intraday_pnl_pct=intraday_pnl_pct,
            circuit_breaker_tripped=True,
            orders=[],
            skipped=[{"reason": "circuit_breaker", "intraday_pnl_pct": intraday_pnl_pct}],
        )

    score_by_ticker = {c.ticker: c.score for c in ranked}
    max_dollars_per_position = account_equity * guardrails.max_position_pct
    cap_target_after_trim = max_dollars_per_position * (1 - guardrails.overcap_trim_buffer)

    # ---- SELL pass ----
    cash_freed = 0.0
    if guardrails.allow_sells:
        for ticker, pos in positions.items():
            mkt_value = pos.get("mkt_value", 0.0)
            qty = pos.get("qty", 0.0)
            avg_cost = pos.get("avg_cost", 0.0)
            if qty <= 0 or mkt_value <= 0:
                continue
            q = quotes.get(ticker, {})
            bid = q.get("bid") or q.get("last")
            last = q.get("last") or bid
            if not last or last <= 0:
                continue

            unrealized_pct = (last - avg_cost) / avg_cost if avg_cost else 0.0
            score = score_by_ticker.get(ticker)
            sell_reason: str | None = None
            sell_fraction: float = 0.0  # 1.0 = full liquidation

            # 1. Signal exit — highest priority
            if score is not None and score <= guardrails.exit_score_threshold:
                sell_reason = f"signal_exit (score={score:+.2f})"
                sell_fraction = 1.0
            # 2. Stop-loss
            elif unrealized_pct <= guardrails.stop_loss_pct:
                sell_reason = f"stop_loss ({unrealized_pct:+.1%})"
                sell_fraction = 1.0
            # 3. Profit take on weakening signal
            elif (
                unrealized_pct >= guardrails.profit_take_pct
                and score is not None
                and score <= guardrails.profit_take_weakening_score
            ):
                sell_reason = f"profit_take ({unrealized_pct:+.1%}, score weakened)"
                sell_fraction = 0.5
            # 4. Over-cap trim
            elif mkt_value > max_dollars_per_position:
                trim_dollars = mkt_value - cap_target_after_trim
                if trim_dollars >= guardrails.min_trade_dollars:
                    sell_reason = (
                        f"overcap_trim ({mkt_value/account_equity:.1%} > {guardrails.max_position_pct:.0%})"
                    )
                    sell_fraction = trim_dollars / mkt_value

            if not sell_reason or sell_fraction <= 0:
                continue

            if sell_fraction >= 0.999:
                # Full sell — use exact qty
                orders.append(
                    PlannedOrder(
                        symbol=ticker,
                        side="sell",
                        dollar_amount=round(mkt_value, 2),
                        limit_price=_round_price(last),
                        quantity_estimate=qty,
                        type="market",  # fractional requires market
                        time_in_force="gfd",
                        rationale=sell_reason,
                        quantity=qty,
                    )
                )
                cash_freed += mkt_value
            else:
                # Partial sell — use dollar_amount path
                sell_dollars = round(mkt_value * sell_fraction, 2)
                if sell_dollars < guardrails.min_trade_dollars:
                    continue
                orders.append(
                    PlannedOrder(
                        symbol=ticker,
                        side="sell",
                        dollar_amount=sell_dollars,
                        limit_price=_round_price(last),
                        quantity_estimate=round(sell_dollars / last, 6),
                        type="market",
                        time_in_force="gfd",
                        rationale=sell_reason,
                    )
                )
                cash_freed += sell_dollars

    # ---- BUY pass ----
    # Cash account: today's sell proceeds settle T+1 and aren't usable for today's buys
    usable_cash = cash_available if guardrails.cash_account_t_plus_1 else cash_available + cash_freed
    remaining_cash = usable_cash
    open_positions = sum(
        1 for t, p in positions.items()
        if p.get("mkt_value", 0.0) > 0 and not any(
            o.symbol == t and o.side == "sell" and o.quantity_estimate >= p["qty"] * 0.999
            for o in orders
        )
    )
    whitelist = set(guardrails.whitelist) if guardrails.whitelist else None

    for c in ranked:
        if open_positions + sum(1 for o in orders if o.side == "buy" and o.symbol not in positions) >= guardrails.max_total_positions:
            skipped.append({"ticker": c.ticker, "reason": "max_positions_reached"})
            continue
        if c.score < guardrails.min_candidate_score:
            skipped.append({"ticker": c.ticker, "reason": f"score_below_min ({c.score:.2f})"})
            continue
        if whitelist and c.ticker not in whitelist:
            skipped.append({"ticker": c.ticker, "reason": "not_in_whitelist"})
            continue
        if c.ticker not in quotes:
            skipped.append({"ticker": c.ticker, "reason": "no_quote"})
            continue

        already_in = positions.get(c.ticker, {}).get("mkt_value", 0.0)
        # If we sold this ticker fully in the sell pass, treat as empty
        full_sold = any(
            o.symbol == c.ticker and o.side == "sell"
            and o.quantity is not None
            and o.quantity >= positions.get(c.ticker, {}).get("qty", 0.0) * 0.999
            for o in orders
        )
        if full_sold:
            already_in = 0.0
        headroom = max(0.0, max_dollars_per_position - already_in)
        target_dollars = min(headroom, remaining_cash)
        if target_dollars < guardrails.min_trade_dollars:
            reason = (
                "position_full" if headroom < guardrails.min_trade_dollars
                else f"insufficient_cash (${remaining_cash:.2f} < ${guardrails.min_trade_dollars:.2f})"
            )
            skipped.append({"ticker": c.ticker, "reason": reason})
            continue

        ask = quotes[c.ticker].get("ask") or quotes[c.ticker].get("last")
        if not ask or ask <= 0:
            skipped.append({"ticker": c.ticker, "reason": "bad_quote"})
            continue
        limit = _round_price(ask * (1 + guardrails.limit_price_buffer))
        qty_est = round(target_dollars / limit, 6)

        orders.append(
            PlannedOrder(
                symbol=c.ticker,
                side="buy",
                dollar_amount=round(target_dollars, 2),
                limit_price=limit,
                quantity_estimate=qty_est,
                type="market" if target_dollars < limit else "limit",
                time_in_force="gfd",
                rationale=c.rationale or f"score {c.score:+.2f}",
            )
        )
        remaining_cash -= target_dollars
        if c.ticker not in positions:
            open_positions += 1

    return TradePlan(
        generated_at=now,
        account_number=account_number,
        account_equity=account_equity,
        cash_available=cash_available,
        intraday_pnl_pct=intraday_pnl_pct,
        circuit_breaker_tripped=False,
        orders=orders,
        skipped=skipped,
    )
