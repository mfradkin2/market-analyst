"""
Local paper-trading book.

Persists to a JSON file so you can run the analyst loop daily and have it
build up a track record before risking real money. The paper book and the
live Robinhood path share the same `Order` interface — flip a flag to go
from paper to live (don't do that until you're confident).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Literal


Side = Literal["buy", "sell"]


@dataclass
class Order:
    ticker: str
    side: Side
    quantity: float
    price: float  # fill price for paper; limit price for live
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    note: str = ""

    @property
    def signed_quantity(self) -> float:
        return self.quantity if self.side == "buy" else -self.quantity


@dataclass
class Position:
    ticker: str
    quantity: float
    avg_cost: float

    def market_value(self, last_price: float) -> float:
        return self.quantity * last_price

    def unrealized_pnl(self, last_price: float) -> float:
        return (last_price - self.avg_cost) * self.quantity


class PaperBook:
    def __init__(self, path: Path, starting_cash: float = 10_000.0):
        self.path = Path(path)
        if self.path.exists():
            self._load()
        else:
            self.cash: float = starting_cash
            self.positions: dict[str, Position] = {}
            self.orders: list[Order] = []
            self._save()

    def _load(self) -> None:
        data = json.loads(self.path.read_text())
        self.cash = float(data["cash"])
        self.positions = {
            t: Position(**p) for t, p in data["positions"].items()
        }
        self.orders = [Order(**o) for o in data["orders"]]

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {
                    "cash": self.cash,
                    "positions": {t: asdict(p) for t, p in self.positions.items()},
                    "orders": [asdict(o) for o in self.orders],
                },
                indent=2,
            )
        )

    def submit(self, order: Order) -> None:
        cost = order.quantity * order.price
        if order.side == "buy":
            if cost > self.cash:
                raise ValueError(f"insufficient cash: need ${cost:,.2f}, have ${self.cash:,.2f}")
            self.cash -= cost
            existing = self.positions.get(order.ticker)
            if existing:
                new_qty = existing.quantity + order.quantity
                new_cost = (existing.avg_cost * existing.quantity + cost) / new_qty
                self.positions[order.ticker] = Position(order.ticker, new_qty, new_cost)
            else:
                self.positions[order.ticker] = Position(order.ticker, order.quantity, order.price)
        else:
            existing = self.positions.get(order.ticker)
            if not existing or existing.quantity < order.quantity:
                raise ValueError(f"cannot sell {order.quantity} {order.ticker}, only hold {existing.quantity if existing else 0}")
            self.cash += cost
            remaining = existing.quantity - order.quantity
            if remaining == 0:
                del self.positions[order.ticker]
            else:
                self.positions[order.ticker] = Position(order.ticker, remaining, existing.avg_cost)
        self.orders.append(order)
        self._save()

    def equity(self, prices: dict[str, float]) -> float:
        mv = sum(p.market_value(prices.get(t, p.avg_cost)) for t, p in self.positions.items())
        return self.cash + mv

    def snapshot(self, prices: dict[str, float]) -> dict:
        rows = []
        for t, p in self.positions.items():
            px = prices.get(t, p.avg_cost)
            rows.append(
                {
                    "ticker": t,
                    "qty": p.quantity,
                    "avg_cost": p.avg_cost,
                    "last": px,
                    "mv": p.market_value(px),
                    "upnl": p.unrealized_pnl(px),
                }
            )
        return {
            "as_of": date.today().isoformat(),
            "cash": self.cash,
            "equity": self.equity(prices),
            "positions": rows,
        }
