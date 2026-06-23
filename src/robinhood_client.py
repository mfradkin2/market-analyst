"""
Robinhood client — optional, opt-in, safety-first.

Robinhood has no official Python SDK. The community library `robin-stocks`
reverse-engineers their app's HTTP API and requires:
  - your username + password (via env vars; never commit them)
  - MFA — typically a TOTP secret (set up an authenticator app, save the seed)

This wrapper:
  1. Refuses to send live orders unless RH_LIVE=1 is set in the environment AND
     the call site passes confirm=True. Two gates by design.
  2. Logs every intent before executing.
  3. Mirrors the paper PaperBook.submit signature so signal code is unaware
     of which backend it's hitting.

You must run `pip install robin-stocks` separately — it's not in the default
requirements because the live path is opt-in.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from .portfolio import Order


log = logging.getLogger(__name__)


@dataclass
class RobinhoodConfig:
    username: str
    password: str
    mfa_totp_secret: str | None = None  # if you use TOTP MFA

    @classmethod
    def from_env(cls) -> "RobinhoodConfig":
        try:
            return cls(
                username=os.environ["RH_USERNAME"],
                password=os.environ["RH_PASSWORD"],
                mfa_totp_secret=os.environ.get("RH_MFA_TOTP"),
            )
        except KeyError as e:
            raise RuntimeError(f"missing env var: {e}. Set RH_USERNAME, RH_PASSWORD, RH_MFA_TOTP.")


def _is_live_mode() -> bool:
    return os.environ.get("RH_LIVE", "0") == "1"


class RobinhoodClient:
    """
    Live trading wrapper. Defaults to dry-run; flip RH_LIVE=1 and pass
    confirm=True to actually send orders.
    """

    def __init__(self, config: RobinhoodConfig | None = None):
        self.config = config or RobinhoodConfig.from_env()
        self._authed = False
        self._rs = None  # lazy import — only required when live

    def _login(self) -> None:
        if self._authed:
            return
        try:
            import robin_stocks.robinhood as rs  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "robin-stocks not installed. `pip install robin-stocks` to use the live path."
            ) from e
        mfa_code = None
        if self.config.mfa_totp_secret:
            try:
                import pyotp  # type: ignore
                mfa_code = pyotp.TOTP(self.config.mfa_totp_secret).now()
            except ImportError as e:
                raise RuntimeError("pyotp required for TOTP MFA. `pip install pyotp`.") from e
        rs.login(self.config.username, self.config.password, mfa_code=mfa_code)
        self._rs = rs
        self._authed = True

    def last_price(self, ticker: str) -> float:
        self._login()
        quote = self._rs.stocks.get_latest_price(ticker)
        return float(quote[0])

    def submit(self, order: Order, *, confirm: bool = False) -> dict:
        """
        Mirrors PaperBook.submit. Returns the broker response or a dry-run record.

        Both gates must be open to actually transmit:
          - env var RH_LIVE=1
          - caller passes confirm=True
        """
        if not (_is_live_mode() and confirm):
            log.warning(
                "DRY-RUN order (not sent): %s %s %s @ %s — RH_LIVE=%s confirm=%s",
                order.side, order.quantity, order.ticker, order.price,
                os.environ.get("RH_LIVE", "0"), confirm,
            )
            return {"status": "dry_run", "order": order.__dict__}

        self._login()
        log.warning("LIVE order: %s %s %s @ %s", order.side, order.quantity, order.ticker, order.price)
        if order.side == "buy":
            return self._rs.orders.order_buy_limit(
                symbol=order.ticker, quantity=order.quantity, limitPrice=order.price
            )
        return self._rs.orders.order_sell_limit(
            symbol=order.ticker, quantity=order.quantity, limitPrice=order.price
        )
