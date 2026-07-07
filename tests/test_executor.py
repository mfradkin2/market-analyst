"""Executor guardrail tests — these protect real money. Run before every push."""

from src.executor import Guardrails, plan_trades, sector_of
from src.signals import Candidate


def mk_cand(ticker: str, score: float, vol: float = 0.20, mom: float = 0.05) -> Candidate:
    return Candidate(
        ticker=ticker,
        score=score,
        momentum_20d=mom,
        vol_ann=vol,
        beta=1.0,
        congress_net_dollar=0.0,
        n_politicians=0,
        social_volume=0,
        social_sentiment=0.0,
        analyst_upside_pct=0.0,
        analyst_rating_score=0.0,
        cluster_momentum=0.0,
        rationale="test",
    )


def quotes_for(*tickers: str, price: float = 100.0) -> dict:
    return {t: {"last": price, "ask": price, "bid": price} for t in tickers}


BASE = dict(
    account_number="TEST",
    account_equity=1000.0,
    cash_available=500.0,
    intraday_pnl_pct=0.0,
)


def test_dust_sweep_liquidates_fragments():
    positions = {"LLY": {"qty": 0.0016, "avg_cost": 1104.69, "mkt_value": 1.89}}
    plan = plan_trades(
        ranked=[], quotes=quotes_for("LLY", price=1200.0), positions=positions,
        guardrails=Guardrails(), **BASE,
    )
    assert len(plan.orders) == 1
    o = plan.orders[0]
    assert o.side == "sell" and o.symbol == "LLY"
    assert o.quantity == 0.0016  # exact full-position qty
    assert "dust_sweep" in o.rationale


def test_dust_sweep_ignores_normal_positions():
    # Position under the 18% cap with flat P&L: no rule should fire.
    positions = {"PG": {"qty": 1.0, "avg_cost": 149.2, "mkt_value": 150.0}}
    plan = plan_trades(
        ranked=[], quotes=quotes_for("PG", price=150.0), positions=positions,
        guardrails=Guardrails(), **BASE,
    )
    assert plan.orders == []


def test_stop_loss_full_exit_still_fires():
    positions = {"QQQ": {"qty": 1.0, "avg_cost": 100.0, "mkt_value": 90.0}}
    plan = plan_trades(
        ranked=[mk_cand("QQQ", score=0.5)], quotes=quotes_for("QQQ", price=90.0),
        positions=positions, guardrails=Guardrails(), **BASE,
    )
    sells = [o for o in plan.orders if o.side == "sell"]
    assert len(sells) == 1 and sells[0].quantity == 1.0
    assert "stop_loss" in sells[0].rationale


def test_t_plus_1_sell_proceeds_not_reused():
    # Cash account with zero settled cash: a stop-loss sell fires, but the buy
    # pass must NOT spend the freed cash in the same session.
    positions = {"HD": {"qty": 1.0, "avg_cost": 100.0, "mkt_value": 90.0}}
    plan = plan_trades(
        ranked=[mk_cand("PG", score=1.0)],
        quotes=quotes_for("HD", "PG", price=90.0),
        positions=positions,
        guardrails=Guardrails(whitelist=["HD", "PG"]),
        account_number="TEST", account_equity=1000.0,
        cash_available=0.0, intraday_pnl_pct=0.0,
    )
    assert any(o.side == "sell" for o in plan.orders)
    assert not any(o.side == "buy" for o in plan.orders)


def test_vol_scaling_shrinks_high_vol_buys():
    g = Guardrails(whitelist=["LOWV", "HIGHV"], sector_cap_pct=1.0)
    plan = plan_trades(
        ranked=[mk_cand("LOWV", 1.0, vol=0.20), mk_cand("HIGHV", 0.9, vol=0.67)],
        quotes=quotes_for("LOWV", "HIGHV"), positions={},
        guardrails=g, **BASE,
    )
    buys = {o.symbol: o.dollar_amount for o in plan.orders if o.side == "buy"}
    # LOWV: vol below target -> full 18% cap = $180. HIGHV: scaled by 0.25/0.67.
    assert buys["LOWV"] == 180.0
    assert buys["HIGHV"] < 180.0 * 0.45
    assert buys["HIGHV"] >= 180.0 * 0.25  # min_vol_scalar floor


def test_sector_cap_blocks_concentration():
    # Financials already at 34% of a $1000 book; the cap (35%) leaves only $10
    # of sector room -> below min_trade_dollars -> skip with sector_cap reason.
    positions = {
        "JPM": {"qty": 1.75, "avg_cost": 100.0, "mkt_value": 175.0},
        "BAC": {"qty": 1.65, "avg_cost": 100.0, "mkt_value": 165.0},
    }
    plan = plan_trades(
        ranked=[mk_cand("V", score=1.5)],
        quotes=quotes_for("JPM", "BAC", "V"), positions=positions,
        guardrails=Guardrails(whitelist=["V"]), **BASE,
    )
    assert not any(o.side == "buy" for o in plan.orders)
    assert any("sector_cap" in s.get("reason", "") for s in plan.skipped)


def test_sector_cap_counts_planned_buys():
    # Two financials ranked back-to-back: first buy consumes sector room,
    # second gets clipped so the pair stays under the 35% cap.
    g = Guardrails(whitelist=["JPM", "BAC"])
    plan = plan_trades(
        ranked=[mk_cand("JPM", 1.5), mk_cand("BAC", 1.4)],
        quotes=quotes_for("JPM", "BAC"), positions={},
        guardrails=g, **BASE,
    )
    fin_total = sum(o.dollar_amount for o in plan.orders if o.side == "buy")
    assert fin_total <= 0.35 * 1000.0 + 0.01


def test_regime_gate_blocks_buys_keeps_sells():
    positions = {"QQQ": {"qty": 1.0, "avg_cost": 100.0, "mkt_value": 90.0}}
    plan = plan_trades(
        ranked=[mk_cand("PG", score=2.0)],
        quotes=quotes_for("QQQ", "PG", price=90.0), positions=positions,
        guardrails=Guardrails(whitelist=["PG"]),
        regime_scale=0.0, **BASE,
    )
    assert any(o.side == "sell" for o in plan.orders)      # stop-loss unaffected
    assert not any(o.side == "buy" for o in plan.orders)   # buys gated
    assert plan.regime_scale == 0.0
    assert any(s.get("reason") == "regime_gate" for s in plan.skipped)


def test_regime_half_budget():
    g = Guardrails(whitelist=["PG"], max_position_pct=1.0, sector_cap_pct=1.0)
    plan = plan_trades(
        ranked=[mk_cand("PG", 1.0)], quotes=quotes_for("PG"), positions={},
        guardrails=g, regime_scale=0.5, **BASE,
    )
    buys = [o for o in plan.orders if o.side == "buy"]
    assert len(buys) == 1
    assert buys[0].dollar_amount == 250.0  # half of $500 cash


def test_min_score_filters_weak_candidates():
    plan = plan_trades(
        ranked=[mk_cand("PG", score=0.05)], quotes=quotes_for("PG"), positions={},
        guardrails=Guardrails(whitelist=["PG"]), **BASE,
    )
    assert plan.orders == []
    assert any("score_below_min" in s.get("reason", "") for s in plan.skipped)


def test_sector_of_unknown_ticker():
    assert sector_of("ZZZTOP") == "other"
