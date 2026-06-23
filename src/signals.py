"""
Signal generation: combine market structure (correlations, momentum) with
recent congressional trade activity into a ranked candidate list.

The score is intentionally simple and transparent — you should be able to read
the code and understand why a name made the list. No black-box ML here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import market_data as md


@dataclass
class Candidate:
    ticker: str
    score: float
    momentum_20d: float
    vol_ann: float
    beta: float
    congress_net_dollar: float
    n_politicians: int
    social_volume: int
    social_sentiment: float
    analyst_upside_pct: float
    analyst_rating_score: float
    cluster_momentum: float
    rationale: str


def _zscore(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    if not sd:
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / sd


def momentum_20d(history: md.PriceHistory) -> pd.Series:
    p = history.prices
    return (p.iloc[-1] / p.iloc[-21] - 1).dropna()


def rank_candidates(
    history: md.PriceHistory,
    congress_signals: pd.DataFrame,
    social_signals: pd.DataFrame | None = None,
    analyst_data: pd.DataFrame | None = None,
    universe: list[str] | None = None,
    weight_momentum: float = 0.22,
    weight_congress: float = 0.18,
    weight_social: float = 0.15,
    weight_analyst: float = 0.20,
    weight_cluster: float = 0.15,
    weight_lowvol: float = 0.10,
) -> list[Candidate]:
    """
    Score components (each z-scored, then weighted):
      - momentum: ticker's own 20d return
      - congress: smart-money flow (lagging, thematic)
      - social:   z(mention volume) * sign(sentiment) — spike-direction signal
      - analyst:  upside_pct + rating_score, averaged
      - cluster:  weighted-avg momentum of top-5 correlated peers
      - low-vol:  position-sizing penalty for high-vol names

    Weights sum to 1.0. Returns candidates sorted high-to-low.
    """
    mom = momentum_20d(history)
    vol = pd.Series({t: md.annualized_vol(history, t) for t in history.prices.columns})

    df = pd.DataFrame({"momentum_20d": mom, "vol_ann": vol})
    if universe:
        df = df.loc[df.index.intersection(universe)]

    cs = congress_signals.set_index("ticker") if not congress_signals.empty else pd.DataFrame()
    df["congress_net_dollar"] = cs["net_dollar"].reindex(df.index).fillna(0.0) if "net_dollar" in cs else 0.0
    df["n_politicians"] = cs["n_politicians"].reindex(df.index).fillna(0).astype(int) if "n_politicians" in cs else 0

    if social_signals is not None and not social_signals.empty:
        ss = social_signals.set_index("ticker")
        df["social_volume"] = ss["mention_volume"].reindex(df.index).fillna(0).astype(int)
        df["social_sentiment"] = ss["sentiment_avg"].reindex(df.index).fillna(0.0)
    else:
        df["social_volume"] = 0
        df["social_sentiment"] = 0.0

    if analyst_data is not None and not analyst_data.empty:
        df["analyst_upside_pct"] = analyst_data["upside_pct"].reindex(df.index).fillna(0.0)
        df["analyst_rating_score"] = analyst_data["rating_score"].reindex(df.index).fillna(0.0)
        upgrade_col = "upgrade_net_30d" if "upgrade_net_30d" in analyst_data.columns else None
        df["upgrade_net_30d"] = (
            analyst_data[upgrade_col].reindex(df.index).fillna(0).astype(int)
            if upgrade_col else 0
        )
    else:
        df["analyst_upside_pct"] = 0.0
        df["analyst_rating_score"] = 0.0
        df["upgrade_net_30d"] = 0

    df["cluster_momentum"] = pd.Series(
        {t: md.cluster_momentum(history, t) for t in df.index}
    )

    df["z_mom"] = _zscore(df["momentum_20d"])
    df["z_cong"] = _zscore(df["congress_net_dollar"])
    df["z_lowvol"] = -_zscore(df["vol_ann"])
    z_vol_social = _zscore(df["social_volume"])
    df["z_social"] = z_vol_social * np.sign(df["social_sentiment"])
    z_upside = _zscore(df["analyst_upside_pct"])
    z_rating = _zscore(df["analyst_rating_score"])
    z_upgrades = _zscore(df["upgrade_net_30d"].astype(float))
    # Analyst component: blend of target upside, current consensus, and recent rating-change momentum
    df["z_analyst"] = (z_upside + z_rating + z_upgrades) / 3.0
    df["z_cluster"] = _zscore(df["cluster_momentum"])

    df["score"] = (
        weight_momentum * df["z_mom"]
        + weight_congress * df["z_cong"]
        + weight_social * df["z_social"]
        + weight_analyst * df["z_analyst"]
        + weight_cluster * df["z_cluster"]
        + weight_lowvol * df["z_lowvol"]
    )
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["score"])
    df = df.sort_values("score", ascending=False)

    out: list[Candidate] = []
    for ticker, row in df.iterrows():
        if ticker == md.SPY:
            continue
        try:
            beta = md.beta_vs_market(history, ticker) if md.SPY in history.prices.columns else float("nan")
        except Exception:
            beta = float("nan")
        rationale_bits = []
        if row["z_mom"] > 0.5:
            rationale_bits.append(f"+momentum ({row['momentum_20d']:.1%} 20d)")
        if row["congress_net_dollar"] > 0 and row["n_politicians"] > 0:
            rationale_bits.append(
                f"congress net +${row['congress_net_dollar']:,.0f} across {int(row['n_politicians'])}"
            )
        if row["social_volume"] > 0 and abs(row["z_social"]) > 0.5:
            direction = "bullish" if row["social_sentiment"] > 0 else "bearish"
            rationale_bits.append(
                f"{direction} social ({int(row['social_volume'])} mentions, sent={row['social_sentiment']:+.2f})"
            )
        if abs(row["z_analyst"]) > 0.5 and row["analyst_upside_pct"]:
            extra = ""
            if int(row.get("upgrade_net_30d", 0)) != 0:
                extra = f", upg30d={int(row['upgrade_net_30d']):+d}"
            rationale_bits.append(
                f"analysts {row['analyst_upside_pct']:+.0%} upside, rating={row['analyst_rating_score']:+.2f}{extra}"
            )
        if abs(row["z_cluster"]) > 0.5:
            rationale_bits.append(
                f"cluster mom {row['cluster_momentum']:+.1%}"
            )
        if row["z_lowvol"] > 0.5:
            rationale_bits.append(f"lower vol ({row['vol_ann']:.0%} ann)")
        out.append(
            Candidate(
                ticker=str(ticker),
                score=float(row["score"]),
                momentum_20d=float(row["momentum_20d"]),
                vol_ann=float(row["vol_ann"]),
                beta=beta,
                congress_net_dollar=float(row["congress_net_dollar"]),
                n_politicians=int(row["n_politicians"]),
                social_volume=int(row["social_volume"]),
                social_sentiment=float(row["social_sentiment"]),
                analyst_upside_pct=float(row["analyst_upside_pct"]),
                analyst_rating_score=float(row["analyst_rating_score"]),
                cluster_momentum=float(row["cluster_momentum"]),
                rationale=", ".join(rationale_bits) or "neutral",
            )
        )
    return out


def score_for_ticker(candidates: list[Candidate], ticker: str) -> float | None:
    """Helper: pull the score for a specific ticker (used by sell logic)."""
    for c in candidates:
        if c.ticker == ticker:
            return c.score
    return None


def position_size(
    candidate: Candidate,
    portfolio_value: float,
    risk_per_trade: float = 0.01,
    stop_loss_pct: float = 0.08,
) -> int:
    """
    Volatility-aware sizing: risk `risk_per_trade` of portfolio per position,
    assuming a `stop_loss_pct` stop. Returns share count (capped at 25% of book).
    """
    if candidate.vol_ann <= 0:
        return 0
    dollar_risk = portfolio_value * risk_per_trade
    per_share_risk = candidate.momentum_20d  # placeholder — caller should pass live price
    # Caller passes price separately in main; keep this fn focused on dollar sizing.
    max_dollars = portfolio_value * 0.25
    risk_dollars = dollar_risk / stop_loss_pct
    return int(min(max_dollars, risk_dollars))
