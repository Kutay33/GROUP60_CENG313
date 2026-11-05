"""Evaluation metric, backtest and plotting helpers."""

from __future__ import annotations

import random

import numpy as np
import pandas as pd

TRADING_DAYS = 252
MIN_POSITION, MAX_POSITION = 0.0, 2.0


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def adjusted_sharpe(forward_returns, risk_free_rate, position) -> float:
    """Re-implementation of the competition's volatility-adjusted Sharpe ratio.

    The strategy holds `position` (0..2) of the S&P 500 and the rest in the risk-free asset.
    The Sharpe ratio is divided by two penalties:
      * volatility penalty if strategy vol > 1.2 x market vol
      * return penalty if the strategy earns less than the market
    NOTE: written from the official metric notebook; double-check against it on Kaggle.
    """
    fr = np.asarray(forward_returns, dtype=float)
    rf = np.asarray(risk_free_rate, dtype=float)
    pos = np.asarray(position, dtype=float)
    if pos.min() < MIN_POSITION or pos.max() > MAX_POSITION:
        raise ValueError("positions must be in [0, 2]")

    strat = rf * (1 - pos) + pos * fr
    strat_excess = strat - rf
    strat_mean = np.prod(1 + strat_excess) ** (1 / len(strat)) - 1
    strat_std = strat.std(ddof=1)
    if strat_std == 0:
        return 0.0
    sharpe = strat_mean / strat_std * np.sqrt(TRADING_DAYS)
    strat_vol = strat_std * np.sqrt(TRADING_DAYS) * 100

    mkt_excess = fr - rf
    mkt_mean = np.prod(1 + mkt_excess) ** (1 / len(fr)) - 1
    mkt_vol = fr.std(ddof=1) * np.sqrt(TRADING_DAYS) * 100

    excess_vol = max(0.0, strat_vol / mkt_vol - 1.2) if mkt_vol > 0 else 0.0
    vol_penalty = 1 + excess_vol
    return_gap = max(0.0, (mkt_mean - strat_mean) * 100 * TRADING_DAYS)
    return_penalty = 1 + return_gap ** 2 / 100
    return float(min(sharpe / (vol_penalty * return_penalty), 1_000_000))


def score_frame(df: pd.DataFrame, position) -> float:
    """Convenience wrapper: df must contain forward_returns and risk_free_rate."""
    return adjusted_sharpe(df["forward_returns"], df["risk_free_rate"], position)


def backtest(df: pd.DataFrame, position) -> pd.DataFrame:
    """Daily strategy vs. buy-and-hold returns and their cumulative wealth curves."""
    pos = np.asarray(position, dtype=float)
    out = pd.DataFrame({"date_id": df["date_id"].values, "position": pos})
    out["market"] = df["forward_returns"].values
    out["strategy"] = df["risk_free_rate"].values * (1 - pos) + pos * df["forward_returns"].values
    out["market_wealth"] = (1 + out["market"]).cumprod()
    out["strategy_wealth"] = (1 + out["strategy"]).cumprod()
    return out


def summary_stats(df: pd.DataFrame, position) -> dict:
    bt = backtest(df, position)
    ann = lambda r: (1 + r).prod() ** (TRADING_DAYS / len(r)) - 1  # noqa: E731
    return {
        "score": score_frame(df, position),
        "ann_return": ann(bt["strategy"]),
        "ann_vol": bt["strategy"].std() * np.sqrt(TRADING_DAYS),
        "market_ann_return": ann(bt["market"]),
        "market_ann_vol": bt["market"].std() * np.sqrt(TRADING_DAYS),
        "mean_position": bt["position"].mean(),
    }


def plot_wealth(bt: pd.DataFrame, title: str = "Strategy vs buy & hold", ax=None):
    import matplotlib.pyplot as plt

    ax = ax or plt.subplots(figsize=(10, 4))[1]
    ax.plot(bt["date_id"], bt["market_wealth"], label="Buy & hold (position = 1)", color="#888888")
    ax.plot(bt["date_id"], bt["strategy_wealth"], label="Strategy", color="#1f6feb")
    ax.set_yscale("log")
    ax.set_xlabel("date_id")
    ax.set_ylabel("Wealth (log scale)")
    ax.set_title(title)
    ax.legend()
    return ax
