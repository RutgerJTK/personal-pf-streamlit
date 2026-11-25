"""
Portfolio-vs-benchmark data and the cumulative return chart, shared by the
generic Dashboard and the generic Benchmark page (same look as the personal
Benchmark page: portfolio in blue, sticky colours per benchmark, labels at
the line ends).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import engine
import generic_session as gs
from market_data import fetch_history
from t212_parser import ParseResult

PORTFOLIO = "Portfolio"
PORTFOLIO_COLOR = "#2a78d6"
SLOTS = ["#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
NEUTRAL = "#8a8986"
MAX_BENCH = len(SLOTS)
DEFAULT_BENCH = ("S&P 500", "MSCI World")


@dataclass
class Comparison:
    levels: pd.DataFrame            # daily growth of 1 euro; Portfolio (TWR) + one column per benchmark
    daily: pd.DataFrame             # engine.portfolio_history()["daily"]
    twr: pd.Series
    failed: list[str] = field(default_factory=list)
    bench_info: list[dict] = field(default_factory=list)


def compare(parsed: ParseResult, selected: list[str], benches: dict) -> Comparison | None:
    """Time-weighted portfolio growth next to the selected benchmarks, all in EUR.
    None when there's no deposit to measure a return from."""
    df = parsed.transactions
    first_day = df["datetime"].min().normalize()
    fetch_from = (first_day - pd.Timedelta(days=10)).strftime("%Y-%m-%d")

    bench_cols, info, failed = {}, [], []
    for name in selected:  # one call per benchmark: cached on its own, shared by all users
        p, i = fetch_history({name: tuple(benches[name]["tickers"])}, fetch_from, adjusted=True)
        if name in p:
            clean, _ = engine.repair_prices(p[[name]])
            bench_cols[name] = clean[name].dropna()
            info.append(i.loc[name].to_dict() | {"name": name})
        else:
            failed.append(name)

    traded = df.loc[df["action"].isin(["BUY", "SELL"]), "isin"].unique()
    res = gs.resolve(parsed, traded)  # closed positions too: they were part of past returns
    own, _ = fetch_history({i: c for i, c in res.candidates.items() if c}, fetch_from, adjusted=False)
    daily = engine.portfolio_history(df, own)["daily"]
    twr = engine.twr_index(daily)
    if twr.empty:
        return None

    days = pd.date_range(first_day, pd.Timestamp.today().normalize(), freq="D")
    levels = pd.DataFrame(index=days)
    levels[PORTFOLIO] = twr.reindex(days)
    for name, s in bench_cols.items():
        levels[name] = s.reindex(s.index.union(days)).sort_index().ffill().reindex(days)
    return Comparison(levels.dropna(axis=1, how="all"), daily, twr, failed, info)


def colors_for(selected: list[str], key: str) -> dict[str, str]:
    """Sticky colour slot per benchmark for as long as it stays selected."""
    held = {k: v for k, v in st.session_state.get(key, {}).items() if k in selected}
    for name in selected:
        if name not in held:
            free = [c for c in SLOTS if c not in held.values()]
            held[name] = free[0] if free else NEUTRAL
    st.session_state[key] = held
    return {PORTFOLIO: PORTFOLIO_COLOR, **held}


def line(name: str, color: str) -> dict:
    return dict(color=color, width=3 if name == PORTFOLIO else 2)


def layout(fig: go.Figure, height: int, pct_axis: bool = True) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=10, b=0), hovermode="x unified",
        legend=dict(orientation="h", y=-0.12, title=None),
        xaxis=dict(title=None), yaxis=dict(title=None, ticksuffix="%" if pct_axis else None),
    )
    return fig


def cumulative_figure(rebased: pd.DataFrame, colors: dict[str, str], height: int = 460) -> go.Figure:
    """Cumulative return since the first row of `rebased`, portfolio drawn last (on top)."""
    series = list(rebased.columns)
    order = [c for c in series if c != PORTFOLIO] + ([PORTFOLIO] if PORTFOLIO in series else [])
    fig = go.Figure()
    for name in order:
        y = (rebased[name] - 1) * 100
        fig.add_scatter(x=y.index, y=y, name=name, mode="lines", line=line(name, colors[name]),
                        hovertemplate="%{y:+.1f}%<extra>" + name + "</extra>")
        last = y.dropna()
        if len(last) and len(series) <= 4:  # direct labels at the line ends while readable
            fig.add_annotation(x=last.index[-1], y=last.iloc[-1], text=f"{name} {last.iloc[-1]:+.0f}%",
                               showarrow=False, xanchor="left", xshift=6, font=dict(size=12))
    fig.add_hline(y=0, line=dict(color=NEUTRAL, width=1))
    fig = layout(fig, height)
    if len(series) <= 4:
        fig.update_layout(margin=dict(r=140))
    return fig