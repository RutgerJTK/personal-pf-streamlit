import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import engine
from market_data import fetch_history, load_benchmarks, load_isin_map
from s3_fetcher import load_transactions_or_stop

# Categorical palette in fixed order. The portfolio always owns slot 1 (blue);
# each benchmark keeps the slot it got when first selected, so removing one
# never repaints the others.
PORTFOLIO = "Portfolio"
PORTFOLIO_COLOR = "#2a78d6"
SLOTS = ["#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
NEUTRAL = "#8a8986"
MAX_BENCH = len(SLOTS)


def pct(x: float, sign: bool = True, digits: int = 1) -> str:
    if x is None or x != x:
        return "–"
    return f"{x:+.{digits}%}" if sign else f"{x:.{digits}%}"


def eur(x: float) -> str:
    return f"€{x:,.0f}".replace("€-", "-€")


def _layout(fig: go.Figure, height: int, pct_axis: bool = True) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=10, b=0), hovermode="x unified",
        legend=dict(orientation="h", y=-0.12, title=None),
        xaxis=dict(title=None), yaxis=dict(title=None, ticksuffix="%" if pct_axis else None),
    )
    return fig


def colors_for(selected: list[str]) -> dict[str, str]:
    """Sticky color slot per benchmark for as long as it stays selected."""
    held = {k: v for k, v in st.session_state.get("bench_colors", {}).items() if k in selected}
    for name in selected:
        if name not in held:
            free = [c for c in SLOTS if c not in held.values()]
            held[name] = free[0] if free else NEUTRAL
    st.session_state["bench_colors"] = held
    return {PORTFOLIO: PORTFOLIO_COLOR, **held}


def _line(name: str, color: str) -> dict:
    is_pf = name == PORTFOLIO
    return dict(color=color, width=3 if is_pf else 2)


st.title("🏁 Am I Beating the Market?")

df = load_transactions_or_stop()
meta = load_isin_map()
benches = load_benchmarks()
if not benches:
    st.error("No benchmarks configured. Check `shared-config/benchmarks.json`.")
    st.stop()

# --- Controls ---------------------------------------------------------------
groups = {}
for name, b in benches.items():
    groups.setdefault(b.get("group", "Other"), []).append(name)

c1, c2, c3 = st.columns([4, 3, 1], vertical_alignment="bottom")
selected = c1.multiselect(
    "Compare against", list(benches), default=[n for n, b in benches.items() if b.get("default")],
    max_selections=MAX_BENCH, placeholder="Pick indices, commodities, crypto…",
    format_func=lambda n: f"{n} · {benches[n].get('group', '')}",
    help=f"Up to {MAX_BENCH} at once. Edit the list in `shared-config/benchmarks.json`.",
)
period = c2.segmented_control("Period", list(engine.PERIODS), default="All", key="bench_period") or "All"
with c3:
    if st.button("🔄 Prices", help="Fetch fresh price history (cached for 6 hours)"):
        fetch_history.clear()
include_pf = st.toggle("Include my portfolio", value=True,
                       help="Your account's time-weighted return: holdings at market value plus cash, "
                            "with the timing of your deposits taken out so it compares fairly with an index.")
if not selected and not include_pf:
    st.info("Pick at least one benchmark, or include your portfolio.")
    st.stop()

# --- Data -------------------------------------------------------------------
first_day = df["datetime"].min().normalize()
fetch_from = (first_day - pd.Timedelta(days=10)).strftime("%Y-%m-%d")

bench_cols, bench_info, failed, bench_repairs = {}, [], [], []
for name in selected:  # one call per benchmark, so each is cached on its own
    p, info = fetch_history({name: tuple(benches[name]["tickers"])}, fetch_from, adjusted=True)
    if name in p:
        clean, ev = engine.repair_prices(p[[name]])
        bench_cols[name] = clean[name].dropna()
        bench_repairs.append(ev)
        bench_info.append(info.loc[name].to_dict() | {"name": name})
    else:
        failed.append(name)
if failed:
    st.warning(f"No price history for {', '.join(failed)} right now (Yahoo Finance did not answer).")

hist = None
if include_pf:
    traded = df.loc[df["action"].isin(["BUY", "SELL"]), "isin"].unique()
    cands = {i: tuple(meta[i]["tickers"]) for i in traded if meta.get(i, {}).get("tickers")}
    own_prices, own_info = fetch_history(cands, fetch_from, adjusted=False)
    hist = engine.portfolio_history(df, own_prices)
    daily = hist["daily"]
    twr = engine.twr_index(daily)

days = pd.date_range(first_day, pd.Timestamp.today().normalize(), freq="D")
levels = pd.DataFrame(index=days)
if include_pf:
    levels[PORTFOLIO] = twr.reindex(days)
for name, s in bench_cols.items():
    levels[name] = s.reindex(s.index.union(days)).sort_index().ffill().reindex(days)
levels = levels.dropna(axis=1, how="all")
if levels.empty:
    st.error("Nothing to show: no price history could be loaded.")
    st.stop()

anchor = twr.index if include_pf else levels.dropna(how="all").index
start = engine.period_start(anchor, period)
rebased = engine.rebase(levels, start)
series = list(rebased.columns)
colors = colors_for([c for c in series if c != PORTFOLIO])
stats = engine.perf_stats(rebased, ref=PORTFOLIO if PORTFOLIO in series else None)
years = (rebased.index[-1] - rebased.index[0]).days / 365.25

# --- Headline numbers -------------------------------------------------------
k = st.columns(4)
if PORTFOLIO in stats.index:
    pf = stats.loc[PORTFOLIO]
    others = stats.drop(PORTFOLIO)
    mwr = engine.money_weighted(daily, start)
    k[0].metric(f"Portfolio return ({period})", pct(pf["total"]),
                f"{pct(pf['cagr'])} per year" if years >= 1 else None, delta_color="off" if years < 1 else "normal",
                help="Time-weighted: what €1 kept in the account the whole period became. "
                     "Deposits do not count as gains.")
    if years < 1 and mwr == mwr:  # don't annualise short periods
        mwr = (1 + mwr) ** years - 1
    k[1].metric("Money-weighted" + (" (per year)" if years >= 1 else ""), pct(mwr),
                help="XIRR of your deposits and today's value. Higher than the time-weighted return "
                     "means your deposits were well timed; lower means they mostly went in before weaker stretches.")
    if len(others):
        beaten = int((others["total"] < pf["total"]).sum())
        best = others["total"].idxmax()
        k[2].metric("Benchmarks beaten", f"{beaten} of {len(others)}",
                    f"{(pf['total'] - others.loc[best, 'total']) * 100:+.1f} pp vs {best}",
                    help="Over the selected period. The delta is the gap in total return, in percentage points, "
                         "to the best selected benchmark.")
    k[3].metric("Max drawdown", pct(pf["max_dd"], sign=False),
                help="Largest fall from a peak during the period, time-weighted.")
else:
    best, worst = stats["total"].idxmax(), stats["total"].idxmin()
    k[0].metric(f"Best ({period})", best, pct(stats.loc[best, "total"]))
    k[1].metric(f"Worst ({period})", worst, pct(stats.loc[worst, "total"]))
    calm = stats["vol"].idxmin()
    k[2].metric("Calmest", calm, f"{pct(stats.loc[calm, 'vol'], sign=False)} volatility", delta_color="off")
    deep = stats["max_dd"].idxmin()
    k[3].metric("Deepest drawdown", deep, pct(stats.loc[deep, "max_dd"], sign=False), delta_color="off")

# --- Main comparison chart --------------------------------------------------
st.subheader("Cumulative return")
st.caption(f"Everything in EUR, rebased to 0% on {rebased.index[0]:%d %b %Y}. "
           "Benchmarks include reinvested dividends where the source allows it (see *Data & sources*).")
fig = go.Figure()
for name in [c for c in series if c != PORTFOLIO] + ([PORTFOLIO] if PORTFOLIO in series else []):
    y = (rebased[name] - 1) * 100
    fig.add_scatter(x=y.index, y=y, name=name, mode="lines", line=_line(name, colors[name]),
                    hovertemplate="%{y:+.1f}%<extra>" + name + "</extra>")
    last = y.dropna()
    if len(last) and len(series) <= 4:  # direct labels at the line ends when it stays readable
        fig.add_annotation(x=last.index[-1], y=last.iloc[-1], text=f"{name} {last.iloc[-1]:+.0f}%",
                           showarrow=False, xanchor="left", xshift=6, font=dict(size=12))
fig.add_hline(y=0, line=dict(color=NEUTRAL, width=1))
fig = _layout(fig, 480)
if len(series) <= 4:
    fig.update_layout(margin=dict(r=140))
st.plotly_chart(fig, width="stretch")

# --- Stats table ------------------------------------------------------------
tbl = stats.copy()
pct_cols = ["total", "cagr", "vol", "max_dd", "best_month", "worst_month", "pos_months"]
for c in pct_cols:
    tbl[c] = tbl[c] * 100
tbl.insert(0, "series", tbl.index)
cc = st.column_config
col_cfg = {
    "series": cc.TextColumn("", pinned=True),
    "total": cc.NumberColumn("Total", format="%+.1f%%"),
    "cagr": cc.NumberColumn("Per year", format="%+.1f%%", help="Annualised (CAGR). Blank for periods under a year."),
    "vol": cc.NumberColumn("Volatility", format="%.1f%%", help="Annualised standard deviation of weekly returns"),
    "ratio": cc.NumberColumn("Return / risk", format="%.2f",
                             help="Per-year return divided by volatility (a Sharpe ratio without the risk-free rate)"),
    "max_dd": cc.NumberColumn("Max drawdown", format="%.1f%%"),
    "best_month": cc.NumberColumn("Best month", format="%+.1f%%"),
    "worst_month": cc.NumberColumn("Worst month", format="%+.1f%%"),
    "pos_months": cc.NumberColumn("Months up", format="%.0f%%"),
}
if "corr" in tbl:
    col_cfg["corr"] = cc.NumberColumn("Correlation", format="%.2f",
                                      help="Of your portfolio's weekly returns with this one. 1 = moves in lockstep.")
    col_cfg["beta"] = cc.NumberColumn("Your beta", format="%.2f",
                                      help="How much your portfolio moved per 1% move of this benchmark (weekly). "
                                           "Above 1: you swing harder than it does.")
st.dataframe(tbl, hide_index=True, width="stretch", column_config=col_cfg,
             column_order=["series"] + [c for c in col_cfg if c != "series" and c in tbl])

# --- Deeper views -----------------------------------------------------------
names = ["💶 What if I'd bought the index?", "↕️ Ahead or behind", "📅 Per year",
         "📉 Drawdowns", "🔗 Correlation", "🔍 Data & sources"]
tabs = dict(zip(names, st.tabs(names)))
bench_names = [c for c in series if c != PORTFOLIO]

with tabs[names[0]]:
    if PORTFOLIO not in series or not bench_names:
        st.info("Include your portfolio and pick at least one benchmark to see this.")
    else:
        st.caption("Your actual account value against shadow portfolios: the value at the start of the period, "
                   "and every deposit after it, put into the benchmark on the same day. Same money, same dates. "
                   "This counts your timing too, so it answers *would I have more money?*")
        part = daily[daily.index >= rebased.index[0]]
        fig = go.Figure()
        fig.add_scatter(x=part.index, y=part["invested"], name="Net invested", mode="lines",
                        line=dict(color=NEUTRAL, width=1.5, dash="dot", shape="hv"),
                        hovertemplate="€%{y:,.0f}<extra>Net invested</extra>")
        end_vals = {PORTFOLIO: float(part["value"].iloc[-1])}
        for name in bench_names:
            sh = engine.shadow_portfolio(daily, levels[name].dropna(), rebased.index[0])
            end_vals[name] = float(sh.iloc[-1])
            fig.add_scatter(x=sh.index, y=sh, name=f"All in {name}", mode="lines", line=_line(name, colors[name]),
                            hovertemplate="€%{y:,.0f}<extra>" + name + "</extra>")
        fig.add_scatter(x=part.index, y=part["value"], name="My portfolio", mode="lines",
                        line=_line(PORTFOLIO, PORTFOLIO_COLOR), hovertemplate="€%{y:,.0f}<extra>Portfolio</extra>")
        fig = _layout(fig, 420, pct_axis=False)
        fig.update_layout(yaxis=dict(tickprefix="€", tickformat=",.0f"))
        st.plotly_chart(fig, width="stretch")
        cols = st.columns(min(len(end_vals), 4))
        mine = end_vals[PORTFOLIO]
        for i, name in enumerate(bench_names[:4]):
            diff = mine - end_vals[name]
            cols[i].metric(f"All in {name}", eur(end_vals[name]),
                           f"you are {eur(abs(diff))} {'ahead' if diff >= 0 else 'behind'}",
                           delta_color="normal" if diff >= 0 else "inverse")

with tabs[names[1]]:
    if PORTFOLIO not in series or not bench_names:
        st.info("Include your portfolio and pick at least one benchmark to see this.")
    else:
        st.caption("Your time-weighted growth relative to each benchmark. Above 0: you are ahead since the start "
                   "of the period. A rising line means you are pulling away right then.")
        fig = go.Figure()
        for name in bench_names:
            rel = (rebased[PORTFOLIO] / rebased[name] - 1) * 100
            fig.add_scatter(x=rel.index, y=rel, name=f"vs {name}", mode="lines", line=_line(name, colors[name]),
                            hovertemplate="%{y:+.1f}%<extra>vs " + name + "</extra>")
        fig.add_hline(y=0, line=dict(color=NEUTRAL, width=1))
        st.plotly_chart(_layout(fig, 380), width="stretch")
        if years >= 1.5:
            st.markdown("**Rolling 1-year return**")
            st.caption("Return over the 12 months up to each date. Shows how often, not just whether, you were ahead.")
            roll = (levels.loc[levels.index >= rebased.index[0] - pd.Timedelta(days=365), series]
                    .pipe(lambda d: d / d.shift(365) - 1) * 100).loc[rebased.index[0] + pd.Timedelta(days=365):]
            fig = go.Figure()
            for name in bench_names + [PORTFOLIO]:
                fig.add_scatter(x=roll.index, y=roll[name], name=name, mode="lines", line=_line(name, colors[name]),
                                hovertemplate="%{y:+.1f}%<extra>" + name + "</extra>")
            fig.add_hline(y=0, line=dict(color=NEUTRAL, width=1))
            st.plotly_chart(_layout(fig, 340), width="stretch")
            ahead = {n: float((roll[PORTFOLIO] > roll[n]).loc[roll[[PORTFOLIO, n]].notna().all(axis=1)].mean())
                     for n in bench_names}
            st.caption("Share of 1-year windows where you beat it: "
                       + " · ".join(f"**{n}** {v:.0%}" for n, v in ahead.items()))

with tabs[names[2]]:
    cal = engine.calendar_returns(rebased) * 100
    first_date, last_date = rebased.index[0], rebased.index[-1]
    labels = []
    for y in cal.index:
        if y == last_date.year and last_date < pd.Timestamp(year=y, month=12, day=31):
            labels.append(f"{y} YTD")
        elif y == first_date.year and first_date > pd.Timestamp(year=y - 1, month=12, day=31):
            labels.append(f"{y} (from {first_date:%d %b})")
        else:
            labels.append(str(y))
    fig = go.Figure()
    for name in ([PORTFOLIO] if PORTFOLIO in series else []) + bench_names:
        fig.add_bar(x=labels, y=cal[name], name=name, marker=dict(color=colors[name], cornerradius=4),
                    hovertemplate="%{x}: %{y:+.1f}%<extra>" + name + "</extra>")
    fig.add_hline(y=0, line=dict(color=NEUTRAL, width=1))
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)
    fig = _layout(fig, 400)
    fig.update_layout(hovermode="closest", xaxis=dict(type="category"))  # else "2022" parses as a number
    st.plotly_chart(fig, width="stretch")
    st.caption("Partial years are marked. Pick a longer period above to see more years.")

with tabs[names[3]]:
    st.caption("How far each one was below its previous high. Deep and long dips are what test your nerves.")
    dd = engine.drawdown(rebased) * 100
    fig = go.Figure()
    for name in bench_names + ([PORTFOLIO] if PORTFOLIO in series else []):
        fig.add_scatter(x=dd.index, y=dd[name], name=name, mode="lines", line=_line(name, colors[name]),
                        hovertemplate="%{y:.1f}%<extra>" + name + "</extra>")
    st.plotly_chart(_layout(fig, 380), width="stretch")
    if PORTFOLIO in series:
        pdd = dd[PORTFOLIO]
        trough = pdd.idxmin()
        peak = rebased.loc[:trough, PORTFOLIO].idxmax()
        rec = rebased.loc[trough:, PORTFOLIO]
        back = rec[rec >= rebased.loc[peak, PORTFOLIO]]
        msg = (f"Your deepest drop in this period: **{pdd.min():.1f}%**, from {peak:%d %b %Y} to {trough:%d %b %Y}. ")
        msg += (f"Recovered on {back.index[0]:%d %b %Y} ({(back.index[0] - peak).days} days peak to peak)."
                if len(back) else f"Not yet recovered; currently {pdd.iloc[-1]:.1f}% below that peak.")
        st.markdown(msg)

with tabs[names[4]]:
    if len(series) < 2:
        st.info("Pick at least two series to compare.")
    else:
        st.caption("Correlation of weekly returns over the period. Low or negative correlation with your "
                   "portfolio means that asset would actually diversify it.")
        corr = engine.weekly_returns(rebased).corr()
        order = ([PORTFOLIO] if PORTFOLIO in series else []) + bench_names
        corr = corr.loc[order, order]
        fig = go.Figure(go.Heatmap(
            z=corr.values, x=order, y=order, zmin=-1, zmax=1,
            colorscale=[[0, "#d95926"], [0.5, "#e7e6e2"], [1, "#256abf"]],
            text=[[f"{v:.2f}" for v in row] for row in corr.values], texttemplate="%{text}",
            hovertemplate="%{y} × %{x}: %{z:.2f}<extra></extra>", xgap=2, ygap=2,
            colorbar=dict(title=None, tickvals=[-1, 0, 1], thickness=12),
        ))
        fig.update_layout(height=max(320, 48 * len(order) + 80), margin=dict(l=0, r=0, t=10, b=0),
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, width="stretch")

with tabs[names[5]]:
    if bench_info:
        src = pd.DataFrame(bench_info)
        src["kind"] = ["Price only" if t.startswith("^") else
                       ("No dividends" if t.endswith(("=F", "-EUR")) else "Total return") for t in src["ticker"]]
        src["note"] = [benches[n].get("note", "") for n in src["name"]]
        late = src[src["first_date"] > rebased.index[0]]
        st.markdown("**Benchmark sources** (Yahoo Finance, converted to EUR with daily FX)")
        st.dataframe(src[["name", "ticker", "currency", "kind", "first_date", "note"]], hide_index=True,
                     width="stretch", column_config={"name": "Benchmark", "ticker": "Symbol", "currency": "Quoted in",
                                                     "kind": "Dividends", "first_date": cc.DateColumn("History from"),
                                                     "note": "What it is"})
        if len(late):
            st.caption("Starts later than the period, so rebased from its first day: " + ", ".join(late["name"]))
    if hist is not None:
        st.markdown("**How complete your own price history is**")
        share = daily["priced_share"].loc[rebased.index[0]:]
        avg = float((share * daily["holdings"]).sum() / daily["holdings"].loc[share.index].sum()) \
            if daily["holdings"].loc[share.index].sum() else 1.0
        st.caption(f"{avg:.0%} of your holdings' value over this period is backed by real daily prices. "
                   "Positions without a ticker are held at their last trade price, so they look flat between "
                   "trades and their gain or loss lands on the day you sold.")
        cov = hist["coverage"]
        if len(cov):
            cov = cov.assign(name=[meta.get(i, {}).get("short", "") for i in cov.index],
                             share=cov["share_of_history"] * 100).reset_index(names="isin")
            st.dataframe(cov[["isin", "name", "share", "held_days", "no_data_at_all"]].head(15), hide_index=True,
                         width="stretch", column_config={
                             "isin": "ISIN", "name": "Ticker",
                             "share": cc.ProgressColumn("Share of history", format="%.1f%%", min_value=0,
                                                        max_value=float(max(cov["share"].max(), 1))),
                             "held_days": cc.NumberColumn("Days held"),
                             "no_data_at_all": cc.CheckboxColumn("No ticker"),
                         })
            st.caption("To fill a gap, add the ISIN with its Yahoo `tickers` to `shared-config/isin_map.json`. "
                       "Closed positions only need `short` and `tickers`; the Holdings page ignores them.")
        odd = hist["factors"]
        if len(odd):
            st.caption("Price history rescaled to match your trade prices (usually a later split): "
                       + ", ".join(f"{meta.get(i, {}).get('short', i)} ×{f:.3g}" for i, f in odd.items()))
        fixes = pd.concat([hist["repairs"]] + bench_repairs, ignore_index=True)
        if len(fixes):
            fixes = fixes.assign(name=[meta.get(c, {}).get("short", c) for c in fixes["column"]])
            st.markdown("**Price data repaired**")
            st.caption("Yahoo data errors removed before calculating: one-day spikes that revert, and jumps of "
                       "more than 3x in a day (a split adjusted wrongly). Check these if a number looks off.")
            st.dataframe(fixes[["name", "date", "kind", "ratio"]], hide_index=True, width="stretch",
                         column_config={"name": "Series", "date": cc.DateColumn("Date"), "kind": "What",
                                        "ratio": cc.NumberColumn("Ratio to expected", format="%.3g")})

        st.markdown("**Biggest single-day moves of your portfolio**")
        st.caption("The days with the largest time-weighted return, and the position that drove most of it. "
                   "*Gain* is the value change minus that day's deposit, so deposits never count as return. "
                   "A huge move driven by one small position points at bad price data for it.")
        moves = engine.biggest_moves(hist, n=10, start=rebased.index[0])
        moves = moves.assign(driver=[meta.get(i, {}).get("short", i) for i in moves["top_isin"]],
                             ret=moves["return"] * 100)
        st.dataframe(moves[["date", "ret", "gain_eur", "deposit", "value", "driver", "top_eur"]], hide_index=True,
                     width="stretch", column_config={
                         "date": cc.DateColumn("Date"), "ret": cc.NumberColumn("Return", format="%+.2f%%"),
                         "gain_eur": cc.NumberColumn("Gain", format="euro"),
                         "deposit": cc.NumberColumn("Deposit that day", format="euro"),
                         "value": cc.NumberColumn("Value at close", format="euro"),
                         "driver": "Biggest driver", "top_eur": cc.NumberColumn("Its gain", format="euro"),
                     })
    with st.expander("How these returns are calculated"):
        st.markdown(
            "- **Time-weighted return (TWR)** chains each day's return, `value today / (value yesterday + deposits "
            "today) − 1`. Deposits are not gains, and when you deposited does not matter, so it compares fairly "
            "with an index.\n"
            "- **Portfolio value** = shares × daily close in EUR + cash. Cash is rebuilt from the log "
            "(deposits − buys + sells + dividends), so idle cash counts, as it would in reality.\n"
            "- **Money-weighted return** (XIRR) does count your timing: it is the yearly rate that turns your "
            "deposits into today's value.\n"
            "- **Benchmarks** use accumulating ETFs where possible, so their dividends are reinvested, like yours. "
            "Price-only indices (symbols starting with `^`) understate returns by roughly the dividend yield.\n"
            "- **Volatility, correlation and beta** use weekly returns, which avoids mismatches between markets "
            "that close at different times.\n"
            "- Not included: DEGIRO's yearly connection fees and interest, which the cleaner leaves out."
        )