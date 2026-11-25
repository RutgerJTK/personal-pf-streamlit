"""
Generic benchmark page: the uploaded portfolio against indices of choice.

A slimmer copy of views/benchmark.py (the personal page) that reads the
uploaded files from generic_session instead of S3. Same engine functions, so
the numbers are calculated the same way. Rules in views/generic/dashboard.py apply.
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import engine
import generic_benchmark as gb
import generic_session as gs
from generic_benchmark import MAX_BENCH, NEUTRAL, PORTFOLIO, PORTFOLIO_COLOR
from market_data import load_benchmarks

_line, _layout = gb.line, gb.layout


def pct(x: float, sign: bool = True, digits: int = 1) -> str:
    if x is None or x != x:
        return "–"
    return f"{x:+.{digits}%}" if sign else f"{x:.{digits}%}"


def eur(x: float) -> str:
    return f"€{x:,.0f}".replace("€-", "-€")


st.title("🏁 Am I Beating the Market?")

parsed = gs.parsed_or_stop()
df = parsed.transactions
if gs.is_sample():
    st.caption("Example data. Upload your own export on the Dashboard.")
benches = load_benchmarks()
if not benches:
    st.error("No benchmarks configured.")
    st.stop()

# --- Controls ---------------------------------------------------------------
c1, c2 = st.columns([4, 3], vertical_alignment="bottom")
selected = c1.multiselect(
    "Compare against", list(benches), default=[n for n in gb.DEFAULT_BENCH if n in benches],
    max_selections=MAX_BENCH, placeholder="Pick indices, commodities, crypto…",
    format_func=lambda n: f"{n} · {benches[n].get('group', '')}", key="gbench_selected",
    help=f"Up to {MAX_BENCH} at once.",
)
period = c2.segmented_control("Period", list(engine.PERIODS), default="All", key="gbench_period") or "All"

# --- Data -------------------------------------------------------------------
cmp = gb.compare(parsed, selected, benches)
if cmp is None:
    st.info("No deposits found, so there is no return to compare yet.")
    st.stop()
if cmp.failed:
    st.warning(f"No price history for {', '.join(cmp.failed)} right now (Yahoo Finance did not answer).")
levels, daily, twr = cmp.levels, cmp.daily, cmp.twr

start = engine.period_start(twr.index, period)
rebased = engine.rebase(levels, start)
series = list(rebased.columns)
bench_names = [c for c in series if c != PORTFOLIO]
colors = gb.colors_for(bench_names, "gbench_colors")
stats = engine.perf_stats(rebased, ref=PORTFOLIO)
years = (rebased.index[-1] - rebased.index[0]).days / 365.25

# --- Headline numbers -------------------------------------------------------
k = st.columns(4)
pf = stats.loc[PORTFOLIO]
others = stats.drop(PORTFOLIO)
mwr = engine.money_weighted(daily, start)
k[0].metric(f"Portfolio return ({period})", pct(pf["total"]),
            f"{pct(pf['cagr'])} per year" if years >= 1 else None,
            help="Time-weighted: what €1 kept in the account the whole period became. "
                 "Deposits do not count as gains.")
if years < 1 and mwr == mwr:  # don't annualise short periods
    mwr = (1 + mwr) ** years - 1
k[1].metric("Money-weighted" + (" (per year)" if years >= 1 else ""), pct(mwr),
            help="XIRR of your deposits and today's value. Higher than the time-weighted return "
                 "means your deposits were well timed.")
if len(others):
    beaten = int((others["total"] < pf["total"]).sum())
    best = others["total"].idxmax()
    k[2].metric("Benchmarks beaten", f"{beaten} of {len(others)}",
                f"{(pf['total'] - others.loc[best, 'total']) * 100:+.1f} pp vs {best}",
                help="Over the selected period. The delta is the gap in total return, in percentage points, "
                     "to the best selected benchmark.")
k[3].metric("Max drawdown", pct(pf["max_dd"], sign=False),
            help="Largest fall from a peak during the period, time-weighted.")
if years < 0.5:
    st.caption("Less than half a year of history: volatility, beta and correlation below are rough.")

# --- Main comparison chart --------------------------------------------------
st.subheader("Cumulative return")
st.caption(f"Everything in EUR, rebased to 0% on {rebased.index[0]:%d %b %Y}. "
           "Benchmarks include reinvested dividends where the source allows it.")
fig = gb.cumulative_figure(rebased, colors)
st.plotly_chart(fig, width="stretch")

# --- Stats table ------------------------------------------------------------
tbl = stats.copy()
for c in ["total", "cagr", "vol", "max_dd", "best_month", "worst_month", "pos_months"]:
    tbl[c] = tbl[c] * 100
tbl.insert(0, "series", tbl.index)
cc = st.column_config
col_cfg = {
    "series": cc.TextColumn("", pinned=True),
    "total": cc.NumberColumn("Total", format="%+.1f%%"),
    "cagr": cc.NumberColumn("Per year", format="%+.1f%%", help="Annualised. Blank for periods under a year."),
    "vol": cc.NumberColumn("Volatility", format="%.1f%%", help="Annualised standard deviation of weekly returns"),
    "max_dd": cc.NumberColumn("Max drawdown", format="%.1f%%"),
    "worst_month": cc.NumberColumn("Worst month", format="%+.1f%%"),
    "best_month": cc.NumberColumn("Best month", format="%+.1f%%"),
}
if "corr" in tbl:
    col_cfg["corr"] = cc.NumberColumn("Correlation", format="%.2f",
                                      help="Of your portfolio's weekly returns with this one. 1 = moves in lockstep.")
    col_cfg["beta"] = cc.NumberColumn("Your beta", format="%.2f",
                                      help="How much your portfolio moved per 1% move of this benchmark (weekly).")
st.dataframe(tbl, hide_index=True, width="stretch", column_config=col_cfg,
             column_order=["series"] + [c for c in col_cfg if c != "series" and c in tbl])

# --- Deeper views -----------------------------------------------------------
names = ["💶 What if I'd bought the index?", "📅 Per year", "📉 Drawdowns"]
tabs = dict(zip(names, st.tabs(names)))

with tabs[names[0]]:
    if not bench_names:
        st.info("Pick at least one benchmark to see this.")
    else:
        st.caption("Your actual account value against shadow portfolios: the value at the start of the period, "
                   "and every deposit after it, put into the benchmark on the same day. Same money, same dates.")
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
        fig = _layout(fig, 400, pct_axis=False)
        fig.update_layout(yaxis=dict(tickprefix="€", tickformat=",.0f"))
        st.plotly_chart(fig, width="stretch")
        cols = st.columns(min(len(bench_names), 4))
        mine = end_vals[PORTFOLIO]
        for i, name in enumerate(bench_names[:4]):
            diff = mine - end_vals[name]
            cols[i].metric(f"All in {name}", eur(end_vals[name]),
                           f"you are {eur(abs(diff))} {'ahead' if diff >= 0 else 'behind'}",
                           delta_color="normal" if diff >= 0 else "inverse")

with tabs[names[1]]:
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
    for name in [PORTFOLIO] + bench_names:
        fig.add_bar(x=labels, y=cal[name], name=name, marker=dict(color=colors[name], cornerradius=4),
                    hovertemplate="%{x}: %{y:+.1f}%<extra>" + name + "</extra>")
    fig.add_hline(y=0, line=dict(color=NEUTRAL, width=1))
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)
    fig = _layout(fig, 380)
    fig.update_layout(hovermode="closest", xaxis=dict(type="category"))
    st.plotly_chart(fig, width="stretch")

with tabs[names[2]]:
    st.caption("How far each one was below its previous high.")
    dd = engine.drawdown(rebased) * 100
    fig = go.Figure()
    for name in bench_names + [PORTFOLIO]:
        fig.add_scatter(x=dd.index, y=dd[name], name=name, mode="lines", line=_line(name, colors[name]),
                        hovertemplate="%{y:.1f}%<extra>" + name + "</extra>")
    st.plotly_chart(_layout(fig, 360), width="stretch")
    pdd = dd[PORTFOLIO]
    trough = pdd.idxmin()
    peak = rebased.loc[:trough, PORTFOLIO].idxmax()
    rec = rebased.loc[trough:, PORTFOLIO]
    back = rec[rec >= rebased.loc[peak, PORTFOLIO]]
    msg = f"Your deepest drop in this period: **{pdd.min():.1f}%**, from {peak:%d %b %Y} to {trough:%d %b %Y}. "
    msg += (f"Recovered on {back.index[0]:%d %b %Y}." if len(back)
            else f"Not yet recovered; currently {pdd.iloc[-1]:.1f}% below that peak.")
    st.markdown(msg)

with st.expander("How these returns are calculated"):
    st.markdown(
        "- **Time-weighted return (TWR)** chains each day's return, `value today / (value yesterday + deposits "
        "today) − 1`. Deposits are not gains and their timing doesn't matter, so it compares fairly with an index.\n"
        "- **Portfolio value** = shares × daily close in EUR + cash. Cash is rebuilt from your export "
        "(deposits − buys + sells + dividends + interest).\n"
        "- **Money-weighted return** (XIRR) does count your timing: the yearly rate that turns your deposits "
        "into today's value.\n"
        "- **Benchmarks** use accumulating ETFs where possible, so their dividends are reinvested, like yours.\n"
        "- **Volatility, correlation and beta** use weekly returns."
    )