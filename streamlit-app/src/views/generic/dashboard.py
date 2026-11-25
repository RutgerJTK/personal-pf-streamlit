"""
Generic dashboard: the landing page for anyone's portfolio.

Rules for everything under views/generic/:
  * never import s3_fetcher or read st.secrets["aws"];
  * user data lives in st.session_state only (per browser session), never on
    disk and never as an argument of an @st.cache_data function (that cache is
    shared by all users). Caching on ISINs / tickers is fine.
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import engine
import generic_benchmark as gb
import generic_session as gs
from market_data import fetch_history, fetch_quotes, load_benchmarks
from security_lookup import betas, etf_valuation, fetch_fundamentals, load_etf_proxies, portfolio_ratios

GROUP_COLORS = {"Stock": "#2a78d6", "ETF": "#eb6834"}   # same as the Holdings page
BETA_YEARS = 3


def _on_upload() -> None:
    """Removing every file in the uploader also forgets them for the session."""
    ups = st.session_state.get("t212_uploader") or []
    if ups:
        gs.set_files([(f.name, f.getvalue()) for f in ups])
    else:
        gs.clear()


def eur(x: float, sign: bool = False) -> str:
    return f"{'+' if sign and x > 0 else ''}€{x:,.2f}".replace("€-", "-€")


st.title("📊 Portfolio Dashboard")

# --- 1. Upload -----------------------------------------------------------------
with st.container(border=True):
    st.markdown("**Upload your Trading 212 history export**")
    st.caption(
        "In Trading 212: *History → Export → CSV*. One export covers at most 12 months, "
        "so upload one file per year back to when you opened the account. "
        "Files are only read into this browser session's memory: nothing is saved, and "
        "they're gone when you close the tab or go back to the selection screen."
    )
    uploads = st.file_uploader("CSV files", type="csv", accept_multiple_files=True,
                               label_visibility="collapsed", key="t212_uploader", on_change=_on_upload)
    if uploads:
        gs.set_files([(f.name, f.getvalue()) for f in uploads])
    current = gs.files()
    if current and not uploads:  # files kept from earlier on this session (or the example)
        names = ", ".join(n for n, _ in current)
        c1, c2 = st.columns([5, 1], vertical_alignment="center")
        c1.caption(f"Using: {names}")
        if c2.button("Remove", type="tertiary", help="Forget these files"):
            gs.clear()
            st.rerun()
    elif st.button("No export at hand? Load example data", type="tertiary"):
        gs.load_sample()
        st.rerun()

if not gs.files():
    st.stop()
if gs.is_sample():
    st.info("Showing **example data**: 5 shares each of six stocks and ETFs, bought on 2 March 2026. "
            "Upload your own export above to replace it.")

parsed = gs.parsed_or_stop(show_link=False)
df, rep = parsed.transactions, parsed.report
first, last = df["datetime"].min(), df["datetime"].max()
with st.expander(f"Import details: {len(df)} transactions, "
                 f"{first:%d %b %Y} – {last:%d %b %Y}",
                 expanded=bool(rep.warnings or rep.corporate)):
    st.write(f"{rep.files} file(s), {rep.rows_read} rows read, {rep.duplicates} duplicate(s) removed.")
    counts = df["action"].value_counts()
    st.write(", ".join(f"{n} × {a.lower()}" for a, n in counts.items()))
    for w in rep.warnings:
        st.warning(w)
    if rep.corporate:
        st.warning(
            "Corporate actions found (" + ", ".join(f"{a}: {n}" for a, n in rep.corporate.items())
            + "). These aren't processed yet, so share counts of the affected positions may be off."
        )
    if rep.skipped:
        st.caption("Ignored row types: " + ", ".join(f"{a} ({n})" for a, n in rep.skipped.items()))

positions = engine.open_positions(df)
if positions.empty:
    st.info("No open positions in these files.")
    st.stop()

# --- 2. Match ISINs to Yahoo symbols ---------------------------------------------
res = gs.resolve(parsed, positions.index)
overrides = gs.overrides()
quotes = fetch_quotes({i: c for i, c in res.candidates.items() if c})
h = engine.build_holdings(positions, res.meta, quotes)
h["symbol"] = quotes["ticker"].reindex(h.index)

unpriced = h.index[~h["priced"]]
with st.expander(f"Price sources: {len(h) - len(unpriced)} of {len(h)} matched",
                 expanded=len(unpriced) > 0):
    st.caption(
        "Each ISIN is looked up on Yahoo Finance. If a price is missing or wrong, type the "
        "right Yahoo symbol in the last column (search the ISIN on finance.yahoo.com; European "
        "listings end in e.g. `.AS` for Amsterdam or `.DE` for Xetra)."
    )
    src = pd.DataFrame({
        "isin": h.index,
        "name": h["name"].values,
        "t212": h["short"].values,
        "price": [f"{quotes.loc[i, 'currency']} {quotes.loc[i, 'price_local']:,.2f}"
                  if i in quotes.index else "not found" for i in h.index],
        "yahoo": [overrides.get(i) or (h.loc[i, "symbol"] if h.loc[i, "priced"] else "")
                  for i in h.index],
    })
    edited = st.data_editor(
        src, hide_index=True, width="stretch", key="source_editor",
        disabled=["isin", "name", "t212", "price"],
        column_config={
            "isin": "ISIN", "name": "Name", "t212": "Trading 212 ticker", "price": "Last price",
            "yahoo": st.column_config.TextColumn("Yahoo symbol", help="Edit to override"),
        },
    )
    changed = edited["yahoo"].fillna("").str.strip().str.upper() != src["yahoo"].str.upper()
    if changed.any():
        for isin, sym in zip(edited.loc[changed, "isin"], edited.loc[changed, "yahoo"]):
            sym = (sym or "").strip().upper()
            if sym:
                overrides[isin] = sym
            else:
                overrides.pop(isin, None)
        st.rerun()
    if len(unpriced):
        st.warning(f"No price for {', '.join(h.loc[unpriced, 'short'])}: valued at what you paid.")

# --- 3. Key numbers ----------------------------------------------------------------
summ = engine.summary(df)
invested = summ["net_invested"]
value = float(h["market_value"].sum()) + parsed.cash
pnl = value - invested
c1, c2, c3 = st.columns(3)
c1.metric("Net invested", eur(invested),
          help="Deposits minus withdrawals. Reinvested dividends and sale proceeds don't count as new money.")
c2.metric("Current value", eur(value),
          help=f"Holdings at live prices (15 min cache) plus {eur(parsed.cash)} cash.")
c3.metric("Profit / loss", eur(pnl, sign=True),
          delta=f"{pnl / invested:+.2%}" if invested > 0 else None,
          help="Current value minus net invested: price gains, dividends, interest, minus all costs.")

# --- 4. Versus the market -------------------------------------------------------------
benches = load_benchmarks()
st.subheader("Versus the market")
m1, m2 = st.columns([4, 3], vertical_alignment="bottom")
chosen = m1.multiselect(
    "Compare against", list(benches), default=[n for n in gb.DEFAULT_BENCH if n in benches],
    max_selections=gb.MAX_BENCH, key="dash_bench", placeholder="Pick indices, commodities, crypto…",
    format_func=lambda n: f"{n} · {benches[n].get('group', '')}",
)
period = m2.segmented_control("Period", list(engine.PERIODS), default="All", key="dash_period") or "All"
cmp = gb.compare(parsed, chosen, benches)
if cmp is None:
    st.info("No deposits found, so there is no return to compare yet.")
else:
    if cmp.failed:
        st.warning(f"No price history for {', '.join(cmp.failed)} right now (Yahoo Finance did not answer).")
    rebased = engine.rebase(cmp.levels, engine.period_start(cmp.twr.index, period))
    colors = gb.colors_for([c for c in rebased.columns if c != gb.PORTFOLIO], "dash_bench_colors")
    st.caption(f"Time-weighted return in EUR, rebased to 0% on {rebased.index[0]:%d %b %Y}: deposits don't "
               "count as gains, so it compares fairly with an index. Benchmarks include reinvested dividends "
               "where the source allows it.")
    st.plotly_chart(gb.cumulative_figure(rebased, colors, height=420), width="stretch")
    st.page_link("views/generic/benchmark.py", label="Returns per year, drawdowns and more on the Benchmark page",
                 icon="🏁")

# --- 5. Valuation -------------------------------------------------------------------
eq_benches = [n for n, b in benches.items() if b.get("group") == "Equity"] or list(benches)
t1, t2 = st.columns([3, 2], vertical_alignment="bottom")
t1.subheader("Valuation")
beta_vs = t2.selectbox("Beta measured against", eq_benches,
                       index=eq_benches.index("MSCI World") if "MSCI World" in eq_benches else 0,
                       help=f"Beta = how much a position moved per 1% move of this index, from "
                            f"weekly returns in euros over the last {BETA_YEARS} years.")

# P/E and P/B: stocks from Yahoo's company data, ETFs from their holdings (or a proxy)
stock_syms = tuple(sorted(h.loc[h["group"] == "Stock", "symbol"].dropna().unique()))
fund = fetch_fundamentals(stock_syms) if stock_syms else pd.DataFrame(columns=["pe", "pb", "sector"])
rules = load_etf_proxies()
val = pd.DataFrame(index=h.index, columns=["pe", "pb", "source"], dtype=object)
for isin, r in h.iterrows():
    sym = r["symbol"] if isinstance(r["symbol"], str) else None
    if r["group"] == "ETF":
        v = etf_valuation(sym, [r["name"], res.meta[isin].get("yahoo_name", "")], rules)
        val.loc[isin] = [v["pe"], v["pb"], v["source"]]
    elif sym and sym in fund.index:
        val.loc[isin] = [fund.loc[sym, "pe"], fund.loc[sym, "pb"], "Yahoo"]
    else:
        val.loc[isin] = [None, None, "no price source"]
h["pe"] = pd.to_numeric(val["pe"], errors="coerce")
h["pb"] = pd.to_numeric(val["pb"], errors="coerce")
h["ratio_source"] = val["source"]

# Beta: measured from weekly price history against the chosen index
start = (pd.Timestamp.today() - pd.DateOffset(years=BETA_YEARS) - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
own_px, _ = fetch_history({i: res.candidates[i] for i in h.index if res.candidates.get(i)}, start, adjusted=True)
bench_px, _ = fetch_history({beta_vs: tuple(benches[beta_vs]["tickers"])}, start, adjusted=True)
if beta_vs in bench_px and not own_px.empty:
    b = betas(own_px, bench_px[beta_vs])
    h["beta"] = b["beta"].reindex(h.index)
else:
    h["beta"] = float("nan")
    st.caption(f"Beta unavailable: no price history for {beta_vs} or your holdings right now.")

ratios = portfolio_ratios(h["market_value"].where(h["priced"]), h[["pe", "pb", "beta"]])


def _ratio(col: str, label: str, fmt: str, how: str) -> None:
    v, cov = ratios.get(col), ratios.get(f"{col}_coverage", 0.0)
    st.metric(label, fmt.format(v) if v is not None else "–",
              help=f"{how} Based on {cov:.0%} of your portfolio's value.")
    st.caption(f"covers {cov:.0%} of portfolio")


v1, v2, v3 = st.columns(3)
with v1:
    _ratio("pe", "Portfolio P/E", "{:.1f}",
           "Trailing price/earnings, value-weighted harmonic mean (total value over total earnings). "
           "ETFs count with the P/E of their holdings. Loss-making companies are left out.")
with v2:
    _ratio("pb", "Portfolio P/B", "{:.1f}",
           "Price/book value, value-weighted harmonic mean. ETFs count with the P/B of their holdings. "
           "P/B says little for tech companies, whose value is mostly not on the balance sheet.")
with v3:
    _ratio("beta", f"Beta vs {beta_vs}", "{:.2f}",
           f"Value-weighted average of each position's beta: 1.0 moves in line with {beta_vs}, "
           "above 1 swings harder.")

# --- 6. Holdings ----------------------------------------------------------------------
st.subheader("Holdings")
table = pd.DataFrame({
    "Name": h["name"], "Ticker": h["short"], "Type": h["group"], "Shares": h["quantity"],
    "Avg cost": h["avg_cost"], "Price": h["price_eur"], "Value": h["market_value"],
    "P/L": h["unrealized"], "P/L %": h["unrealized_pct"], "Weight": h["weight"],
    "P/E": h["pe"], "P/B": h["pb"], "Beta": h["beta"], "Ratios from": h["ratio_source"],
})
money = st.column_config.NumberColumn(format="€%.2f")
st.dataframe(
    table, hide_index=True, width="stretch",
    column_config={
        "Shares": st.column_config.NumberColumn(format="%.4g"),
        "Avg cost": money, "Price": money, "Value": money, "P/L": money,
        "P/L %": st.column_config.NumberColumn(format="percent"),
        "Weight": st.column_config.NumberColumn(format="percent"),
        "P/E": st.column_config.NumberColumn(format="%.1f"),
        "P/B": st.column_config.NumberColumn(format="%.1f"),
        "Beta": st.column_config.NumberColumn(format="%.2f", help=f"Against {beta_vs}"),
        "Ratios from": st.column_config.TextColumn(
            help="Where P/E and P/B come from. ETFs without their own data use a US ETF "
                 "tracking the same index (edit the list in shared-config/etf_proxies.json)."),
    },
)
st.caption("Prices are in euros. Avg cost includes Trading 212's FX fee.")

# --- 7. Allocation ----------------------------------------------------------------------
st.subheader("Allocation")
a1, a2 = st.columns([2, 1])

by_pos = h.sort_values("weight")
fig = go.Figure()
for group, color in GROUP_COLORS.items():
    part = by_pos[by_pos["group"] == group]
    if part.empty:
        continue
    fig.add_bar(
        y=part["short"], x=part["weight"], orientation="h", name=f"{group}s",
        marker=dict(color=color, cornerradius=4), customdata=part[["name", "market_value"]],
        hovertemplate="<b>%{customdata[0]}</b><br>%{x:.1%} · €%{customdata[1]:,.2f}<extra></extra>",
    )
height = max(220, 36 * len(by_pos) + 60)
fig.update_layout(
    height=height, margin=dict(l=0, r=0, t=10, b=0), bargap=0.3,
    yaxis=dict(categoryorder="array", categoryarray=list(by_pos["short"]), title=None),
    xaxis=dict(tickformat=".0%", title=None, gridcolor="rgba(128,128,128,0.2)"),
    legend=dict(orientation="h", y=-0.15, title=None),
)
a1.markdown("**By position**")
a1.plotly_chart(fig, width="stretch")

groups = h.groupby("group")["market_value"].sum()
groups = groups / groups.sum()
fig2 = go.Figure()
for group, color in GROUP_COLORS.items():
    if group in groups:
        fig2.add_bar(x=[f"{group}s"], y=[groups[group]], name=f"{group}s",
                     marker=dict(color=color, cornerradius=4), text=[f"{groups[group]:.0%}"],
                     textposition="outside",
                     hovertemplate=f"{group}s: %{{y:.1%}}<extra></extra>")
fig2.update_layout(
    height=height, margin=dict(l=0, r=0, t=10, b=0), bargap=0.35,
    showlegend=False, yaxis=dict(tickformat=".0%", range=[0, 1.1], title=None,
                                 gridcolor="rgba(128,128,128,0.2)"),
)
a2.markdown("**Stocks vs ETFs**")
a2.plotly_chart(fig2, width="stretch")