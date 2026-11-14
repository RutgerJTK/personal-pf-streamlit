import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import engine
from market_data import fetch_quotes, load_isin_map
from s3_fetcher import load_transactions_or_stop

# Categorical slots 1 and 2 of the dashboard palette: identity follows the
# group everywhere on this page (blue = stocks, orange = ETFs).
GROUP_COLORS = {"Stock": "#2a78d6", "ETF": "#eb6834", "Unmapped": "#8a8986"}
SINGLE = "#2a78d6"
GROUP_LABEL = {"Stock": "stocks", "ETF": "ETFs", "Unmapped": "unmapped"}


def eur(x: float, sign: bool = False) -> str:
    return f"{'+' if sign and x > 0 else ''}€{x:,.2f}".replace("+-", "-").replace("€-", "-€")


def pct(x: float, sign: bool = False, digits: int = 1) -> str:
    return f"{x:+.{digits}%}" if sign else f"{x:.{digits}%}"


def _layout(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=10, b=0),
                      legend=dict(orientation="h", y=-0.12, title=None))
    return fig


st.title("💼 Holdings")

df = load_transactions_or_stop()
meta = load_isin_map()
positions = engine.open_positions(df)
if positions.empty:
    st.info("No open positions in the transaction log.")
    st.stop()

# --- Controls ---------------------------------------------------------------
c1, c2, c3 = st.columns([2, 2, 1], vertical_alignment="bottom")
basis_label = c1.segmented_control(
    "Weigh positions by", ["Market value", "Cost basis"], default="Market value",
    help="Market value uses live prices (15 min cache). Cost basis is what you paid, incl. fees.",
)
basis = "cost_basis" if basis_label == "Cost basis" else "market_value"
look_through = c2.toggle(
    "Look through ETFs", value=True,
    help="Split each ETF over the sectors and regions it holds, instead of counting it as one block.",
)
if c3.button("🔄 Prices", help="Fetch fresh prices"):
    fetch_quotes.clear()

tickers = {i: tuple(meta.get(i, {}).get("tickers", ())) for i in positions.index}
quotes = fetch_quotes({i: t for i, t in tickers.items() if t})
h = engine.build_holdings(positions, meta, quotes, basis=basis)

unmapped = h.index[~h["mapped"]].tolist()
if unmapped:
    st.warning(f"No metadata for {', '.join(unmapped)}. Add them to `shared-config/isin_map.json`.")
unpriced = h.loc[~h["priced"], "short"].tolist()
if len(unpriced) == len(h):
    st.warning("Live prices are unavailable right now, so all values below are at cost basis.")
elif unpriced:
    st.warning(f"No live price for {', '.join(unpriced)}; valued at cost basis.")

# --- Headline numbers -------------------------------------------------------
priced = h[h["priced"]]
market_total = h["market_value"].sum()
cost_total = h["cost_basis"].sum()
unreal = priced["unrealized"].sum()
unreal_pct = unreal / priced["cost_basis"].sum() if len(priced) else 0.0
day = h["day_change_eur"].sum(min_count=1)
day_base = (h["market_value"] - h["day_change_eur"]).where(h["day_change_eur"].notna()).sum()
conc = engine.concentration(h)
n_by_group = h["group"].value_counts()
w_by_group = h.groupby("group")["weight"].sum()

k = st.columns(4)
k[0].metric("Portfolio value", eur(market_total),
            help="Open positions at live prices. Cash in your DEGIRO account is not included.")
k[1].metric("Unrealized P/L", eur(unreal, sign=True), pct(unreal_pct, sign=True),
            help=f"Against an average-cost basis of {eur(cost_total)}. Positions without a price are left out.")
k[2].metric("Today", eur(day, sign=True) if day == day else "–",
            pct(day / day_base, sign=True, digits=2) if day == day and day_base else None,
            help="Change since the previous close, per each listing's own trading day.")
k[3].metric("Positions", len(h),
            help=" · ".join(f"{n} {GROUP_LABEL.get(g, g)}" for g, n in n_by_group.items()))

k = st.columns(4)
k[0].metric("Stocks / ETFs", f"{w_by_group.get('Stock', 0):.0%} / {w_by_group.get('ETF', 0):.0%}",
            help="ETCs and ETPs (gold, bitcoin) count as ETFs here.")
top = h.iloc[0]
_largest = dict(label="Largest position", value=top["short"], delta=f"{top['weight']:.1%} of portfolio",
                delta_color="off",
                help=f"{top['name']}: {top['weight']:.1%} of the portfolio, "
                     f"{top['weight_in_group']:.1%} of your {GROUP_LABEL.get(top['group'], '')}.")
try:
    k[1].metric(**_largest, delta_arrow="off")
except TypeError:  # Streamlit without delta_arrow
    k[1].metric(**_largest)
k[2].metric("Top 5 weight", pct(conc["top5"]), help="Share of the portfolio in your five largest positions.")
k[3].metric("Effective # of positions", f"{conc['effective_n']:.1f}",
            help=f"1 / Σ(weight²). With {len(h)} equal-sized positions this would be {len(h)}; "
                 "lower means a few positions dominate.")

st.caption(f"Weights based on **{basis_label.lower()}**.")

# --- Allocation treemap -----------------------------------------------------
st.subheader("Allocation")
st.caption("Box size = weight. Each box shows its share of the whole portfolio and of its group (stocks or ETFs).")
tm = h.reset_index().assign(group_label=lambda d: d["group"].map({"Stock": "Stocks", "ETF": "ETFs"}).fillna(d["group"]))
fig = px.treemap(
    tm, path=[px.Constant("Portfolio"), "group_label", "short"], values="value",
    color="group_label",
    color_discrete_map={"Stocks": GROUP_COLORS["Stock"], "ETFs": GROUP_COLORS["ETF"],
                        "Unmapped": GROUP_COLORS["Unmapped"], "(?)": "rgba(128,128,128,0.15)"},
    custom_data=["name"],
)
fig.update_traces(
    texttemplate="<b>%{label}</b><br>%{percentRoot:.1%} of total<br>%{percentParent:.1%} of %{parent}",
    hovertemplate="<b>%{label}</b><br>€%{value:,.2f}<br>%{percentRoot:.1%} of portfolio"
                  "<br>%{percentParent:.1%} of %{parent}<extra></extra>",
    marker=dict(line=dict(width=2)), root_color="rgba(0,0,0,0)", tiling=dict(pad=3),
)
st.plotly_chart(_layout(fig, 460), width="stretch")

# --- Sector exposure --------------------------------------------------------
st.subheader("Sector exposure")
sec = engine.exposure(h, meta, "sector", look_through=look_through)
totals = sec.groupby("sector", observed=True)["weight"].sum()
fig = go.Figure()
for g in [g for g in GROUP_COLORS if g in set(sec["group"])]:
    part = sec[sec["group"] == g]
    fig.add_bar(
        y=part["sector"].astype(str), x=part["weight"], name=f"via {GROUP_LABEL[g]}",
        orientation="h", marker=dict(color=GROUP_COLORS[g], line=dict(width=1, color="rgba(255,255,255,0.9)")),
        customdata=part["value"],
        hovertemplate="%{y}<br>%{x:.1%} of portfolio (€%{customdata:,.0f})<extra>" + f"via {GROUP_LABEL[g]}</extra>",
    )
fig.add_scatter(  # total label at the end of each bar
    y=totals.index.astype(str), x=totals.values, mode="text", text=[f"{v:.1%}" for v in totals.values],
    textposition="middle right", showlegend=False, hoverinfo="skip",
)
fig.update_layout(barmode="stack", bargap=0.35,
                  yaxis=dict(categoryorder="array", categoryarray=list(totals.index.astype(str))[::-1], title=None),
                  xaxis=dict(tickformat=".0%", title=None, range=[0, totals.max() * 1.15], showgrid=True))
st.plotly_chart(_layout(fig, max(300, 32 * len(totals) + 60)), width="stretch")
if look_through:
    st.caption("ETF sector splits are approximate snapshots from the fund factsheets "
               "(dates under *About these numbers* below). Gold and bitcoin are shown as their own asset class.")
else:
    st.caption("Each ETF counted as one block. Turn on *Look through ETFs* to see what they hold.")

# --- Region + currency ------------------------------------------------------
left, right = st.columns(2)
with left:
    st.subheader("Region")
    reg = engine.exposure(h, meta, "region", look_through=look_through)
    reg = reg.groupby("region", observed=True)["weight"].sum().sort_values()
    fig = go.Figure(go.Bar(
        y=reg.index.astype(str), x=reg.values, orientation="h", marker_color=SINGLE,
        text=[f"{v:.1%}" for v in reg.values], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:.1%}<extra></extra>",
    ))
    fig.update_layout(xaxis=dict(tickformat=".0%", title=None, range=[0, reg.max() * 1.25]), yaxis_title=None)
    st.plotly_chart(_layout(fig, max(260, 34 * len(reg) + 40)), width="stretch")
    st.caption("Where the companies are listed/based. QNTM is a global fund.")
with right:
    st.subheader("Underlying currency")
    cur = engine.currency_exposure(h).sort_values()
    fig = go.Figure(go.Bar(
        y=cur.index, x=cur.values, orientation="h", marker_color=SINGLE,
        text=[f"{v:.1%}" for v in cur.values], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:.1%}<extra></extra>",
    ))
    fig.update_layout(xaxis=dict(tickformat=".0%", title=None, range=[0, cur.max() * 1.25]), yaxis_title=None)
    st.plotly_chart(_layout(fig, max(260, 34 * len(cur) + 40)), width="stretch")
    st.caption("Currency risk you carry, even when the ETF itself trades in EUR (e.g. VUSA is USD exposure).")

# --- Positions table --------------------------------------------------------
st.subheader("Positions")
show = st.segmented_control("Show", ["All", "Stocks", "ETFs"], default="All", key="holdings_filter")
view = h if show in (None, "All") else h[h["group"] == ("Stock" if show == "Stocks" else "ETF")]
table = view.assign(
    weight_pct=view["weight"] * 100,
    weight_in_group_pct=view["weight_in_group"] * 100,
    unrealized_pct_100=view["unrealized_pct"] * 100,
    day_pct_100=view["day_change_pct"] * 100,
)[[
    "short", "name", "kind", "sector", "quantity", "avg_cost", "price_eur", "market_value",
    "weight_pct", "weight_in_group_pct", "cost_basis", "unrealized", "unrealized_pct_100",
    "day_pct_100", "dividends", "first_buy",
]].reset_index()

cc = st.column_config
st.dataframe(
    table, hide_index=True, width="stretch", height=35 * (len(table) + 1) + 3,
    column_config={
        "isin": cc.TextColumn("ISIN"),
        "short": cc.TextColumn("Ticker", pinned=True),
        "name": cc.TextColumn("Name"),
        "kind": cc.TextColumn("Type"),
        "sector": cc.TextColumn("Sector"),
        "quantity": cc.NumberColumn("Qty", format="%.4g"),
        "avg_cost": cc.NumberColumn("Avg cost", format="euro", help="Per share, incl. fees"),
        "price_eur": cc.NumberColumn("Price", format="euro", help="Live, converted to EUR"),
        "market_value": cc.NumberColumn("Value", format="euro"),
        "weight_pct": cc.ProgressColumn("% of portfolio", format="%.1f%%", min_value=0,
                                        max_value=float(h["weight"].max() * 100)),
        "weight_in_group_pct": cc.ProgressColumn("% of stocks / ETFs", format="%.1f%%", min_value=0, max_value=100,
                                                 help="Share of all your stocks (for a stock) or all your ETFs (for an ETF)"),
        "cost_basis": cc.NumberColumn("Cost basis", format="euro"),
        "unrealized": cc.NumberColumn("Unrealized €", format="euro"),
        "unrealized_pct_100": cc.NumberColumn("Unrealized %", format="%.1f%%"),
        "day_pct_100": cc.NumberColumn("Today", format="%.2f%%"),
        "dividends": cc.NumberColumn("Dividends", format="euro", help="All-time, after withholding tax"),
        "first_buy": cc.DateColumn("Held since", format="D MMM YYYY",
                                   help="First buy of the current holding period"),
    },
)

# --- Notes ------------------------------------------------------------------
with st.expander("About these numbers"):
    st.markdown(
        "- **Cost basis** uses the average-cost method: a sale removes cost in proportion to the shares sold. "
        "Fees and FX costs are included.\n"
        "- **Prices** come from Yahoo Finance via the tickers in `shared-config/isin_map.json` and are cached "
        "for 15 minutes. A position without a price is valued at cost.\n"
        "- **Stocks vs ETFs**: the gold ETC and bitcoin ETP are counted as ETFs.\n"
        "- **Cash** in your DEGIRO account is not part of the portfolio value (the cleaner strips balances)."
    )
    etf_rows = [
        {"ETF": m.get("short", i), "Look-through weights as of": m.get("weights_as_of", "–")}
        for i, m in meta.items() if i in h.index and "sectors" in m
    ]
    if etf_rows:
        st.dataframe(etf_rows, hide_index=True)
    if len(quotes):
        st.caption("Price sources: " + ", ".join(f"{h.loc[i, 'short']} → {quotes.loc[i, 'ticker']}"
                                                 for i in quotes.index if i in h.index))