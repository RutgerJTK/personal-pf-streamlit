import plotly.graph_objects as go
import streamlit as st

import engine
from s3_fetcher import load_transactions_or_stop, refresh

st.title("🏠 Overview")

df = load_transactions_or_stop()
s = engine.summary(df)

c1, c2, c3 = st.columns(3)
c1.metric("Net invested", f"€{s['net_invested']:,.2f}", help="Your own money put in via iDEAL deposits. Reinvested dividends and sale proceeds are not counted.")
c2.metric("Dividends received", f"€{s['dividends']:,.2f}", help="After withholding tax.")
c3.metric("Open positions", s["open_positions"])

st.subheader("Money in, dividends back")
flows = engine.cumulative_cashflows(df)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=flows.index, y=flows["net_invested"], name="Net invested",
    mode="lines", line=dict(shape="hv", width=2.5),
    hovertemplate="%{x|%d %b %Y}<br>€%{y:,.2f}<extra>Net invested</extra>",
))
fig.add_trace(go.Scatter(
    x=flows.index, y=flows["dividends_cum"], name="Cumulative dividends",
    mode="lines", line=dict(shape="hv", width=2.5, dash="dot"),
    hovertemplate="%{x|%d %b %Y}<br>€%{y:,.2f}<extra>Dividends</extra>",
))
fig.update_layout(
    hovermode="x unified",
    yaxis=dict(tickprefix="€", tickformat=",.0f", title=None),
    xaxis=dict(title=None),
    legend=dict(orientation="h", y=-0.15),
    margin=dict(l=0, r=0, t=10, b=0),
    height=420,
)
st.plotly_chart(fig, width="stretch")

st.caption(
    f"Data from {s['first_date']:%d %b %Y} to {s['last_date']:%d %b %Y}, "
    "straight from your transaction log (no market prices yet). "
    "Cached for 1 hour."
)
if st.button("🔄 Reload data"):
    refresh()
    st.rerun()