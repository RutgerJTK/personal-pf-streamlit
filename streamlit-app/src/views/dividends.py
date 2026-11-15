import inspect

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import engine
from market_data import fetch_quotes, load_isin_map
from s3_fetcher import load_transactions_or_stop, refresh

# Categorical slots 1 and 2 of the dashboard palette (same as Holdings/Benchmark).
HELD = "#2a78d6"        # blue: holdings you still own
SOLD = "#eb6834"        # orange: holdings sold since
NEUTRAL = "#8a8986"     # 12-month average line
NO_ARROW = {"delta_arrow": "off"} if "delta_arrow" in inspect.signature(st.metric).parameters else {}
# Sequential green ramp (light -> dark) for the dividend calendar. The light end
# stays dark enough (2:1 vs the page) that small payments remain visible.
GREENS = [[0.0, "#78c37f"], [0.25, "#53aa5d"], [0.5, "#368d42"], [0.75, "#22702f"], [1.0, "#0f5323"]]
CAL_RECEIVED = "#22702f"   # dark green: already paid this month
CAL_EXPECTED = "#78c37f"   # light green: still to come
# Diverging pair for growth: blue = raised, red = cut, gray midpoint = unchanged.
RAISED, CUT, FLAT = "#2a78d6", "#e34948", "#b4b2ac"
STATUS_COLOR = {"raised": RAISED, "unchanged": FLAT, "cut": CUT, "unknown": FLAT}


def eur(x: float, digits: int = 2) -> str:
    return f"€{x:,.{digits}f}".replace("€-", "-€")


def pct(x: float, sign: bool = False) -> str:
    return f"{x:+.1%}" if sign else f"{x:.1%}"


def change(now: float, before: float) -> str | None:
    return pct(now / before - 1, sign=True) if before else None


def _layout(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=10, b=0), bargap=0.25,
                      legend=dict(orientation="h", y=-0.15, title=None),
                      yaxis=dict(tickprefix="€", tickformat=",.0f", title=None),
                      xaxis=dict(title=None))
    return fig


st.title("📅 Dividends")

df = load_transactions_or_stop()
meta = load_isin_map()
today = pd.Timestamp.today().normalize()

payments = engine.dividend_payments(df, meta)
if payments.empty:
    st.info("No dividends in the transaction log yet.")
    st.stop()

stats = engine.dividend_stats(df, today)
yearly = engine.dividends_by_year(df, today)
monthly = engine.dividends_by_month(df, today)
forecast = engine.dividend_forecast(df, meta, today)
per_holding = engine.dividends_by_holding(df, forecast, meta, today)
forward = float(forecast["amount"].sum())
rest_of_year = float(forecast.loc[forecast["date"].dt.year == today.year, "amount"].sum())

held = per_holding[per_holding["held"]]
cost = float(held["cost_basis"].sum())
tickers = {i: tuple(meta.get(i, {}).get("tickers", ())) for i in held.index}
quotes = fetch_quotes({i: t for i, t in tickers.items() if t})
prices = quotes["price_eur"].reindex(held.index) if "price_eur" in quotes else pd.Series(index=held.index)
mv = (engine.open_positions(df)["quantity"].reindex(held.index) * prices).fillna(held["cost_basis"])
market_value = float(mv.sum())

# --- Headline numbers ---------------------------------------------------------
k = st.columns(5)
k[0].metric("All-time dividends", eur(stats["total"]), help="Net, after withholding tax.")
k[1].metric(f"{today.year} so far", eur(stats["ytd"]), change(stats["ytd"], stats["ytd_prev"]),
            help=f"Compared with 1 Jan - {today:%d %b} last year ({eur(stats['ytd_prev'])}).")
k[2].metric("Last 12 months", eur(stats["ttm"]), change(stats["ttm"], stats["ttm_prev"]),
            help=f"Compared with the 12 months before ({eur(stats['ttm_prev'])}).")
k[3].metric("Expected next 12 months", eur(forward), f"≈ {eur(forward / 12)} per month", delta_color="off", **NO_ARROW,
            help="Forecast from your current holdings and last year's payments. No dividend growth assumed.")
k[4].metric("Forward yield on cost", pct(forward / cost) if cost else "–",
            f"{pct(forward / market_value)} on market value" if market_value else None,
            delta_color="off", **NO_ARROW,
            help="Expected next-12-month dividends divided by what you paid for the positions you hold now "
                 "(incl. non-payers like gold and bitcoin), and by their current market value.")

# --- Per year: chart + statement ---------------------------------------------
st.subheader("Dividends per year")
held_isins = set(held.index)
by_year = (payments.assign(year=payments["date"].dt.year, still_held=payments["isin"].isin(held_isins))
           .pivot_table(index="year", columns="still_held", values="amount", aggfunc="sum")
           .reindex(index=yearly.index, columns=[True, False]).fillna(0.0))

left, right = st.columns([3, 2], gap="large")
with left:
    fig = go.Figure()
    x = by_year.index.astype(str)
    fig.add_bar(x=x, y=by_year[True], name="Still held", marker_color=HELD,
                hovertemplate="%{x}: %{y:€,.2f}<extra>Still held</extra>")
    fig.add_bar(x=x, y=by_year[False], name="Sold since", marker_color=SOLD,
                hovertemplate="%{x}: %{y:€,.2f}<extra>Sold since</extra>")
    if rest_of_year > 0 and today.year in by_year.index:
        fig.add_bar(x=[str(today.year)], y=[rest_of_year], name="Expected rest of year",
                    marker=dict(color="rgba(42,120,214,0.15)", line=dict(color=HELD, width=1.5),
                                pattern=dict(shape="/", fgcolor=HELD, size=6)),
                    hovertemplate="Still to come in %{x}: %{y:€,.2f}<extra></extra>")
    totals = yearly["received"] + pd.Series(rest_of_year, index=[today.year]).reindex(yearly.index).fillna(0.0)
    fig.add_scatter(x=x, y=totals, mode="text", text=[eur(v, 0) for v in totals], textposition="top center",
                    showlegend=False, hoverinfo="skip", textfont=dict(size=12))
    fig.update_layout(barmode="stack")
    st.plotly_chart(_layout(fig, 380), width="stretch")

with right:
    table = yearly.copy()
    table["label"] = [f"{y} (so far)" if p else str(y) for y, p in zip(table.index, table["partial"])]
    table["growth"] = table["growth"].astype(float) * 100
    table = table.iloc[::-1]
    cc = st.column_config
    st.dataframe(
        table[["label", "received", "growth", "per_month", "payments", "payers"]],
        hide_index=True, width="stretch",
        column_config={
            "label": cc.TextColumn("Year"),
            "received": cc.NumberColumn("Received", format="€%.2f"),
            "growth": cc.NumberColumn("vs prior year", format="%+.1f%%",
                                      help="The current year is compared with the same dates last year."),
            "per_month": cc.NumberColumn("Per month", format="€%.2f"),
            "payments": cc.NumberColumn("Payments"),
            "payers": cc.NumberColumn("Payers", help="Number of different holdings that paid."),
        },
    )
    first = payments["date"].min()
    if rest_of_year > 0:
        st.caption(f"{today.year}: {eur(stats['ytd'])} received, about {eur(rest_of_year)} still expected, "
                   f"so roughly {eur(stats['ytd'] + rest_of_year)} for the full year (hatched bar).")
    full = yearly[~yearly["partial"] & ((yearly.index > first.year) | (first.month == 1))]
    if len(full) >= 2:
        n = len(full) - 1
        cagr = (full["received"].iloc[-1] / full["received"].iloc[0]) ** (1 / n) - 1
        skipped = f" ({first.year} left out: you started in {first:%B})" if full.index[0] > first.year else ""
        st.caption(f"{full.index[0]}–{full.index[-1]}: dividends grew {pct(cagr)} per year on average{skipped}. "
                   "That growth comes from new money as well as dividend raises.")

# --- Per month -----------------------------------------------------------------
st.subheader("Dividends per month")
span = st.segmented_control("Show", ["Last 24 months", "All"], default="Last 24 months",
                            key="div_month_span", label_visibility="collapsed")
m = monthly if span == "All" else monthly.tail(24)
fig = go.Figure()
fig.add_bar(x=m.index, y=m["received"], name="Received", marker_color=HELD,
            hovertemplate="%{x|%b %Y}: %{y:€,.2f}<extra>Received</extra>")
fig.add_scatter(x=m.index, y=m["avg_12m"], name="12-month average", mode="lines",
                line=dict(color=NEUTRAL, width=2),
                hovertemplate="%{x|%b %Y}: %{y:€,.2f} per month<extra>12-month average</extra>")
fig.update_xaxes(dtick="M3" if span != "All" else "M6", tickformat="%b '%y")
st.plotly_chart(_layout(fig, 340), width="stretch")
st.caption("The line is the trailing 12-month total divided by 12: what a typical month pays, "
           "with the quarterly and yearly lumps smoothed out.")

# --- Dividend calendar -----------------------------------------------------------
st.subheader("Dividend calendar: the next 12 months")
cal = engine.dividend_calendar(forecast, today)
if cal.empty:
    st.info("None of your current holdings has paid a dividend recently, so there is nothing to forecast.")
else:
    this_month = payments[payments["date"].dt.to_period("M") == today.to_period("M")]["amount"].sum()
    months = cal.columns
    expected = cal.sum()
    fig = go.Figure()
    fig.add_bar(x=months, y=[this_month] + [0.0] * (len(months) - 1), name="Already received",
                marker_color=CAL_RECEIVED, hovertemplate="%{x|%b %Y}: %{y:€,.2f}<extra>Received</extra>")
    fig.add_bar(x=months, y=expected, name="Expected", marker_color=CAL_EXPECTED,
                hovertemplate="%{x|%b %Y}: %{y:€,.2f}<extra>Expected</extra>")
    fig.update_layout(barmode="stack")
    fig.update_xaxes(dtick="M1", tickformat="%b '%y")
    st.plotly_chart(_layout(fig, 260), width="stretch")

    z = cal.where(cal > 0.005)
    text = z.map(lambda v: "" if pd.isna(v) else f"€{v:,.2f}")
    fig = go.Figure(go.Heatmap(
        z=z.to_numpy(), x=[d.strftime("%b '%y") for d in months], y=list(cal.index),
        text=text.to_numpy(), texttemplate="%{text}", textfont=dict(size=11),
        colorscale=GREENS, showscale=False, xgap=2, ygap=2,
        hovertemplate="%{y}, %{x}: %{z:€,.2f}<extra></extra>",
    ))
    fig.update_layout(height=40 + 30 * len(cal), margin=dict(l=0, r=0, t=10, b=0),
                      yaxis=dict(autorange="reversed", title=None), xaxis=dict(side="top", title=None),
                      plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch")

    soon = forecast[forecast["date"] <= today + pd.Timedelta(days=45)]
    with st.expander(f"Coming up in the next 45 days ({eur(soon['amount'].sum())})", expanded=True):
        st.dataframe(
            soon[["date", "short", "amount", "per_share", "shares"]], hide_index=True, width="stretch",
            column_config={
                "date": st.column_config.DateColumn("Expected around", format="ddd D MMM YYYY"),
                "short": st.column_config.TextColumn("Holding"),
                "amount": st.column_config.NumberColumn("Expected (net)", format="€%.2f"),
                "per_share": st.column_config.NumberColumn("Per share (net)", format="€%.4f"),
                "shares": st.column_config.NumberColumn("Shares now", format="%.4g"),
            },
        )
    st.caption("Dates repeat last year's pay dates, so expect a few days' drift. Amounts are last year's "
               "net dividend per share times the shares you hold now, without growth. A holding with less "
               "than a year of history only repeats what it paid so far (monthly payers are extended).")

# --- Dividend growth ---------------------------------------------------------------
st.subheader("Dividend growth")
growth, gsum = engine.dividend_growth(df, meta, today)
history = engine.dividend_per_share_history(df, meta, today)
if growth.empty:
    st.info("None of your current holdings pays a dividend yet.")
else:
    income_growth = stats["ttm"] / stats["ttm_prev"] - 1 if stats["ttm_prev"] else float("nan")
    k = st.columns(3)
    organic = gsum["organic"]
    k[0].metric("Organic dividend growth", pct(organic, sign=True) if organic == organic else "–",
                help="How much more the last 12 months' dividends paid per share than the same payments a year "
                     "earlier, weighted by the shares you hold now. This is growth from raises alone, "
                     "without buying anything.")
    k[1].metric("Raised · unchanged · cut", f"{gsum['raised']} · {gsum['unchanged']} · {gsum['cut']}",
                f"{gsum['unknown']} not comparable yet" if gsum["unknown"] else None, delta_color="off", **NO_ARROW,
                help="Per holding you own, last 12 months vs a year earlier. Within ±2% counts as unchanged.")
    if income_growth == income_growth and organic == organic:
        k[2].metric("Your dividend income, last 12 months", pct(income_growth, sign=True),
                    f"{pct(organic, sign=True)} from raises, the rest from buying and selling",
                    delta_color="off", **NO_ARROW,
                    help="Total dividends received vs the 12 months before. Mostly driven by new money; "
                         "organic growth is the part that comes in without it.")

    g = growth.dropna(subset=["ttm_growth"]).sort_values("ttm_growth")
    fig = go.Figure(go.Bar(
        x=g["ttm_growth"], y=g["short"], orientation="h",
        marker_color=[STATUS_COLOR[x] for x in g["status"]],
        text=[pct(v, sign=True) + ("" if b == "12 months" else f" ({b[:4]} vs {int(b[:4]) - 1})")
              for v, b in zip(g["ttm_growth"], g["basis"])], textposition="outside", cliponaxis=False,
        customdata=list(zip(g["basis"], g["matched"])),
        hovertemplate="%{y}: %{x:+.1%}<br>%{customdata[0]}, %{customdata[1]} matched payment(s)<extra></extra>",
    ))
    pad = max(abs(g["ttm_growth"].min()), abs(g["ttm_growth"].max())) * 1.25
    fig.update_layout(height=60 + 28 * len(g), margin=dict(l=0, r=0, t=10, b=0), bargap=0.3,
                      xaxis=dict(tickformat="+.0%", range=[min(-0.05, g["ttm_growth"].min() * 1.35), pad],
                                 zeroline=True, zerolinecolor="#8a8986", title=None),
                      yaxis=dict(title=None))
    st.plotly_chart(fig, width="stretch")
    st.caption("Dividend per share, last 12 months vs the same payments a year earlier (or the last two "
               "calendar years where that's all there is). Blue = raised, red = cut, gray = within ±2%.")

    # Drill-down per holding
    options = list(growth.index)
    pick = st.selectbox("Look at one holding", options, format_func=lambda i: growth.loc[i, "short"],
                        index=options.index(growth["income_now"].idxmax()) if growth["income_now"].max() > 0 else 0)
    annual = engine.annual_dividend_per_share(history, pick, df, today)
    pay = history[(history["isin"] == pick) & history["per_share"].notna()]
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("**Dividend per share per year**")
        full = annual[~annual["partial"]]
        part = annual[annual["partial"]]
        fig = go.Figure()
        fig.add_bar(x=full.index.astype(str), y=full["dps"], name="Full year", marker_color=RAISED,
                    text=["" if pd.isna(v) else pct(v, sign=True) for v in full["growth"]], textposition="outside",
                    cliponaxis=False, customdata=full["payments"],
                    hovertemplate="%{x}: %{y:€,.3f} per share<br>%{customdata} payments<extra></extra>")
        fig.add_bar(x=part.index.astype(str), y=part["dps"], name="Partial year",
                    marker=dict(color="rgba(42,120,214,0.15)", line=dict(color=RAISED, width=1.5),
                                pattern=dict(shape="/", fgcolor=RAISED, size=6)),
                    customdata=part["payments"],
                    hovertemplate="%{x} (partial): %{y:€,.3f} per share<br>%{customdata} payments<extra></extra>")
        fig.update_layout(barmode="overlay", xaxis=dict(type="category", categoryorder="category ascending"))
        _layout(fig, 300).update_yaxes(tickformat=",.2f")
        st.plotly_chart(fig, width="stretch")
    with b:
        st.markdown("**Every payment, per share**")
        clear, unclear = pay[pay["clear"]], pay[~pay["clear"]]
        fig = go.Figure()
        fig.add_scatter(x=clear["date"], y=clear["per_share"], mode="lines+markers", name="Payment",
                        line=dict(color=RAISED, width=2), marker=dict(size=8, color=RAISED),
                        customdata=clear["growth"],
                        hovertemplate="%{x|%d %b %Y}: %{y:€,.4f}<br>vs a year earlier: %{customdata:+.1%}<extra></extra>")
        if len(unclear):
            fig.add_scatter(x=unclear["date"], y=unclear["per_share"], mode="markers", name="Shares changed just before",
                            marker=dict(size=8, color="rgba(0,0,0,0)", line=dict(color=RAISED, width=2)),
                            hovertemplate="%{x|%d %b %Y}: %{y:€,.4f} (estimate)<extra></extra>")
        _layout(fig, 300).update_yaxes(tickformat=",.2f", rangemode="tozero")
        st.plotly_chart(fig, width="stretch")

    cc = st.column_config
    tbl = growth.assign(
        latest_growth=growth["latest_growth"] * 100, ttm_growth=growth["ttm_growth"] * 100,
        long_run=["" if pd.isna(c) else f"{c:+.1%} ({y})" for c, y in zip(growth["cagr"], growth["cagr_years"])],
    )
    st.dataframe(
        tbl[["short", "latest_date", "latest_ps", "latest_growth", "ttm_growth", "basis", "long_run"]],
        hide_index=True, width="stretch",
        column_config={
            "short": cc.TextColumn("Holding"),
            "latest_date": cc.DateColumn("Last comparable payment", format="D MMM YYYY",
                                         help="Latest payment with a known share count (see the note below)."),
            "latest_ps": cc.NumberColumn("Per share (net)", format="€%.4f"),
            "latest_growth": cc.NumberColumn("vs a year earlier", format="%+.1f%%",
                                             help="That one payment vs the same payment a year before."),
            "ttm_growth": cc.NumberColumn("Growth", format="%+.1f%%",
                                          help="All matched payments of the last 12 months vs a year earlier."),
            "basis": cc.TextColumn("Based on"),
            "long_run": cc.TextColumn("Per year (long run)",
                                      help="Average yearly growth of dividend per share over the full calendar "
                                           "years you held it. Needs at least two full years."),
        },
    )
    st.caption("Per-share amounts are net of withholding tax and in euros, so for US and other non-euro payers a "
               "weaker dollar shows up as a cut (MAIN's dividend rose in dollars). Payments where you bought or sold "
               "shortly before the pay date are left out of the growth figures, because the ex-date isn't in the "
               "export; they show as open circles.")

# --- Per holding -----------------------------------------------------------------
st.subheader("Per holding")
show_sold = st.toggle("Include holdings you sold", value=False)
tbl = per_holding if show_sold else per_holding[per_holding["held"]]
tbl = tbl.assign(status=tbl["held"].map({True: "Held", False: "Sold"}), yield_on_cost=tbl["yield_on_cost"] * 100)
cc = st.column_config
st.dataframe(
    tbl[["short", "status", "total", "ttm", "forward", "yield_on_cost", "per_year", "last_paid", "next_expected"]],
    hide_index=True, width="stretch", height=38 + 35 * min(len(tbl), 25),
    column_config={
        "short": cc.TextColumn("Holding"),
        "status": cc.TextColumn("Status"),
        "total": cc.NumberColumn("All-time", format="€%.2f"),
        "ttm": cc.NumberColumn("Last 12 months", format="€%.2f"),
        "forward": cc.NumberColumn("Next 12 months", format="€%.2f", help="Forecast, see the calendar notes."),
        "yield_on_cost": cc.NumberColumn("Yield on cost", format="%.1f%%",
                                         help="Next 12 months / what you paid for the shares you hold."),
        "per_year": cc.NumberColumn("Payments / yr", help="Payments in the last 12 months."),
        "last_paid": cc.DateColumn("Last paid", format="D MMM YYYY"),
        "next_expected": cc.DateColumn("Next expected", format="D MMM YYYY"),
    },
)
non_payers = per_holding[per_holding["held"] & (per_holding["forward"] == 0)]["short"].tolist()
if non_payers:
    st.caption(f"No dividend expected from {', '.join(non_payers)} "
               f"({eur(float(held.loc[held['forward'] == 0, 'cost_basis'].sum()), 0)} "
               "of cost basis). They lower the portfolio's yield on cost.")

st.caption("All amounts are net of withholding tax, in EUR, dated on the pay date. Cached for 1 hour.")
if st.button("🔄 Reload data"):
    refresh()
    st.rerun()