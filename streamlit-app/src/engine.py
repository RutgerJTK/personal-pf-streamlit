"""
Financial calculations. Pure pandas, no Streamlit imports, so everything here
can be tested on its own. TWR, PnL and dividend projections will land here.

Definitions
-----------
net_invested   Money you put in from outside (iDEAL deposits) minus money you
               took out. Reinvested dividends and sale proceeds do NOT count
               as new money, because they never left the account.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def summary(df: pd.DataFrame) -> dict:
    """Headline numbers straight from the transaction log."""
    net_invested = df.loc[df["action"] == "DEPOSIT", "net_amount"].sum()
    dividends = df.loc[df["action"] == "DIVIDEND", "net_amount"].sum()

    t = df[df["action"].isin(["BUY", "SELL"])]
    signed = np.where(t["action"] == "BUY", t["quantity"], -t["quantity"])
    held = pd.Series(signed, index=t["isin"].values).groupby(level=0).sum()
    return {
        "net_invested": float(net_invested),
        "dividends": float(dividends),
        "open_positions": int((held.abs() > 1e-4).sum()),
        "first_date": df["datetime"].min(),
        "last_date": df["datetime"].max(),
    }


def cash_balance(df: pd.DataFrame) -> tuple[float | None, pd.Timestamp | None]:
    """
    Cash in the DEGIRO account (EUR) and when it was measured, from the CASH
    row the cleaner writes. (None, None) for files made before that existed.
    """
    c = df[df["action"] == "CASH"]
    if c.empty:
        return None, None
    last = c.sort_values("datetime", kind="stable").iloc[-1]
    return float(last["net_amount"]), last["datetime"]


def cumulative_cashflows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Daily series with:
      net_invested    cumulative deposits, in EUR
      dividends_cum   cumulative net dividends, in EUR
    Every calendar day between the first and last event is present, so the
    lines are flat between events.
    """
    d = df.copy()
    d["day"] = d["datetime"].dt.normalize()
    d["invested_delta"] = np.where(d["action"] == "DEPOSIT", d["net_amount"], 0.0)
    d["dividend_delta"] = np.where(d["action"] == "DIVIDEND", d["net_amount"], 0.0)
    daily = d.groupby("day")[["invested_delta", "dividend_delta"]].sum()
    full_range = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_range, fill_value=0.0)
    out = pd.DataFrame({
        "net_invested": daily["invested_delta"].cumsum(),
        "dividends_cum": daily["dividend_delta"].cumsum(),
    })
    out.index.name = "date"
    return out

# =============================================================================
# Holdings
# =============================================================================
# cost_basis     Average-cost method: a SELL removes cost in proportion to the
#                shares sold, so the remaining shares keep their average price.
#                Fees and FX costs are part of the cost (they are in net_amount).
# market_value   quantity x live price in EUR. Falls back to cost_basis when no
#                price is available, flagged by priced == False.
_EPS = 1e-6


def open_positions(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per ISIN that is still held, indexed by isin, with columns
    quantity, cost_basis, avg_cost, first_buy, last_trade, realized, dividends.

    realized  profit on partial sales during the current holding period
              (sale proceeds after fees minus the average cost they remove).
    """
    trades = df[df["action"].isin(["BUY", "SELL"])].sort_values("datetime", kind="stable")
    rows = []
    for isin, g in trades.groupby("isin", sort=False):
        qty = cost = realized = 0.0
        first_buy = None
        for r in g.itertuples(index=False):
            if r.action == "BUY":
                if qty <= _EPS:          # (re)opening a position starts a new holding period
                    first_buy = r.datetime
                    qty, cost, realized = 0.0, 0.0, 0.0
                qty += r.quantity
                cost += r.net_amount
            else:
                if qty > _EPS:
                    removed = cost * min(r.quantity / qty, 1.0)
                    realized += r.net_amount - removed
                    cost -= removed
                qty -= r.quantity
                if qty <= 1e-4:
                    qty, cost = 0.0, 0.0
        if qty > 1e-4:
            rows.append({
                "isin": isin, "quantity": qty, "cost_basis": cost,
                "avg_cost": cost / qty, "first_buy": first_buy,
                "last_trade": g["datetime"].iloc[-1], "realized": realized,
            })
    cols = ["isin", "quantity", "cost_basis", "avg_cost", "first_buy", "last_trade", "realized"]
    out = pd.DataFrame(rows, columns=cols).set_index("isin")
    divs = df[df["action"] == "DIVIDEND"].groupby("isin")["net_amount"].sum()
    out["dividends"] = divs.reindex(out.index).fillna(0.0)
    return out


def build_holdings(positions: pd.DataFrame, meta: dict, quotes: pd.DataFrame | None = None,
                   basis: str = "market_value") -> pd.DataFrame:
    """
    Join open positions with metadata and prices, then compute weights.

    quotes  indexed by isin with columns price_eur and (optional) prev_close_eur.
    basis   'market_value' or 'cost_basis': what the weights are based on.

    weight           share of the whole portfolio (0..1)
    weight_in_group  share of its own group, i.e. of all stocks or of all ETFs
    """
    h = positions.copy()
    for col, default in [("name", None), ("short", None), ("group", "Unmapped"),
                         ("kind", "Unmapped"), ("sector", None), ("region", None),
                         ("currency", "Unknown")]:
        h[col] = [meta.get(i, {}).get(col, default) for i in h.index]
    h["name"] = h["name"].fillna(pd.Series(h.index, index=h.index))
    h["short"] = h["short"].fillna(h["name"])
    h["mapped"] = [i in meta for i in h.index]
    # ETFs without a single sector/region show what they are made of in look-through views
    h["sector"] = [s if isinstance(s, str) and s else
                   ("Diversified (see look-through)" if "sectors" in meta.get(i, {}) else "Unmapped")
                   for i, s in zip(h.index, h["sector"])]
    h["region"] = h["region"].fillna("Unmapped")

    q = quotes if quotes is not None else pd.DataFrame(columns=["price_eur", "prev_close_eur"])
    h["price_eur"] = q["price_eur"].reindex(h.index) if "price_eur" in q else np.nan
    prev = q["prev_close_eur"].reindex(h.index) if "prev_close_eur" in q else np.nan
    h["priced"] = h["price_eur"].notna()
    h["market_value"] = np.where(h["priced"], h["quantity"] * h["price_eur"], h["cost_basis"])
    h["unrealized"] = np.where(h["priced"], h["market_value"] - h["cost_basis"], np.nan)
    h["unrealized_pct"] = h["unrealized"] / h["cost_basis"]
    # Same idea as DEGIRO's "Totale W/V": price gain after fees, plus profit on
    # partial sales this holding period, plus dividends received
    h["total_return"] = h["unrealized"] + h["realized"] + h["dividends"]
    h["day_change_pct"] = h["price_eur"] / prev - 1
    h["day_change_eur"] = (h["price_eur"] - prev) * h["quantity"]

    if basis not in ("market_value", "cost_basis"):
        raise ValueError(f"basis must be market_value or cost_basis, got {basis!r}")
    h["value"] = h[basis]
    total = h["value"].sum()
    h["weight"] = h["value"] / total if total else 0.0
    group_total = h.groupby("group")["value"].transform("sum")
    h["weight_in_group"] = h["value"] / group_total.replace(0, np.nan)
    return h.sort_values("value", ascending=False)


def exposure(holdings: pd.DataFrame, meta: dict, dim: str = "sector",
             look_through: bool = True) -> pd.DataFrame:
    """
    Portfolio exposure per sector (dim='sector') or region (dim='region').

    look_through=True splits each ETF over its 'sectors' / 'regions' weights
    from the metadata; otherwise every ETF counts as one block under its own
    'kind' (e.g. 'Equity ETF'). Returns one row per (bucket, group) with value
    and weight (share of the whole portfolio), sorted largest bucket first.
    """
    many = {"sector": "sectors", "region": "regions"}[dim]
    parts = []
    for isin, r in holdings.iterrows():
        m = meta.get(isin, {})
        split = m.get(many) if look_through else None
        if split:
            tot = sum(split.values())
            for bucket, w in split.items():
                parts.append((bucket, r["group"], r["value"] * w / tot))
        elif not look_through and r["group"] == "ETF" and dim == "sector":
            parts.append((r["kind"], r["group"], r["value"]))
        else:
            parts.append((r[dim], r["group"], r["value"]))
    out = pd.DataFrame(parts, columns=[dim, "group", "value"])
    out = out.groupby([dim, "group"], as_index=False)["value"].sum()
    total = out["value"].sum()
    out["weight"] = out["value"] / total if total else 0.0
    out = out[out["weight"] >= 0.0005]                  # drop rounding dust (< 0.05%)
    order = out.groupby(dim)["value"].sum().sort_values(ascending=False).index
    out[dim] = pd.Categorical(out[dim], categories=list(order), ordered=True)
    return out.sort_values([dim, "group"]).reset_index(drop=True)


def currency_exposure(holdings: pd.DataFrame) -> pd.Series:
    """Share of the portfolio per underlying currency, largest first."""
    s = holdings.groupby("currency")["value"].sum()
    return (s / s.sum()).sort_values(ascending=False)


def concentration(holdings: pd.DataFrame) -> dict:
    """
    top1 / top5      weight of the largest / five largest positions
    effective_n      1 / sum(w^2): how many equal-sized positions the portfolio
                     'behaves like'. 18 equal positions -> 18; one big one -> ~1.
    """
    w = holdings["weight"].sort_values(ascending=False)
    hhi = float((w ** 2).sum())
    return {
        "top1": float(w.iloc[0]) if len(w) else 0.0,
        "top5": float(w.head(5).sum()),
        "effective_n": 1 / hhi if hhi else 0.0,
    }
# =============================================================================
# Benchmark: portfolio value history, time-weighted return, comparisons
# =============================================================================
# value          holdings at market (daily close, EUR) + cash. Cash is rebuilt
#                from the log: deposits - buys + sells + dividends.
# TWR            time-weighted return. Each day's return is
#                    r_t = V_t / (V_{t-1} + F_t) - 1
#                where F_t is that day's deposit (treated as arriving at the start
#                of the day, because it is usually invested the same day). Chaining
#                the r_t removes the effect of *when* you added money, so the
#                result is comparable with an index.
# MWR            money-weighted return (XIRR): the yearly rate that makes all your
#                deposits grow into today's value. Unlike TWR it rewards or
#                punishes your timing.
# Prices without a market history fall back to the last trade price, so a
# position held without data is flat between its trades (see `coverage`).
# Market prices are cleaned first (repair_prices) and tied to your own trade
# prices (anchor_factors), because Yahoo's split adjustments are not always right.

PERIODS = {"1M": 30, "3M": 91, "YTD": None, "1Y": 365, "3Y": 3 * 365, "5Y": 5 * 365, "All": None}


def _days(df: pd.DataFrame, end: pd.Timestamp | None = None) -> pd.DatetimeIndex:
    start = df["datetime"].min().normalize()
    end = (end or pd.Timestamp.today()).normalize()
    return pd.date_range(start, max(end, df["datetime"].max().normalize()), freq="D")


def daily_quantities(df: pd.DataFrame, end: pd.Timestamp | None = None) -> pd.DataFrame:
    """Shares held at the end of each calendar day, one column per ISIN."""
    t = df[df["action"].isin(["BUY", "SELL"])].copy()
    days = _days(df, end)
    if t.empty:
        return pd.DataFrame(index=days)
    t["day"] = t["datetime"].dt.normalize()
    t["signed"] = np.where(t["action"] == "BUY", t["quantity"], -t["quantity"])
    q = t.pivot_table(index="day", columns="isin", values="signed", aggfunc="sum")
    q = q.reindex(days, fill_value=0.0).fillna(0.0).cumsum()
    return q.where(q.abs() > 1e-4, 0.0)


def daily_cash(df: pd.DataFrame, end: pd.Timestamp | None = None) -> pd.DataFrame:
    """Daily cash balance and external flows (deposits), in EUR."""
    sign = {"DEPOSIT": 1.0, "BUY": -1.0, "SELL": 1.0, "DIVIDEND": 1.0}
    d = df.assign(day=df["datetime"].dt.normalize(),
                  cash=df["net_amount"] * df["action"].map(sign).fillna(0.0),
                  flow=np.where(df["action"] == "DEPOSIT", df["net_amount"], 0.0))
    g = d.groupby("day")[["cash", "flow"]].sum().reindex(_days(df, end), fill_value=0.0)
    return pd.DataFrame({"cash": g["cash"].cumsum(), "flow": g["flow"]})


def trade_prices(df: pd.DataFrame) -> pd.DataFrame:
    """EUR price per share of every trade (fees included), one column per ISIN."""
    t = df[df["action"].isin(["BUY", "SELL"]) & (df["quantity"] > 0)].copy()
    t["day"] = t["datetime"].dt.normalize()
    t["px"] = t["net_amount"] / t["quantity"]
    return t.pivot_table(index="day", columns="isin", values="px", aggfunc="median")


def repair_prices(prices: pd.DataFrame, jump: float = 3.0, spike: float = 0.35) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean obvious data errors out of daily price series.

    spike   a close more than 35% away from the median of the 5 days around it
            (a one- or two-day blip that reverts) is dropped.
    jump    a day-on-day move above 3x or below 1/3 is treated as a broken split
            or unit adjustment: everything before it is rescaled so the series
            is continuous. Real 3x days are vanishingly rare; Yahoo split
            mistakes are not.

    Returns (clean prices, events) with events columns: column, date, kind, ratio.
    """
    out, events = {}, []
    for c in prices.columns:
        s = prices[c].dropna()
        s = s[s > 0]
        if len(s) < 3:
            out[c] = s
            continue
        med = s.rolling(5, center=True, min_periods=3).median()
        bad = (s / med - 1).abs() > spike
        for d in s.index[bad]:
            events.append({"column": c, "date": d, "kind": "spike removed", "ratio": float(s[d] / med[d])})
        s = s[~bad]
        r = s / s.shift(1)
        breaks = r[(r > jump) | (r < 1 / jump)]
        if len(breaks):
            adj = pd.Series(1.0, index=s.index)
            for d, ratio in breaks.items():
                adj[adj.index < d] *= ratio
                events.append({"column": c, "date": d, "kind": "level break fixed", "ratio": float(ratio)})
            s = s * adj
        out[c] = s
    ev = pd.DataFrame(events, columns=["column", "date", "kind", "ratio"])
    return pd.DataFrame(out).reindex(prices.index.union(pd.DataFrame(out).index)), ev


def anchor_factors(mkt: pd.DataFrame, df: pd.DataFrame, lo: float = 0.6, hi: float = 1.6) -> pd.DataFrame:
    """
    Daily scale factor per ISIN that ties the market price to your own trades.

    At every trade the ratio trade price / market close is checked. Within
    0.6-1.6 it is fees, FX and intraday noise and the factor is 1. Outside it
    the market series is on a different scale (typically a split Yahoo applied
    after you sold), so that ratio is used until your next trade in it.
    Returns a frame like `mkt`, 1.0 where nothing needed fixing.
    """
    tp = trade_prices(df)
    f = pd.DataFrame(1.0, index=mkt.index, columns=mkt.columns)
    for isin in mkt.columns:
        if isin not in tp or mkt[isin].isna().all():
            continue
        t = tp[isin].dropna()
        ratio = (t / mkt[isin].reindex(t.index)).replace([np.inf, -np.inf], np.nan).dropna()
        ratio = ratio.where((ratio < lo) | (ratio > hi), 1.0)
        if (ratio == 1.0).all():
            continue
        f[isin] = ratio.reindex(f.index).ffill().bfill().fillna(1.0)
    return f


def portfolio_history(df: pd.DataFrame, prices: pd.DataFrame | None = None,
                      end: pd.Timestamp | None = None) -> dict:
    """
    Daily portfolio value.

    prices   daily closes in EUR, one column per ISIN (any subset). Gaps such as
             weekends are forward-filled. ISINs without a column, or days before
             their history starts, use the last trade price.

    Returns {"daily": DataFrame[holdings, cash, value, flow, invested, priced_share],
             "factors": Series, "coverage": DataFrame}.
    """
    q = daily_quantities(df, end)
    cash = daily_cash(df, end)
    days = cash.index
    prices = prices if prices is not None else pd.DataFrame(index=days)
    clean, repairs = repair_prices(prices.reindex(columns=[c for c in q.columns if c in prices]))
    mkt = clean.reindex(columns=q.columns)
    mkt = mkt.reindex(mkt.index.union(days)).sort_index().ffill().reindex(days)
    factors = anchor_factors(mkt, df)
    mkt = mkt * factors
    fallback = trade_prices(df).reindex(columns=q.columns)
    fallback = fallback.reindex(fallback.index.union(days)).sort_index().ffill().reindex(days)
    px = mkt.combine_first(fallback).fillna(0.0)

    pos_value = q * px
    holdings = pos_value.sum(axis=1)
    priced_value = (q * mkt).sum(axis=1, min_count=1).fillna(0.0)
    daily = pd.DataFrame({
        "holdings": holdings, "cash": cash["cash"], "flow": cash["flow"],
    })
    daily["value"] = daily["holdings"] + daily["cash"]
    daily["invested"] = daily["flow"].cumsum()
    daily["priced_share"] = np.where(holdings > 0, priced_value / holdings.replace(0, np.nan), 1.0)
    daily.index.name = "date"

    # How much of the history each unpriced position accounts for (EUR-days)
    unpriced = [c for c in q.columns if c not in mkt.columns or mkt[c].isna().all()]
    eur_days = pos_value.clip(lower=0).sum()
    partial = {c: float((pos_value[c] * mkt[c].isna()).clip(lower=0).sum()) for c in q.columns}
    coverage = pd.DataFrame({
        "unpriced_eur_days": pd.Series(partial),
        "share_of_history": pd.Series(partial) / eur_days.sum() if eur_days.sum() else 0.0,
        "no_data_at_all": [c in unpriced for c in q.columns],
        "held_days": (q > 0).sum(),
    }).sort_values("unpriced_eur_days", ascending=False)
    # Net money put into each position per day (buys - sells), to separate trading from price moves
    t = df[df["action"].isin(["BUY", "SELL"])]
    traded = (t.assign(day=t["datetime"].dt.normalize(),
                       amt=np.where(t["action"] == "BUY", t["net_amount"], -t["net_amount"]))
              .pivot_table(index="day", columns="isin", values="amt", aggfunc="sum")
              .reindex(index=days, columns=q.columns).fillna(0.0))
    odd = factors.loc[:, (factors != 1.0).any()]
    return {"daily": daily, "positions": pos_value, "traded": traded, "repairs": repairs,
            "factors": odd.apply(lambda c: c[c != 1.0].median()) if len(odd.columns) else pd.Series(dtype=float),
            "coverage": coverage[coverage["unpriced_eur_days"] > 0]}


def biggest_moves(hist: dict, n: int = 10, start: pd.Timestamp | None = None) -> pd.DataFrame:
    """
    The n days with the largest time-weighted portfolio return, and what drove them.

    driver_eur per position = change in its value - money traded into it that day,
    so buying or selling is not mistaken for a gain. Deposits are shown to prove
    they are not counted as return.
    """
    d = hist["daily"]
    base = d["value"].shift(1).fillna(0.0) + d["flow"]
    r = pd.Series(np.where(base > 1.0, d["value"] / base - 1, 0.0), index=d.index)
    if start is not None:
        r = r[r.index >= start]
    top = r.abs().sort_values(ascending=False).head(n).index
    pnl = hist["positions"].diff().fillna(hist["positions"]) - hist["traded"]
    rows = []
    for day in top:
        contrib = pnl.loc[day]
        isin = contrib.abs().idxmax() if len(contrib) else None
        rows.append({
            "date": day, "return": r[day], "value": d.loc[day, "value"], "deposit": d.loc[day, "flow"],
            "gain_eur": d.loc[day, "value"] - base[day],
            "top_isin": isin, "top_eur": float(contrib[isin]) if isin else np.nan,
        })
    return pd.DataFrame(rows)


def twr_index(daily: pd.DataFrame) -> pd.Series:
    """Growth of 1 euro under the time-weighted return, starting at the first deposit."""
    v, f = daily["value"], daily["flow"]
    base = v.shift(1).fillna(0.0) + f
    r = np.where(base > 1.0, v / base - 1, 0.0)
    idx = pd.Series(np.cumprod(1 + r), index=daily.index, name="Portfolio")
    started = base > 1.0
    return idx[started.cummax()]


def period_start(index: pd.DatetimeIndex, period: str) -> pd.Timestamp:
    """First date of a period ending at the last date of `index`."""
    last = index.max()
    if period == "All":
        return index.min()
    if period == "YTD":
        start = pd.Timestamp(year=last.year, month=1, day=1) - pd.Timedelta(days=1)  # rebase on 31 Dec
    else:
        start = last - pd.Timedelta(days=PERIODS[period])
    return max(start, index.min())


def rebase(levels: pd.DataFrame, start: pd.Timestamp) -> pd.DataFrame:
    """Each column divided by its value on (or the first value after) `start`; earlier rows dropped."""
    part = levels[levels.index >= start].ffill()
    first = part.bfill().iloc[0] if len(part) else part.iloc[:0]
    return part / first


def xirr(dates: pd.Series | list, amounts: pd.Series | list) -> float:
    """Annual money-weighted return. Deposits negative, final value positive. NaN if undefined."""
    d = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    a = pd.Series(amounts, dtype=float).reset_index(drop=True)
    keep = a != 0
    d, a = d[keep], a[keep]
    if len(a) < 2 or (a > 0).all() or (a < 0).all():
        return float("nan")
    t = ((d - d.min()).dt.days / 365.25).to_numpy()
    a = a.to_numpy()

    def npv(r):
        return float(np.sum(a / (1 + r) ** t))

    lo, hi = -0.9999, 10.0
    f_lo, f_hi = npv(lo), npv(hi)
    if np.sign(f_lo) == np.sign(f_hi):
        return float("nan")
    for _ in range(200):  # bisection: slow but cannot diverge
        mid = (lo + hi) / 2
        f_mid = npv(mid)
        if abs(f_mid) < 1e-7:
            break
        if np.sign(f_mid) == np.sign(f_lo):
            lo, f_lo = mid, f_mid
        else:
            hi = mid
    return float((lo + hi) / 2)


def money_weighted(daily: pd.DataFrame, start: pd.Timestamp) -> float:
    """XIRR over [start, last day]: opening value and deposits in, closing value out."""
    part = daily[daily.index >= start]
    if part.empty:
        return float("nan")
    before = daily[daily.index < start]
    opening = float(before["value"].iloc[-1]) if len(before) else 0.0
    dates = [start] + list(part.index[part["flow"] != 0]) + [part.index[-1]]
    amounts = [-opening] + list(-part.loc[part["flow"] != 0, "flow"]) + [float(part["value"].iloc[-1])]
    return xirr(dates, amounts)


def drawdown(levels: pd.DataFrame) -> pd.DataFrame:
    """Distance below the running peak, 0 at a new high, e.g. -0.25 = 25% below."""
    return levels / levels.cummax() - 1


def weekly_returns(levels: pd.DataFrame) -> pd.DataFrame:
    """Friday-to-Friday returns. Weekly smooths out time-zone and holiday mismatches between markets."""
    return levels.resample("W-FRI").last().pct_change(fill_method=None).iloc[1:]


def perf_stats(levels: pd.DataFrame, ref: str | None = None) -> pd.DataFrame:
    """
    One row per column of rebased `levels`:
    total, cagr, vol (annualised, weekly), ratio (cagr / vol), max_dd,
    best_month, worst_month, pos_months, and when `ref` is given,
    corr and beta of `ref` against each other column (weekly).
    """
    w = weekly_returns(levels)
    m = levels.resample("ME").last()
    m = pd.concat([levels.iloc[[0]], m]).pct_change(fill_method=None).iloc[1:]
    rows = {}
    for c in levels.columns:
        s = levels[c].dropna()
        if len(s) < 2:
            continue
        years = (s.index[-1] - s.index[0]).days / 365.25
        total = s.iloc[-1] / s.iloc[0] - 1
        cagr = (1 + total) ** (1 / years) - 1 if years >= 1 else np.nan  # don't annualise < 1 year
        vol = w[c].std() * np.sqrt(52)
        mc = m[c].dropna()
        row = {
            "total": total, "cagr": cagr, "vol": vol,
            "ratio": cagr / vol if vol else np.nan,
            "max_dd": float(drawdown(s.to_frame()).min().iloc[0]),
            "best_month": mc.max(), "worst_month": mc.min(),
            "pos_months": (mc > 0).mean() if len(mc) else np.nan,
        }
        if ref and ref in w and c != ref:
            pair = w[[ref, c]].dropna()
            row["corr"] = pair[ref].corr(pair[c])
            var = pair[c].var()
            row["beta"] = pair[ref].cov(pair[c]) / var if var else np.nan
        rows[c] = row
    return pd.DataFrame.from_dict(rows, orient="index")


def calendar_returns(levels: pd.DataFrame) -> pd.DataFrame:
    """Return per calendar year (rows = year). The first and last year may be partial."""
    yearly = levels.resample("YE").last()
    yearly = pd.concat([levels.iloc[[0]], yearly[yearly.index > levels.index[0]]])
    out = yearly.pct_change(fill_method=None).iloc[1:]
    out.index = out.index.year
    return out[~out.index.duplicated(keep="last")]


def shadow_portfolio(daily: pd.DataFrame, bench: pd.Series, start: pd.Timestamp) -> pd.Series:
    """
    EUR value if the portfolio's value at `start`, and every deposit after it,
    had been put into `bench` on the same day instead.
    """
    part = daily[daily.index >= start]
    b = bench.reindex(bench.index.union(part.index)).sort_index().ffill().reindex(part.index).bfill()
    before = daily[daily.index < start]
    flows = part["flow"].copy()
    flows.iloc[0] += float(before["value"].iloc[-1]) if len(before) else 0.0
    units = (flows / b).fillna(0.0).cumsum()
    return units * b

# =============================================================================
# Dividends
# =============================================================================
# All amounts are net: after withholding tax, converted to EUR by the cleaner.
# Dividends are dated on their pay date (the cleaner uses DEGIRO's value date).
#
# Forecast (next 12 months, open positions only)
#   per share   each past payment / shares held around its ex-date. The ex-date
#               is not in the export, so the most shares held in the 45 days
#               before the pay date is used.
#   >= 1 year of payments
#               every payment of the last ~12 months repeats one year later at
#               the same amount per share, times the shares held now. A payment
#               whose 'next year' already arrived (pay dates drift by a few days)
#               is not counted twice.
#   < 1 year    with 4+ payments the rhythm is read from the gaps between them
#               (monthly / quarterly / half-yearly / yearly) and the latest
#               amount per share is repeated at that rhythm. With fewer, two
#               payments can't tell quarterly from half-yearly, so only the
#               payments seen so far are repeated a year later (cautious).
#   Positions that have not paid for 400+ days are treated as non-payers.
#   No dividend growth is assumed, so the forecast is on the cautious side.
_DIV_LOOKBACK = 380          # days of history that make up 'last year's schedule'
_DIV_SAME_PAYMENT = 25       # days: two payments this close are the same event
_DIV_STOPPED = 400           # days without payment: no longer a payer


def dividend_payments(df: pd.DataFrame, meta: dict | None = None) -> pd.DataFrame:
    """One row per dividend payment: date, isin, amount, short, group."""
    meta = meta or {}
    d = df[df["action"] == "DIVIDEND"]
    out = pd.DataFrame({
        "date": d["datetime"].dt.normalize().to_numpy(),
        "isin": d["isin"].to_numpy(),
        "amount": d["net_amount"].to_numpy(dtype=float),
    })
    out["short"] = [meta.get(i, {}).get("short") or meta.get(i, {}).get("name") or i for i in out["isin"]]
    out["group"] = [meta.get(i, {}).get("group", "Unmapped") for i in out["isin"]]
    return out.sort_values("date", kind="stable").reset_index(drop=True)


def dividends_by_year(df: pd.DataFrame, today: pd.Timestamp | None = None) -> pd.DataFrame:
    """
    Indexed by calendar year:
      received   net dividends that year
      payments   number of payments
      payers     number of different holdings that paid
      growth     change vs the previous year (NaN for the first year). For the
                 current year: vs the same stretch of last year (1 Jan - today's date).
      per_month  received / 12, or / months elapsed for the current year
      partial    True for the current (unfinished) year
    """
    today = (today or pd.Timestamp.today()).normalize()
    p = dividend_payments(df)
    if p.empty:
        return pd.DataFrame(columns=["received", "payments", "payers", "growth", "per_month", "partial"])
    years = range(p["date"].dt.year.min(), max(today.year, p["date"].dt.year.max()) + 1)
    g = p.groupby(p["date"].dt.year)
    out = pd.DataFrame({
        "received": g["amount"].sum(),
        "payments": g.size(),
        "payers": g["isin"].nunique(),
    }).reindex(years).fillna({"received": 0.0, "payments": 0, "payers": 0})
    out = out.astype({"payments": int, "payers": int})
    out["growth"] = out["received"] / out["received"].shift(1).replace(0, np.nan) - 1
    out["partial"] = out.index == today.year
    if today.year in out.index and today.year - 1 in out.index:
        s = dividend_stats(df, today)
        out.loc[today.year, "growth"] = s["ytd"] / s["ytd_prev"] - 1 if s["ytd_prev"] else np.nan
    months = np.where(out["partial"], today.month, 12)
    out["per_month"] = out["received"] / months
    out.index.name = "year"
    return out


def dividends_by_month(df: pd.DataFrame, today: pd.Timestamp | None = None) -> pd.DataFrame:
    """
    Every month from the first dividend to the current month:
      received   net dividends that month
      ttm        sum of the last 12 months (NaN until 12 months exist)
      avg_12m    ttm / 12, i.e. the 'typical month' on the same scale as received
    """
    today = (today or pd.Timestamp.today()).normalize()
    p = dividend_payments(df)
    if p.empty:
        return pd.DataFrame(columns=["received", "ttm", "avg_12m"])
    m = p.groupby(p["date"].dt.to_period("M"))["amount"].sum()
    idx = pd.period_range(m.index.min(), max(m.index.max(), today.to_period("M")), freq="M")
    m = m.reindex(idx, fill_value=0.0)
    out = pd.DataFrame({"received": m.to_numpy()}, index=idx.to_timestamp())
    out["ttm"] = out["received"].rolling(12, min_periods=12).sum()
    out["avg_12m"] = out["ttm"] / 12
    out.index.name = "month"
    return out


def dividend_stats(df: pd.DataFrame, today: pd.Timestamp | None = None) -> dict:
    """
    total      all-time net dividends
    ytd        this calendar year so far; ytd_prev the same stretch of last year
    ttm        last 365 days; ttm_prev the 365 days before that
    """
    today = (today or pd.Timestamp.today()).normalize()
    p = dividend_payments(df)
    s, a = p["date"], p["amount"]
    jan1 = pd.Timestamp(year=today.year, month=1, day=1)
    last_year_today = today - pd.DateOffset(years=1)
    y = pd.Timedelta(days=365)
    return {
        "total": float(a.sum()),
        "ytd": float(a[(s >= jan1) & (s <= today)].sum()),
        "ytd_prev": float(a[(s >= jan1 - pd.DateOffset(years=1)) & (s <= last_year_today)].sum()),
        "ttm": float(a[(s > today - y) & (s <= today)].sum()),
        "ttm_prev": float(a[(s > today - 2 * y) & (s <= today - y)].sum()),
    }


def _payments_per_share(df: pd.DataFrame, today: pd.Timestamp) -> tuple[pd.DataFrame, pd.Series]:
    """
    Dividend payments with per_share, shares_then and clear, plus shares held today.
    clear is False when the position changed in the 45 days before the pay date:
    the ex-date is unknown, so it's unclear how many shares the payment was for.
    """
    p = dividend_payments(df)
    q = daily_quantities(df, end=today)
    shares, clear = [], []
    for d, isin in zip(p["date"], p["isin"]):
        window = q.loc[d - pd.Timedelta(days=45):d, isin] if isin in q.columns else pd.Series(dtype=float)
        shares.append(float(window.max()) if len(window) else 0.0)
        clear.append(bool(len(window)) and float(window.max() - window.min()) < 1e-4)
    p["shares_then"] = shares
    p["clear"] = clear   # False: shares changed shortly before paying, so per_share is a guess
    p["per_share"] = np.where(p["shares_then"] > 1e-4, p["amount"] / p["shares_then"].replace(0, np.nan), np.nan)
    now = q.iloc[-1] if len(q) else pd.Series(dtype=float)
    return p, now[now > 1e-4]


def _snap_months(median_gap_days: float) -> int:
    """Nearest of monthly / quarterly / half-yearly / yearly."""
    options = np.array([1, 3, 6, 12])
    return int(options[np.abs(options * 30.44 - median_gap_days).argmin()])


def dividend_forecast(df: pd.DataFrame, meta: dict | None = None,
                      today: pd.Timestamp | None = None, horizon_days: int = 365) -> pd.DataFrame:
    """
    Expected payments in (today, today + horizon_days] for positions held today.
    Columns: date, isin, short, group, amount, per_share, shares, method.
    method is 'last year' or 'rhythm' (see the notes above).
    """
    meta = meta or {}
    today = (today or pd.Timestamp.today()).normalize()
    end = today + pd.Timedelta(days=horizon_days)
    pay, held = _payments_per_share(df, today)
    rows = []
    for isin, shares in held.items():
        p = pay[(pay["isin"] == isin) & pay["per_share"].notna()]
        if p.empty or p["date"].max() < today - pd.Timedelta(days=_DIV_STOPPED):
            continue
        gaps = p["date"].diff().dt.days.dropna()
        gaps = gaps[gaps >= 20]                      # ignore supplementals right after a regular payment
        if p["date"].min() <= today - pd.Timedelta(days=330) or len(gaps) < 3:
            recent = p[p["date"] > today - pd.Timedelta(days=_DIV_LOOKBACK)]
            for r in recent.itertuples(index=False):
                nxt = r.date + pd.DateOffset(years=1)
                already = (p["date"] - nxt).abs() <= pd.Timedelta(days=_DIV_SAME_PAYMENT)
                if already.any():
                    continue
                nxt = max(nxt, today + pd.Timedelta(days=1))  # a bit late this year: due any day
                if nxt <= end:
                    rows.append((nxt, isin, r.per_share, shares, "last year"))
        else:
            step = pd.DateOffset(months=_snap_months(gaps.median()))
            per_share = float(p["per_share"].tail(3).median())
            nxt = p["date"].max() + step
            while nxt <= today:
                nxt += step
            while nxt <= end:
                rows.append((nxt, isin, per_share, shares, "rhythm"))
                nxt += step
    out = pd.DataFrame(rows, columns=["date", "isin", "per_share", "shares", "method"])
    out["amount"] = out["per_share"] * out["shares"]
    out["short"] = [meta.get(i, {}).get("short") or meta.get(i, {}).get("name") or i for i in out["isin"]]
    out["group"] = [meta.get(i, {}).get("group", "Unmapped") for i in out["isin"]]
    cols = ["date", "isin", "short", "group", "amount", "per_share", "shares", "method"]
    return out[cols].sort_values("date", kind="stable").reset_index(drop=True)


def dividend_calendar(forecast: pd.DataFrame, today: pd.Timestamp | None = None, months: int = 12) -> pd.DataFrame:
    """Expected EUR per holding (rows, by short name) per month (columns), starting this month."""
    today = (today or pd.Timestamp.today()).normalize()
    cols = pd.period_range(today.to_period("M"), periods=months, freq="M")
    if forecast.empty:
        return pd.DataFrame(columns=cols.to_timestamp())
    f = forecast.assign(month=forecast["date"].dt.to_period("M"))
    cal = f.pivot_table(index="short", columns="month", values="amount", aggfunc="sum")
    cal = cal.reindex(columns=cols).fillna(0.0)
    cal.columns = cols.to_timestamp()
    return cal.loc[cal.sum(axis=1).sort_values(ascending=False).index]


def dividends_by_holding(df: pd.DataFrame, forecast: pd.DataFrame, meta: dict | None = None,
                         today: pd.Timestamp | None = None) -> pd.DataFrame:
    """
    One row per holding that ever paid or is held, indexed by isin:
    short, group, held, total, ttm, forward (next 12 months), cost_basis,
    yield_on_cost (forward / cost_basis), per_year (payments in the last year),
    last_paid, next_expected.
    """
    meta = meta or {}
    today = (today or pd.Timestamp.today()).normalize()
    p = dividend_payments(df, meta)
    pos = open_positions(df)
    isins = list(dict.fromkeys(list(pos.index) + list(p["isin"].unique())))
    g = p.groupby("isin")
    last_year = p[p["date"] > today - pd.Timedelta(days=365)].groupby("isin")
    fg = forecast.groupby("isin")
    out = pd.DataFrame(index=pd.Index(isins, name="isin"))
    out["short"] = [meta.get(i, {}).get("short") or meta.get(i, {}).get("name") or i for i in out.index]
    out["group"] = [meta.get(i, {}).get("group", "Unmapped") for i in out.index]
    out["held"] = out.index.isin(pos.index)
    out["total"] = g["amount"].sum().reindex(out.index).fillna(0.0)
    out["ttm"] = last_year["amount"].sum().reindex(out.index).fillna(0.0)
    out["forward"] = fg["amount"].sum().reindex(out.index).fillna(0.0)
    out["cost_basis"] = pos["cost_basis"].reindex(out.index)
    out["yield_on_cost"] = out["forward"] / out["cost_basis"]
    out["per_year"] = last_year.size().reindex(out.index).fillna(0).astype(int)
    out["last_paid"] = g["date"].max().reindex(out.index)
    out["next_expected"] = fg["date"].min().reindex(out.index)
    return out.sort_values(["held", "forward", "total"], ascending=False)



# -----------------------------------------------------------------------------
# Dividend growth
# -----------------------------------------------------------------------------
# Like-for-like  each payment is compared with the same payment one year earlier
#                (pay date within 25 days of a year before; 45 days for holdings
#                that pay twice a year or less, whose dates drift more). That handles
#                seasonal patterns (a big June payment vs a small March one),
#                drifting pay dates and special/supplemental dividends without
#                needing to know the payment frequency.
# Only 'clear' payments are used (see _payments_per_share), so a buy right
# before a pay date can't show up as a fake cut.
# Everything is net and in EUR: currency moves and withholding-tax changes show
# up as growth or cuts too.
_DIV_FLAT = 0.02            # within +/- 2% counts as unchanged


def dividend_per_share_history(df: pd.DataFrame, meta: dict | None = None,
                               today: pd.Timestamp | None = None) -> pd.DataFrame:
    """
    One row per payment: date, isin, short, amount, shares_then, per_share, clear,
    prev_date / prev_per_share (the matching payment a year earlier, if any) and
    growth (per_share / prev_per_share - 1). Matching uses clear payments only.
    """
    meta = meta or {}
    today = (today or pd.Timestamp.today()).normalize()
    p, _ = _payments_per_share(df, today)
    p["short"] = [meta.get(i, {}).get("short") or meta.get(i, {}).get("name") or i for i in p["isin"]]
    p["prev_date"] = pd.NaT
    p["prev_per_share"] = np.nan
    ok = p["clear"] & p["per_share"].notna()
    for isin, g in p[ok].groupby("isin"):
        dates = g["date"]
        all_dates = p.loc[p["isin"] == isin, "date"]
        years_paying = max((all_dates.max() - all_dates.min()).days / 365.25, 1.0)
        per_year = len(all_dates) / years_paying
        tol = pd.Timedelta(days=_DIV_SAME_PAYMENT if per_year > 2.5 else 45)
        for idx, d in dates.items():
            target = d - pd.DateOffset(years=1)
            gap = (dates - target).abs()
            gap = gap[(gap <= tol) & (dates < d - pd.Timedelta(days=300))]
            if len(gap):
                j = gap.idxmin()
                p.loc[idx, "prev_date"] = p.loc[j, "date"]
                p.loc[idx, "prev_per_share"] = p.loc[j, "per_share"]
    p["growth"] = p["per_share"] / p["prev_per_share"] - 1
    return p


def annual_dividend_per_share(history: pd.DataFrame, isin: str, df: pd.DataFrame,
                              today: pd.Timestamp | None = None) -> pd.DataFrame:
    """
    Dividend per share per calendar year for one holding:
    dps, payments, uncertain (some payments not 'clear'), partial (not held the
    whole year, or the current year), growth (vs previous year, full years only).
    """
    today = (today or pd.Timestamp.today()).normalize()
    h = history[(history["isin"] == isin) & history["per_share"].notna()]
    if h.empty:
        return pd.DataFrame(columns=["dps", "payments", "uncertain", "partial", "growth"])
    g = h.groupby(h["date"].dt.year)
    out = pd.DataFrame({"dps": g["per_share"].sum(), "payments": g.size(),
                        "uncertain": ~g["clear"].all()})
    q = daily_quantities(df, end=today)
    held = q[isin] > 1e-4 if isin in q.columns else pd.Series(dtype=bool)
    full = {y: bool(held[str(y)].all()) and len(held[str(y)]) >= 365 and y < today.year for y in out.index}
    out["partial"] = [not full[y] for y in out.index]
    prev = out["dps"].shift(1).where(~out["partial"].shift(1, fill_value=True))
    out["growth"] = (out["dps"] / prev - 1).where(~out["partial"])
    out.index.name = "year"
    return out


def dividend_growth(df: pd.DataFrame, meta: dict | None = None,
                    today: pd.Timestamp | None = None) -> tuple[pd.DataFrame, dict]:
    """
    Per held holding that pays (indexed by isin):
      short, latest_date, latest_ps, latest_growth   most recent clear payment
                                                     and its change vs a year earlier
      ttm_growth   like-for-like: last 12 months' matched payments vs the same
                   payments a year earlier (sum of per-share amounts). When no
                   payment matches, the latest full calendar year vs the one
                   before; basis says which ('12 months' / 'calendar year').
      matched      number of matched payments behind ttm_growth
      cagr         per-year growth of dividend per share over full calendar years
                   (NaN with fewer than 2 full years), cagr_years its span
      status       'raised' / 'unchanged' / 'cut' / 'unknown' (from ttm_growth)
      income_now / income_before   matched payments x shares held now, in EUR
    And a summary dict:
      organic      income_now / income_before - 1 over all holdings: how much your
                   dividends grew from raises alone, with the shares you have now
      raised / unchanged / cut / unknown   counts
    """
    meta = meta or {}
    today = (today or pd.Timestamp.today()).normalize()
    hist = dividend_per_share_history(df, meta, today)
    _, held = _payments_per_share(df, today)
    rows = []
    for isin, shares in held.items():
        h = hist[hist["isin"] == isin]
        if h.empty or h["date"].max() < today - pd.Timedelta(days=_DIV_STOPPED):
            continue
        clear = h[h["clear"] & h["per_share"].notna()]
        recent = clear[(clear["date"] > today - pd.Timedelta(days=365)) & clear["prev_per_share"].notna()]
        now, before = recent["per_share"].sum(), recent["prev_per_share"].sum()
        ttm = now / before - 1 if before else np.nan
        annual = annual_dividend_per_share(hist, isin, df, today)
        full = annual[~annual["partial"]]
        basis = "12 months"
        if np.isnan(ttm) and full["growth"].notna().any():
            ttm, basis = float(full["growth"].dropna().iloc[-1]), f"{full['growth'].dropna().index[-1]} vs year before"
        cagr = ((full["dps"].iloc[-1] / full["dps"].iloc[0]) ** (1 / (len(full) - 1)) - 1
                if len(full) >= 2 and full["dps"].iloc[0] > 0 else np.nan)
        last = clear.iloc[-1] if len(clear) else None
        if np.isnan(ttm):
            status = "unknown"
        elif ttm > _DIV_FLAT:
            status = "raised"
        elif ttm < -_DIV_FLAT:
            status = "cut"
        else:
            status = "unchanged"
        rows.append({
            "isin": isin, "short": h["short"].iloc[0],
            "latest_date": last["date"] if last is not None else pd.NaT,
            "latest_ps": last["per_share"] if last is not None else np.nan,
            "latest_growth": last["growth"] if last is not None else np.nan,
            "ttm_growth": ttm, "basis": basis, "matched": len(recent),
            "cagr": cagr, "cagr_years": f"{full.index[0]}–{full.index[-1]}" if len(full) >= 2 else "",
            "status": status, "income_now": now * shares, "income_before": before * shares,
        })
    cols = ["isin", "short", "latest_date", "latest_ps", "latest_growth", "ttm_growth", "basis", "matched",
            "cagr", "cagr_years", "status", "income_now", "income_before"]
    out = pd.DataFrame(rows, columns=cols).set_index("isin").sort_values("ttm_growth", ascending=False)
    before = out["income_before"].sum()
    counts = out["status"].value_counts()
    summary = {
        "organic": float(out["income_now"].sum() / before - 1) if before else float("nan"),
        "raised": int(counts.get("raised", 0)), "unchanged": int(counts.get("unchanged", 0)),
        "cut": int(counts.get("cut", 0)), "unknown": int(counts.get("unknown", 0)),
    }
    return out, summary