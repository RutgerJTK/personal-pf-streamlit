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
    quantity, cost_basis, avg_cost, first_buy, last_trade, dividends.
    """
    trades = df[df["action"].isin(["BUY", "SELL"])].sort_values("datetime", kind="stable")
    rows = []
    for isin, g in trades.groupby("isin", sort=False):
        qty = cost = 0.0
        first_buy = None
        for r in g.itertuples(index=False):
            if r.action == "BUY":
                if qty <= _EPS:          # (re)opening a position starts a new holding period
                    first_buy = r.datetime
                    qty, cost = 0.0, 0.0
                qty += r.quantity
                cost += r.net_amount
            else:
                if qty > _EPS:
                    cost -= cost * min(r.quantity / qty, 1.0)
                qty -= r.quantity
                if qty <= 1e-4:
                    qty, cost = 0.0, 0.0
        if qty > 1e-4:
            rows.append({
                "isin": isin, "quantity": qty, "cost_basis": cost,
                "avg_cost": cost / qty, "first_buy": first_buy,
                "last_trade": g["datetime"].iloc[-1],
            })
    cols = ["isin", "quantity", "cost_basis", "avg_cost", "first_buy", "last_trade"]
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