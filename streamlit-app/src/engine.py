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