import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import engine  # noqa: E402


def _df():
    return pd.DataFrame({
        "datetime": pd.to_datetime(["2025-01-01 00:00", "2025-01-01 10:00", "2025-01-03 10:00",
                                    "2025-01-05 00:00", "2025-01-06 10:00"]),
        "isin": ["", "X", "Y", "X", "Y"],
        "action": ["DEPOSIT", "BUY", "BUY", "DIVIDEND", "SELL"],
        "quantity": [0.0, 10.0, 5.0, 0.0, 5.0],
        "net_amount": [150.0, 100.0, 50.0, 3.0, 60.0],
    })


def test_summary_net_invested_is_deposits_only():
    s = engine.summary(_df())
    assert s["net_invested"] == 150          # not buys - sells (= 90)
    assert s["dividends"] == 3
    assert s["open_positions"] == 1          # Y fully sold, X still held


def test_cumulative_cashflows_is_daily_and_flat_between_events():
    f = engine.cumulative_cashflows(_df())
    assert len(f) == 6  # 1 Jan .. 6 Jan
    assert (f["net_invested"] == 150).all()
    assert f.loc["2025-01-04", "dividends_cum"] == 0
    assert f.iloc[-1]["dividends_cum"] == 3

# --- Holdings -----------------------------------------------------------------
def _trades():
    return pd.DataFrame({
        "datetime": pd.to_datetime(["2025-01-01", "2025-02-01", "2025-03-01", "2025-03-05",
                                    "2025-04-01", "2025-05-01", "2025-06-01"]),
        "isin": ["S1", "S1", "S1", "S1", "E1", "E2", "GONE"],
        "action": ["BUY", "BUY", "SELL", "DIVIDEND", "BUY", "BUY", "BUY"],
        "quantity": [10.0, 10.0, 5.0, 0.0, 4.0, 2.0, 1.0],
        "net_amount": [100.0, 300.0, 120.0, 7.0, 400.0, 100.0, 50.0],
    })


META = {
    "S1": {"short": "S1", "group": "Stock", "kind": "Stock", "sector": "Health Care", "region": "US", "currency": "USD"},
    "E1": {"short": "E1", "group": "ETF", "kind": "Equity ETF", "region": "US", "currency": "USD",
           "sectors": {"Information Technology": 75, "Health Care": 25}},
    "E2": {"short": "E2", "group": "ETF", "kind": "Gold ETC", "sector": "Commodities", "region": "Commodity",
           "currency": "USD"},
}


def _positions():
    df = _trades()
    df = pd.concat([df, pd.DataFrame({"datetime": [pd.Timestamp("2025-07-01")], "isin": ["GONE"],
                                      "action": ["SELL"], "quantity": [1.0], "net_amount": [55.0]})])
    return engine.open_positions(df)


def test_open_positions_average_cost():
    p = _positions()
    assert set(p.index) == {"S1", "E1", "E2"}           # GONE fully sold
    assert p.loc["S1", "quantity"] == 15
    assert abs(p.loc["S1", "cost_basis"] - 300) < 1e-9  # 400 avg cost 20/share, sold 5 -> 300 left
    assert abs(p.loc["S1", "avg_cost"] - 20) < 1e-9
    assert p.loc["S1", "dividends"] == 7


def test_open_positions_reopened_resets_holding_period():
    df = pd.DataFrame({
        "datetime": pd.to_datetime(["2025-01-01", "2025-02-01", "2025-03-01"]),
        "isin": ["X", "X", "X"], "action": ["BUY", "SELL", "BUY"],
        "quantity": [1.0, 1.0, 2.0], "net_amount": [10.0, 12.0, 30.0],
    })
    p = engine.open_positions(df)
    assert p.loc["X", "cost_basis"] == 30
    assert p.loc["X", "first_buy"] == pd.Timestamp("2025-03-01")


def test_build_holdings_weights_and_fallback():
    quotes = pd.DataFrame({"price_eur": [30.0, 150.0], "prev_close_eur": [25.0, 150.0]}, index=["S1", "E1"])
    h = engine.build_holdings(_positions(), META, quotes)
    # S1 15*30=450, E1 4*150=600, E2 no price -> cost 100. Total 1150.
    assert abs(h["value"].sum() - 1150) < 1e-9
    assert not h.loc["E2", "priced"] and h.loc["E2", "value"] == 100
    assert abs(h.loc["E1", "weight"] - 600 / 1150) < 1e-9
    assert abs(h.loc["E1", "weight_in_group"] - 600 / 700) < 1e-9   # share of all ETFs
    assert h.loc["S1", "weight_in_group"] == 1.0
    assert h.loc["S1", "unrealized"] == 150
    assert abs(h.loc["S1", "day_change_pct"] - 0.2) < 1e-9
    hc = engine.build_holdings(_positions(), META, quotes, basis="cost_basis")
    assert abs(hc.loc["E1", "weight"] - 400 / 800) < 1e-9


def test_exposure_look_through():
    quotes = pd.DataFrame({"price_eur": [30.0, 150.0]}, index=["S1", "E1"])
    h = engine.build_holdings(_positions(), META, quotes)
    sec = engine.exposure(h, META, "sector", look_through=True)
    tot = sec.groupby("sector", observed=True)["value"].sum()
    assert abs(tot["Health Care"] - (450 + 150)) < 1e-9
    assert abs(tot["Information Technology"] - 450) < 1e-9
    assert abs(sec["weight"].sum() - 1) < 1e-9
    assert list(tot.index)[0] == "Health Care"          # largest first
    flat = engine.exposure(h, META, "sector", look_through=False)
    assert set(flat["sector"].astype(str)) == {"Health Care", "Equity ETF", "Gold ETC"}


def test_concentration():
    h = pd.DataFrame({"weight": [0.5, 0.5]})
    c = engine.concentration(h)
    assert c["top1"] == 0.5 and c["effective_n"] == 2