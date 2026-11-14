import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import engine  # noqa: E402

END = pd.Timestamp("2025-01-10")


def _log():
    """Deposit 1000, buy 10 X at 100, deposit 1000 on day 5, buy 10 more at 110, dividend 5."""
    return pd.DataFrame({
        "datetime": pd.to_datetime(["2025-01-01 09:00", "2025-01-01 10:00", "2025-01-05 09:00",
                                    "2025-01-05 10:00", "2025-01-08 00:00"]),
        "isin": ["", "X", "", "X", "X"],
        "action": ["DEPOSIT", "BUY", "DEPOSIT", "BUY", "DIVIDEND"],
        "quantity": [0.0, 10.0, 0.0, 10.0, 0.0],
        "net_amount": [1000.0, 1000.0, 1000.0, 1100.0, 5.0],
    })


def _prices(level=1.0):
    days = pd.date_range("2025-01-01", END, freq="D")
    px = pd.Series(100.0, index=days)
    px["2025-01-05":] = 110.0
    px["2025-01-09":] = 121.0
    return pd.DataFrame({"X": px * level})


def test_cash_and_value():
    d = engine.portfolio_history(_log(), _prices(), end=END)["daily"]
    assert d.loc["2025-01-01", "cash"] == 0
    assert d.loc["2025-01-05", "cash"] == pytest.approx(-100)  # bought more than deposited
    assert d.loc["2025-01-10", "value"] == pytest.approx(20 * 121 - 100 + 5)
    assert d["invested"].iloc[-1] == 2000


def test_twr_ignores_deposits():
    d = engine.portfolio_history(_log(), _prices(), end=END)["daily"]
    idx = engine.twr_index(d)
    # Day 5: the deposit counts from the start of the day, so +100 on 1000 + 1000 = +5%.
    # Day 8: +5 dividend on 2100. Day 9: shares +10% (2200 -> 2420) on 2105.
    expected = (2100 / 2000) * (2105 / 2100) * (2325 / 2105)
    assert idx.iloc[-1] == pytest.approx(expected)
    assert idx.iloc[0] == 1.0


def test_anchoring_absorbs_later_split():
    h = engine.portfolio_history(_log(), _prices(level=10.0), end=END)  # history 10x too high
    assert h["factors"]["X"] == pytest.approx(0.1, rel=0.05)
    assert h["daily"]["value"].iloc[-1] == pytest.approx(20 * 121 - 100 + 5, rel=0.01)


def test_broken_split_window_is_repaired():
    # Yahoo shows a 30x too high price for a stretch in the middle of the holding
    px = _prices()
    px.loc["2025-01-06":"2025-01-08", "X"] *= 30
    h = engine.portfolio_history(_log(), px, end=END)
    assert h["daily"]["value"].iloc[-1] == pytest.approx(20 * 121 - 100 + 5, rel=0.01)
    assert h["daily"]["value"].max() < 3000  # no 30x excursion
    assert len(h["repairs"]) >= 1


def test_repair_keeps_real_moves():
    days = pd.date_range("2025-01-01", periods=10)
    s = pd.DataFrame({"A": [10, 10, 10, 13, 13, 13, 13, 13, 13, 13.0]}, index=days)  # +30% and stays
    clean, ev = engine.repair_prices(s)
    assert clean["A"].tolist() == s["A"].tolist() and ev.empty
    s.iloc[5, 0] = 50  # one-day blip
    clean, ev = engine.repair_prices(s)
    assert days[5] not in clean["A"].dropna().index and ev["kind"].iloc[0] == "spike removed"


def test_biggest_moves_ignores_deposits_and_buys():
    h = engine.portfolio_history(_log(), _prices(), end=END)
    m = engine.biggest_moves(h, n=3)
    top = m.iloc[0]
    assert top["date"] == pd.Timestamp("2025-01-09")          # the +10% price day
    assert top["top_isin"] == "X" and top["top_eur"] == pytest.approx(220)
    dep_day = m[m["date"] == pd.Timestamp("2025-01-05")].iloc[0]
    assert dep_day["deposit"] == 1000 and dep_day["gain_eur"] == pytest.approx(100)


def test_unpriced_falls_back_to_trade_price():
    h = engine.portfolio_history(_log(), None, end=END)
    assert h["daily"].loc["2025-01-10", "holdings"] == pytest.approx(20 * 110)
    assert h["coverage"].loc["X", "no_data_at_all"]
    assert h["daily"]["priced_share"].iloc[-1] == 0


def test_xirr_simple_year():
    assert engine.xirr(["2024-01-01", "2025-01-01"], [-100, 110]) == pytest.approx(0.0998, abs=1e-3)
    assert np.isnan(engine.xirr(["2024-01-01"], [-100]))


def test_shadow_portfolio_buys_on_deposit_days():
    d = engine.portfolio_history(_log(), _prices(), end=END)["daily"]
    bench = pd.Series(np.linspace(100, 200, len(d)), index=d.index)
    sh = engine.shadow_portfolio(d, bench, d.index[0])
    units = 1000 / bench.iloc[0] + 1000 / bench.loc["2025-01-05"]
    assert sh.iloc[-1] == pytest.approx(units * 200)


def test_rebase_stats_and_calendar():
    days = pd.date_range("2023-12-31", "2025-12-31", freq="D")
    lv = pd.DataFrame({"A": np.linspace(1, 2, len(days)), "B": np.linspace(1, 1.5, len(days))}, index=days)
    rb = engine.rebase(lv * 7, days[0])
    assert rb.iloc[0].tolist() == [1, 1]
    st_ = engine.perf_stats(rb, ref="A")
    assert st_.loc["A", "total"] == pytest.approx(1.0)
    assert st_.loc["A", "cagr"] == pytest.approx(2 ** 0.5 - 1, abs=0.01)
    assert "corr" in st_ and np.isnan(st_.loc["A"].get("corr", np.nan))
    cal = engine.calendar_returns(rb)
    assert list(cal.index) == [2024, 2025]
    assert (1 + cal["A"]).prod() == pytest.approx(2.0)
    assert engine.drawdown(rb).max().max() == 0


def test_period_start():
    idx = pd.date_range("2021-02-01", "2026-11-14", freq="D")
    assert engine.period_start(idx, "YTD") == pd.Timestamp("2025-12-31")
    assert engine.period_start(idx, "All") == idx[0]
    assert engine.period_start(idx, "5Y") == pd.Timestamp("2021-11-15")