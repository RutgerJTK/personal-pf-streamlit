"""
Tests for the stale-listing guard in market_data._quote, with a fake yfinance
so no network is needed.

Run from the repo root:  pytest streamlit-app/tests
"""
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import market_data  # noqa: E402


def _install_fake_yf(monkeypatch, histories: dict, currency="EUR"):
    class FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol
            self.history_metadata = {"currency": currency}
            self.fast_info = types.SimpleNamespace(currency=currency)

        def history(self, **_):
            return histories.get(self.symbol, pd.DataFrame())

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(Ticker=FakeTicker))


def _bars(end: pd.Timestamp, closes: list[float]) -> pd.DataFrame:
    idx = pd.bdate_range(end=end, periods=len(closes), tz="Europe/Amsterdam")
    return pd.DataFrame({"Close": closes}, index=idx)


TODAY = pd.Timestamp.today().normalize()


def test_fresh_listing_gives_last_and_previous_close(monkeypatch):
    _install_fake_yf(monkeypatch, {"OK.DE": _bars(TODAY, [50.0, 51.0, 52.0, 53.0, 54.0, 55.9])})
    last, prev, cur, as_of = market_data._quote("OK.DE")
    assert (last, prev, cur) == (55.9, 54.0, "EUR")
    assert (TODAY - as_of).days <= 3


def test_old_last_trade_is_rejected(monkeypatch):
    _install_fake_yf(monkeypatch, {"DEAD.AS": _bars(TODAY - pd.Timedelta(days=30), [42.0, 42.5, 42.48])})
    assert market_data._quote("DEAD.AS") is None


def test_flat_price_is_rejected_as_stale(monkeypatch):
    _install_fake_yf(monkeypatch, {"FLAT.PA": _bars(TODAY, [67.93] * 8)})
    assert market_data._quote("FLAT.PA") is None


def test_fetch_quotes_falls_through_to_next_ticker(monkeypatch):
    _install_fake_yf(monkeypatch, {
        "DEAD.AS": _bars(TODAY - pd.Timedelta(days=30), [42.0, 42.48]),
        "OK.DE": _bars(TODAY, [55.0, 55.88]),
    })
    q = market_data.fetch_quotes.__wrapped__({"GB00BLD4ZL17": ("DEAD.AS", "OK.DE")})
    assert q.loc["GB00BLD4ZL17", "ticker"] == "OK.DE"
    assert q.loc["GB00BLD4ZL17", "price_eur"] == pytest.approx(55.88)
    assert q.loc["GB00BLD4ZL17", "prev_close_eur"] == pytest.approx(55.0)