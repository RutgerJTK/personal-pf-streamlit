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


def _bars(end: pd.Timestamp, closes: list[float], tz="Europe/Amsterdam") -> pd.DataFrame:
    idx = pd.bdate_range(end=end, periods=len(closes), tz=tz)
    return pd.DataFrame({"Close": closes}, index=idx)


TODAY = pd.Timestamp.today().normalize()
LAST_WEEKDAY = TODAY if TODAY.weekday() < 5 else TODAY - pd.offsets.BDay(1)


def test_fresh_listing_gives_last_and_previous_close(monkeypatch):
    _install_fake_yf(monkeypatch, {"OK.DE": _bars(LAST_WEEKDAY, [50.0, 51.0, 52.0, 53.0, 54.0, 55.9])})
    q = market_data._quote("OK.DE")
    assert (q["last"], q["prev"], q["currency"]) == (55.9, 54.0, "EUR")
    assert (TODAY - q["as_of"]).days <= 3


def test_old_last_trade_is_rejected(monkeypatch):
    _install_fake_yf(monkeypatch, {"DEAD.AS": _bars(TODAY - pd.Timedelta(days=30), [42.0, 42.5, 42.48])})
    assert market_data._quote("DEAD.AS") is None


def test_flat_price_is_rejected_as_stale(monkeypatch):
    _install_fake_yf(monkeypatch, {"FLAT.PA": _bars(LAST_WEEKDAY, [67.93] * 8)})
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
    assert q.loc["GB00BLD4ZL17", "check"] == ""


def test_not_traded_today_moves_only_with_fx(monkeypatch):
    # US stock whose last session was yesterday; EUR/USD went from 1.10 to 1.12 since
    yesterday = TODAY - pd.Timedelta(days=1)
    _install_fake_yf(monkeypatch, {
        "MAIN": _bars(yesterday, [50.0, 52.0, 55.0, 56.0, 55.87], tz="America/New_York"),
        "EURUSD=X": _bars(TODAY, [1.09, 1.08, 1.10, 1.12], tz="Europe/London"),
    }, currency="USD")
    q = market_data.fetch_quotes.__wrapped__({"US56035L1044": ("MAIN",)}).loc["US56035L1044"]
    assert not q["traded_today"]
    assert q["price_eur"] == pytest.approx(55.87 / 1.12)
    assert q["prev_close_eur"] == pytest.approx(55.87 / 1.10)   # yesterday's price move left out


def test_disagreeing_listings_are_flagged(monkeypatch):
    _install_fake_yf(monkeypatch, {
        "GOLD.AS": _bars(TODAY, [134.0, 134.3]),
        "GLDA.DE": _bars(TODAY, [124.0, 122.7]),
    })
    q = market_data.fetch_quotes.__wrapped__({"FR0013416716": ("GOLD.AS", "GLDA.DE")})
    assert q.loc["FR0013416716", "ticker"] == "GOLD.AS"
    assert "GLDA.DE" in q.loc["FR0013416716", "check"]