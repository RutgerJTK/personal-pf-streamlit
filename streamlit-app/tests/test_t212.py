"""Tests for the Trading 212 parser and the ISIN lookup helpers (no network)."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import engine  # noqa: E402
from security_lookup import portfolio_ratios, rank_candidates, security_type  # noqa: E402
from t212_parser import ParseError, parse_trading212  # noqa: E402

SAMPLE = Path(__file__).resolve().parents[1] / "sample_data" / "trading212_sample.csv"
HEADER = ("Action,Time,ISIN,Ticker,Name,Notes,ID,No. of shares,Price / share,"
          "Currency (Price / share),Exchange rate,Result,Currency (Result),Total,Currency (Total),"
          "Withholding tax,Currency (Withholding tax),Currency conversion fee,"
          "Currency (Currency conversion fee)\n")


def _csv(*rows: str) -> bytes:
    return (HEADER + "\n".join(rows) + "\n").encode()


def test_sample_file_parses_to_six_positions_and_zero_cash():
    res = parse_trading212([(SAMPLE.name, SAMPLE.read_bytes())])
    df = res.transactions
    assert list(df.columns) == ["datetime", "isin", "action", "quantity", "net_amount"]
    assert (df["action"] == "BUY").sum() == 6
    assert df.loc[df["action"] == "DEPOSIT", "net_amount"].sum() == pytest.approx(11178.95)
    assert res.cash == pytest.approx(0.0)
    pos = engine.open_positions(df)
    assert len(pos) == 6 and (pos["quantity"] == 5).all()
    # USD buy: 5 x 262.41 / 1.1698 + 0.15% FX fee
    assert pos.loc["US0378331005", "cost_basis"] == pytest.approx(1123.28)
    assert res.securities.loc["IE00B4L5Y983", "ticker"] == "EUNL"
    assert res.securities.loc["US0378331005", "currency"] == "USD"


def test_overlapping_exports_are_deduplicated():
    raw = SAMPLE.read_bytes()
    res = parse_trading212([("2026.csv", raw), ("again.csv", raw)])
    assert res.report.duplicates == 7
    assert len(res.transactions) == 7


def test_sell_dividend_withdrawal_interest_and_ignored_rows():
    res = parse_trading212([("x.csv", _csv(
        'Deposit,2024-01-02 10:00:00,,,,"Bank Transfer",d1,,,,,,,1000.00,"EUR",,,,',
        'Market buy,2024-01-03 15:30:00,US0378331005,AAPL,"Apple",,b1,4,180,USD,1.09,,,661.47,"EUR",,,0.99,"EUR"',
        'Limit sell,2024-06-03 15:30:00,US0378331005,AAPL,"Apple",,s1,1,200,USD,1.08,10.5,"EUR",184.90,"EUR",,,0.28,"EUR"',
        'Dividend (Ordinary),2024-05-16 12:00:00,US0378331005,AAPL,"Apple",,,3,0.21,USD,1.08,,,0.50,"EUR",0.09,USD,,',
        'Interest on cash,2024-06-30 23:00:00,,,,,i1,,,,,,,1.23,"EUR",,,,',
        'Withdrawal,2024-07-01 10:00:00,,,,,w1,,,,,,,-100.00,"EUR",,,,',
        'Currency conversion,2024-07-02 10:00:00,,,,,c1,,,,,,,5.00,"EUR",,,,',
        'Stock split open,2024-07-03 10:00:00,US0378331005,AAPL,"Apple",,x1,8,,,,,,0.00,"EUR",,,,',
    ))])
    df = res.transactions.set_index("action")
    assert df.loc["SELL", "net_amount"] == pytest.approx(184.90)
    assert df.loc["DIVIDEND", "net_amount"] == pytest.approx(0.50)
    assert df.loc["DIVIDEND", "quantity"] == 0
    assert df.loc["FEE", "net_amount"] == pytest.approx(1.23)
    assert sorted(res.transactions.loc[res.transactions["action"] == "DEPOSIT", "net_amount"]) == [-100.0, 1000.0]
    assert res.report.corporate == {"Stock split open": 1}
    assert "Currency conversion" not in res.report.skipped
    assert res.cash == pytest.approx(1000 - 661.47 + 184.90 + 0.50 + 1.23 - 100)


def test_old_format_with_currency_in_header_and_day_first_dates():
    old = ("Action,Time,ISIN,Ticker,Name,No. of shares,Price / share,Currency (Price / share),"
           "Exchange rate,Result (EUR),Total (EUR)\n"
           "Market buy,25/09/2020 13:34,US88579Y1010,MMM,3M,1.0938659,159.52,USD,1.16329,,150\n")
    res = parse_trading212([("old.csv", old.encode())])
    row = res.transactions.iloc[0]
    assert row["datetime"] == pd.Timestamp("2020-09-25 13:34")
    assert row["net_amount"] == 150 and row["quantity"] == pytest.approx(1.0938659)


def test_non_euro_account_is_refused():
    with pytest.raises(ParseError, match="euro"):
        parse_trading212([("gbp.csv", _csv(
            'Deposit,2024-01-02 10:00:00,,,,,d1,,,,,,,100.00,"GBP",,,,'))])


def test_not_a_trading212_file():
    with pytest.raises(ParseError, match="Trading 212"):
        parse_trading212([("degiro.csv", b"Datum,Tijd,Product\n01-01-2024,10:00,X\n")])


def test_selling_more_than_bought_warns():
    res = parse_trading212([("x.csv", _csv(
        'Market sell,2024-06-03 15:30:00,US0378331005,AAPL,Apple,,s1,1,200,USD,1.08,,,184.90,EUR,,,,'))])
    assert any("earlier period" in w for w in res.report.warnings)


def test_rank_candidates_prefers_trading_currency_and_ticker():
    hits = [
        {"symbol": "VUSA.L", "exchange": "LSE", "type": "ETF", "name": ""},
        {"symbol": "VUSA.DE", "exchange": "GER", "type": "ETF", "name": ""},
        {"symbol": "VUSA.AS", "exchange": "AMS", "type": "ETF", "name": ""},
        {"symbol": "VUAA.AS", "exchange": "AMS", "type": "ETF", "name": ""},
    ]
    ranked = rank_candidates("IE00B3XXRP09", "VUSA", "EUR", hits)
    assert ranked[:3] == ["VUSA.AS", "VUSA.DE", "VUAA.AS"]
    assert ranked.index("VUSA.L") > ranked.index("VUAA.AS")
    # search failed: fall back to guesses from the Trading 212 ticker
    assert rank_candidates("US0378331005", "AAPL", "USD", []) == ["AAPL"]
    assert rank_candidates("IE00B4L5Y983", "EUNL", "EUR", [])[:2] == ["EUNL.AS", "EUNL.DE"]


def test_security_type():
    assert security_type([{"type": "ETF"}], "x") == "ETF"
    assert security_type([], "iShares Core MSCI World (Acc)") == "ETF"
    assert security_type([], "Apple") == "Stock"


def test_portfolio_ratios_harmonic_pe_and_coverage():
    values = pd.Series([100.0, 100.0, 200.0], index=["a", "b", "c"])
    fund = pd.DataFrame({"pe": [10.0, 40.0, None], "pb": [2.0, -1.0, None],
                         "beta": [1.0, 2.0, None]}, index=["a", "b", "c"])
    r = portfolio_ratios(values, fund)
    assert r["pe"] == pytest.approx(16.0)          # 200 / (10 + 2.5)
    assert r["pe_coverage"] == pytest.approx(0.5)
    assert r["pb"] == pytest.approx(2.0) and r["pb_coverage"] == pytest.approx(0.25)
    assert r["beta"] == pytest.approx(1.5)


# --- ETF valuation and beta --------------------------------------------------
import json  # noqa: E402

import numpy as np  # noqa: E402

from security_lookup import betas, etf_proxy, fund_ratios_from  # noqa: E402

RULES = json.loads((Path(__file__).resolve().parents[2] / "shared-config" / "etf_proxies.json")
                   .read_text())["rules"]


def test_fund_ratios_inverted_by_yahoo_are_flipped():
    assert fund_ratios_from(0.04, 0.2) == pytest.approx((25.0, 5.0))
    assert fund_ratios_from(25.0, 5.0) == (25.0, 5.0)
    assert fund_ratios_from(None, 0.25) == pytest.approx((None, 4.0))
    assert fund_ratios_from(None, 0.9) == (None, 0.9)          # a real value-fund P/B
    assert fund_ratios_from(-3.0, None) == (None, None)


def test_etf_proxy_matching():
    assert etf_proxy(["Vanguard S&P 500 (Dist)"], RULES)["proxy"] == "IVV"
    assert etf_proxy(["iShares Core MSCI World (Acc)"], RULES)["proxy"] == "URTH"
    assert etf_proxy(["", "Vanguard FTSE All-World UCITS ETF"], RULES)["proxy"] == "VT"
    assert etf_proxy(["Invesco S&P 500 Equal Weight"], RULES)["proxy"] == "RSP"
    assert etf_proxy(["iShares Core MSCI EM IMI (Acc)"], RULES)["proxy"] == "IEMG"
    assert etf_proxy(["Some Gold ETC"], RULES) is None


def test_betas_recover_known_beta():
    rng = np.random.default_rng(1)
    days = pd.bdate_range("2023-01-02", periods=800)
    m = rng.normal(0, 0.01, len(days))
    bench = pd.Series(100 * np.cumprod(1 + m), index=days)
    prices = pd.DataFrame({
        "double": 50 * np.cumprod(1 + 2 * m),
        "noise": 20 * np.cumprod(1 + rng.normal(0, 0.01, len(days))),
        "short": np.r_[[np.nan] * 700, 10 * np.cumprod(1 + m[700:])],
    }, index=days)
    b = betas(prices, bench)
    assert b.loc["double", "beta"] == pytest.approx(2.0, abs=0.1)
    assert abs(b.loc["noise", "beta"]) < 0.3
    assert np.isnan(b.loc["short", "beta"])                    # under 26 weeks of overlap


def test_etf_valuation_fills_missing_ratio_from_proxy(monkeypatch):
    import security_lookup as sl
    data = {"EUNL.DE": {"pe": 20.4, "pb": None}, "URTH": {"pe": 22.0, "pb": 3.7},
            "VUSA.AS": {"pe": None, "pb": None}, "IVV": {"pe": 27.0, "pb": 5.6}}
    monkeypatch.setattr(sl, "fetch_fund_ratios", lambda s: data.get(s, {"pe": None, "pb": None}))
    v = sl.etf_valuation("EUNL.DE", ["iShares Core MSCI World (Acc)"], RULES)
    assert (v["pe"], v["pb"]) == (20.4, 3.7) and v["source"] == "EUNL.DE fund data + URTH"
    v = sl.etf_valuation("VUSA.AS", ["Vanguard S&P 500 (Dist)"], RULES)
    assert (v["pe"], v["pb"]) == (27.0, 5.6) and v["source"].startswith("via IVV")
    v = sl.etf_valuation("GOLD.AS", ["Some Gold ETC"], RULES)
    assert v["pe"] is None and "not recognised" in v["source"]