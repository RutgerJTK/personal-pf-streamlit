import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import engine  # noqa: E402

TODAY = pd.Timestamp("2026-11-15")


def _log(rows):
    """rows: (date, isin, action, quantity, net_amount)"""
    d = pd.DataFrame(rows, columns=["datetime", "isin", "action", "quantity", "net_amount"])
    d["datetime"] = pd.to_datetime(d["datetime"])
    return d.sort_values("datetime", kind="stable").reset_index(drop=True)


def _quarterly(isin, start, n, amount):
    dates = pd.date_range(start, periods=n, freq="3MS") + pd.Timedelta(days=9)
    return [(d, isin, "DIVIDEND", 0.0, amount) for d in dates]


def test_by_year_totals_and_current_year_vs_same_period():
    df = _log([
        ("2024-01-01", "Q", "BUY", 10, 100.0),
        ("2025-03-10", "Q", "DIVIDEND", 0, 2.0),
        ("2025-12-10", "Q", "DIVIDEND", 0, 2.0),
        ("2026-03-10", "Q", "DIVIDEND", 0, 3.0),
    ])
    y = engine.dividends_by_year(df, TODAY)
    assert y.loc[2025, "received"] == 4 and y.loc[2025, "payments"] == 2
    assert y.loc[2026, "partial"]
    # 2026 so far (3) vs 1 Jan - 15 Nov 2025 (2), not vs the full 2025 (4)
    assert y.loc[2026, "growth"] == pytest.approx(0.5)
    assert y.loc[2026, "per_month"] == pytest.approx(3 / 11)


def test_by_month_fills_gaps_and_averages_after_12_months():
    df = _log([("2025-01-01", "Q", "BUY", 1, 1.0)] + _quarterly("Q", "2025-01-01", 8, 3.0))
    m = engine.dividends_by_month(df, TODAY)
    assert m.index[0] == pd.Timestamp("2025-01-01") and m.index[-1] == pd.Timestamp("2026-11-01")
    assert m["avg_12m"].iloc[:11].isna().all()
    assert m.loc["2025-12-01", "avg_12m"] == pytest.approx(12 / 12)


def test_forecast_repeats_last_year_scaled_to_shares_held_now():
    df = _log([("2024-06-01", "Q", "BUY", 10, 100.0), ("2026-10-20", "Q", "BUY", 10, 100.0)]
              + _quarterly("Q", "2024-07-01", 10, 1.0))      # 0.10 per share, last on 10 Oct 2026
    f = engine.dividend_forecast(df, today=TODAY)
    assert len(f) == 4                                        # quarterly, one year ahead
    assert f["amount"].tolist() == pytest.approx([2.0] * 4)  # 0.10 x 20 shares held now
    assert (f["date"] > TODAY).all() and (f["date"] <= TODAY + pd.Timedelta(days=365)).all()


def test_forecast_does_not_double_count_a_drifting_pay_date():
    rows = [("2024-01-01", "Q", "BUY", 10, 100.0)]
    rows += [(d, "Q", "DIVIDEND", 0, 1.0) for d in
             ["2025-11-01", "2026-02-01", "2026-05-01", "2026-08-01", "2026-10-28"]]  # 5 in the lookback
    f = engine.dividend_forecast(_log(rows), today=TODAY)
    assert len(f) == 4                     # 1 Nov 2025 already recurred on 28 Oct 2026


def test_forecast_skips_sold_and_stopped_positions():
    df = _log([
        ("2023-01-01", "SOLD", "BUY", 10, 100.0), ("2026-01-01", "SOLD", "SELL", 10, 120.0),
        ("2023-01-01", "STOP", "BUY", 10, 100.0),
    ] + _quarterly("SOLD", "2024-01-01", 8, 1.0) + _quarterly("STOP", "2023-01-01", 4, 1.0))
    assert engine.dividend_forecast(df, today=TODAY).empty


def test_forecast_young_position_with_few_payments_repeats_only_those():
    # two payments 3 months apart could be quarterly or half-yearly: don't guess
    df = _log([("2026-05-01", "Y", "BUY", 10, 100.0),
               ("2026-06-17", "Y", "DIVIDEND", 0, 3.0), ("2026-09-24", "Y", "DIVIDEND", 0, 2.0)])
    f = engine.dividend_forecast(df, today=TODAY)
    assert f["date"].dt.month.tolist() == [6, 9] and f["amount"].sum() == pytest.approx(5.0)


def test_forecast_young_monthly_payer_is_extended():
    rows = [("2026-04-01", "M", "BUY", 10, 100.0)]
    rows += [(f"2026-{m:02d}-15", "M", "DIVIDEND", 0, 1.0) for m in range(4, 11)]
    rows += [("2026-06-29", "M", "DIVIDEND", 0, 2.0)]        # a supplemental doesn't break the rhythm
    f = engine.dividend_forecast(_log(rows), today=TODAY)
    assert len(f) == 12                                       # 15 Dec 2026 .. 15 Nov 2027
    assert f["date"].iloc[0] == pd.Timestamp("2026-12-15")
    assert f["amount"].iloc[0] == pytest.approx(1.0)


def test_calendar_and_per_holding():
    df = _log([("2024-06-01", "Q", "BUY", 10, 100.0)] + _quarterly("Q", "2024-07-01", 10, 1.0))
    f = engine.dividend_forecast(df, today=TODAY)
    cal = engine.dividend_calendar(f, TODAY)
    assert cal.shape == (1, 12) and cal.columns[0] == pd.Timestamp("2026-11-01")
    assert cal.to_numpy().sum() == pytest.approx(4.0)
    h = engine.dividends_by_holding(df, f, today=TODAY)
    assert h.loc["Q", "total"] == 10 and h.loc["Q", "forward"] == 4
    assert h.loc["Q", "yield_on_cost"] == pytest.approx(0.04)


# --- Dividend growth ------------------------------------------------------------
def test_like_for_like_growth_ignores_seasonality_and_share_count():
    # small March + big September payment; both raised 10% in 2026. Shares doubled in between.
    rows = [("2024-01-01", "S", "BUY", 10, 100.0), ("2025-11-01", "S", "BUY", 10, 100.0),
            ("2025-03-10", "S", "DIVIDEND", 0, 1.0), ("2025-09-10", "S", "DIVIDEND", 0, 5.0),   # 10 shares
            ("2026-03-10", "S", "DIVIDEND", 0, 2.2), ("2026-09-10", "S", "DIVIDEND", 0, 11.0)]  # 20 shares
    g, s = engine.dividend_growth(_log(rows), today=TODAY)
    assert g.loc["S", "ttm_growth"] == pytest.approx(0.10)
    assert g.loc["S", "matched"] == 2 and g.loc["S", "status"] == "raised"
    assert s["organic"] == pytest.approx(0.10) and s["raised"] == 1


def test_buy_just_before_pay_date_is_not_a_fake_cut():
    rows = [("2024-01-01", "B", "BUY", 10, 100.0),
            ("2025-06-10", "B", "DIVIDEND", 0, 5.0),
            ("2026-06-01", "B", "BUY", 10, 100.0),           # after the ex-date, before paying
            ("2026-06-10", "B", "DIVIDEND", 0, 5.0)]          # still for 10 shares: looks like -50%
    h = engine.dividend_per_share_history(_log(rows), today=TODAY)
    assert not h["clear"].iloc[-1]
    g, _ = engine.dividend_growth(_log(rows), today=TODAY)
    assert g.loc["B", "status"] == "unknown"


def test_drifting_annual_payer_is_still_matched():
    rows = [("2024-01-01", "A", "BUY", 10, 100.0),
            ("2025-05-30", "A", "DIVIDEND", 0, 10.0), ("2026-04-29", "A", "DIVIDEND", 0, 15.0)]
    g, _ = engine.dividend_growth(_log(rows), today=TODAY)
    assert g.loc["A", "ttm_growth"] == pytest.approx(0.5)      # 31 days apart: fine for a yearly payer
    assert g.loc["A", "basis"] == "12 months"
    a = engine.annual_dividend_per_share(engine.dividend_per_share_history(_log(rows), today=TODAY),
                                         "A", _log(rows), TODAY)
    assert a["partial"].tolist() == [False, True]               # 2025 held all year, 2026 still running


def test_annual_dps_marks_partial_years():
    df = _log([("2024-06-01", "Q", "BUY", 10, 100.0)] + _quarterly("Q", "2024-07-01", 10, 1.0))
    a = engine.annual_dividend_per_share(engine.dividend_per_share_history(df, today=TODAY), "Q", df, TODAY)
    assert a["partial"].tolist() == [True, False, True]          # bought mid-2024, 2026 still running
    assert a.loc[2025, "dps"] == pytest.approx(0.4) and pd.isna(a.loc[2025, "growth"])


def test_unclear_latest_payment_falls_back_to_calendar_years():
    rows = [("2023-01-01", "H", "BUY", 10, 100.0),
            ("2024-05-15", "H", "DIVIDEND", 0, 10.0), ("2025-05-15", "H", "DIVIDEND", 0, 12.0),
            ("2026-05-01", "H", "BUY", 10, 100.0), ("2026-05-12", "H", "DIVIDEND", 0, 30.0)]  # unclear
    g, s = engine.dividend_growth(_log(rows), today=TODAY)
    assert g.loc["H", "ttm_growth"] == pytest.approx(0.2)
    assert g.loc["H", "basis"] == "2025 vs year before" and g.loc["H", "status"] == "raised"
    assert pd.isna(s["organic"])          # nothing matched like-for-like, so no organic figure