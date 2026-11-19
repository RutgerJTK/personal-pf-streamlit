"""
Tests for cleaner.py using small synthetic exports (no real data in the repo).

Run from the repo root:  pytest cleaning-pipeline/tests
"""
import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import cleaner  # noqa: E402

FAKE_ISIN_A = "US0000000001"   # plain US stock, USD
FAKE_ISIN_B = "NL0000000002"   # plain Dutch stock, EUR
FAKE_ISIN_OLD = "US0000000003"  # reverse-split victim
FAKE_ISIN_NEW = "US0000000004"


def _tx_row(date, time, isin, qty, total, order_id="oid-1"):
    row = [""] * len(cleaner.TX_HEADER)
    row[0], row[1], row[2], row[3] = date, time, f"PRODUCT {isin}", isin
    row[6], row[15], row[16] = str(qty), total, order_id
    return row


def _acc_row(date, time, value_date, isin, desc, cur, amount, fx="", order_id=""):
    row = [""] * len(cleaner.ACC_HEADER)
    row[0], row[1], row[2], row[4], row[5] = date, time, value_date, isin, desc
    row[6], row[7], row[8], row[11] = fx, cur, amount, order_id
    return row


def _write(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


@pytest.fixture
def rules(tmp_path):
    p = tmp_path / "rules.json"
    p.write_text(json.dumps({
        "isin_changes": [{"date": "2025-06-01", "old_isin": FAKE_ISIN_OLD,
                          "new_isin": FAKE_ISIN_NEW, "old_per_new": 10}],
        "keep_as_trade": [], "ignore": [],
    }))
    return p


def _run(tmp_path, tx_rows, acc_rows, rules):
    _write(tmp_path / "Transactions.csv", cleaner.TX_HEADER, tx_rows)
    _write(tmp_path / "Account.csv", cleaner.ACC_HEADER, acc_rows)
    return cleaner.clean(tmp_path / "Account.csv", tmp_path / "Transactions.csv", rules)


def test_buy_and_sell_use_total_eur_and_are_positive(tmp_path, rules):
    tx = [  # newest first, like DEGIRO
        _tx_row("10-03-2025", "10:00", FAKE_ISIN_B, -2, "205,00", "b"),
        _tx_row("01-03-2025", "09:30", FAKE_ISIN_B, 5, "-503,00", "a"),
    ]
    out, _ = _run(tmp_path, tx, [], rules)
    assert list(out.columns) == cleaner.OUTPUT_COLUMNS
    assert out.action.tolist() == ["BUY", "SELL"]
    assert out.quantity.tolist() == [5, 2]
    assert out.net_amount.tolist() == [503.00, 205.00]
    assert out.datetime.iloc[0] == "2025-03-01 09:30:00"


def test_no_personal_columns_leak(tmp_path, rules):
    tx = [_tx_row("01-03-2025", "09:30", FAKE_ISIN_B, 5, "-503,00", "secret-order-id")]
    out, _ = _run(tmp_path, tx, [], rules)
    assert "secret-order-id" not in out.to_csv()
    assert set(out.columns) == set(cleaner.OUTPUT_COLUMNS)


def test_dividend_is_net_of_tax_and_converted_with_degiro_rate(tmp_path, rules):
    acc = [  # newest first
        _acc_row("17-10-2025", "06:35", "17-10-2025", "", "Valuta Debitering", "USD", "-5,94", fx="1,1267"),
        _acc_row("16-10-2025", "08:23", "16-10-2025", FAKE_ISIN_A, "Dividendbelasting", "USD", "-2,54"),
        _acc_row("16-10-2025", "08:23", "16-10-2025", FAKE_ISIN_A, "Dividend", "USD", "8,48"),
    ]
    out, _ = _run(tmp_path, [], acc, rules)
    out = out[out.action != "CASH"]
    assert len(out) == 1
    row = out.iloc[0]
    assert row.action == "DIVIDEND" and row.quantity == 0
    assert row.net_amount == pytest.approx(5.94 / 1.1267, abs=0.005)


def test_dividend_reversal_nets_to_zero_and_is_dropped(tmp_path, rules):
    acc = [
        _acc_row("20-05-2025", "10:00", "15-05-2025", FAKE_ISIN_B, "Dividend", "EUR", "-10,00"),
        _acc_row("15-05-2025", "10:00", "15-05-2025", FAKE_ISIN_B, "Dividend", "EUR", "10,00"),
    ]
    out, report = _run(tmp_path, [], acc, rules)
    assert (out.action == "CASH").all()   # only the cash balance row is left
    assert any("net to zero" in n for n in report.notes)


def test_reverse_split_rewrites_earlier_trades(tmp_path, rules):
    tx = [
        # corporate action rows: no order id
        _tx_row("01-06-2025", "00:00", FAKE_ISIN_OLD, -100, "50,00", order_id=""),
        _tx_row("01-06-2025", "00:00", FAKE_ISIN_NEW, 10, "-50,00", order_id=""),
        _tx_row("01-05-2025", "10:00", FAKE_ISIN_OLD, 100, "-50,00", "a"),
    ]
    out, _ = _run(tmp_path, tx, [], rules)
    assert len(out) == 1
    assert out["isin"].iloc[0] == FAKE_ISIN_NEW
    assert out.quantity.iloc[0] == 10
    assert out.net_amount.iloc[0] == 50.00


def test_unhandled_corporate_action_fails_loudly(tmp_path, rules):
    tx = [_tx_row("01-06-2025", "00:00", "US9999999999", 3, "-30,00", order_id="")]
    with pytest.raises(cleaner.CleanerError, match="without a rule"):
        _run(tmp_path, tx, [], rules)


def test_wrong_header_fails_loudly(tmp_path, rules):
    bad_header = list(cleaner.TX_HEADER)
    bad_header[0] = "Date"
    _write(tmp_path / "Transactions.csv", bad_header, [])
    _write(tmp_path / "Account.csv", cleaner.ACC_HEADER, [])
    with pytest.raises(cleaner.CleanerError, match="header differs"):
        cleaner.clean(tmp_path / "Account.csv", tmp_path / "Transactions.csv", rules)


def test_negative_holdings_fail_loudly(tmp_path, rules):
    tx = [_tx_row("01-03-2025", "09:30", FAKE_ISIN_B, -5, "500,00", "a")]
    with pytest.raises(cleaner.CleanerError, match="negative"):
        _run(tmp_path, tx, [], rules)


def test_partial_fills_with_same_order_id_are_both_kept(tmp_path, rules):
    tx = [
        _tx_row("01-03-2025", "09:30", FAKE_ISIN_B, 3, "-300,00", "same"),
        _tx_row("01-03-2025", "09:30", FAKE_ISIN_B, 2, "-200,00", "same"),
    ]
    out, _ = _run(tmp_path, tx, [], rules)
    assert out.quantity.sum() == 5
    assert out.net_amount.sum() == 500.00


def test_deposits_settled_reservation_nets_out_and_pending_counts(tmp_path, rules):
    acc = [  # newest first
        _acc_row("13-11-2025", "16:20", "13-11-2025", "", "Reservation iDEAL", "EUR", "750,00"),   # pending
        _acc_row("12-10-2025", "10:00", "12-10-2025", "", "iDEAL Deposit", "EUR", "300,00"),
        _acc_row("12-10-2025", "10:00", "12-10-2025", "", "Reservation iDEAL", "EUR", "-300,00"),
        _acc_row("10-10-2025", "09:00", "10-10-2025", "", "Reservation iDEAL", "EUR", "300,00"),
        _acc_row("01-02-2025", "09:00", "01-02-2025", "", "iDEAL storting", "EUR", "135,00"),
    ]
    out, report = _run(tmp_path, [], acc, rules)
    dep = out[out.action == "DEPOSIT"]
    assert dep.net_amount.sum() == 1185.00           # 135 + 300 + 750 pending
    assert dep.datetime.tolist() == ["2025-02-01 00:00:00", "2025-10-10 00:00:00", "2025-11-13 00:00:00"]
    assert (dep["isin"] == "").all() and (dep.quantity == 0).all()
    assert any("pending" in n for n in report.notes)


def test_one_cent_test_deposit_is_ignored(tmp_path, rules):
    acc = [
        _acc_row("01-02-2025", "10:00", "01-02-2025", "", "iDEAL storting", "EUR", "135,00"),
        _acc_row("01-02-2025", "10:00", "01-02-2025", "", "iDEAL storting", "EUR", "0,01"),
    ]
    out, report = _run(tmp_path, [], acc, rules)
    dep = out[out.action == "DEPOSIT"]
    assert dep.net_amount.tolist() == [135.00]
    assert any("test deposit" in n for n in report.notes)


def test_cash_balance_skips_sweeps_and_uses_newest_row_time(tmp_path, rules):
    acc = [  # newest first, like DEGIRO
        _acc_row("13-11-2026", "17:31", "13-11-2026", "", "Overboeking van uw geldrekening bij flatexDEGIRO Bank 81,67 EUR", "", ""),
        _acc_row("13-11-2026", "17:31", "13-11-2026", "", "Degiro Cash Sweep Transfer", "EUR", "81,67"),
        _acc_row("13-11-2026", "16:45", "13-11-2026", FAKE_ISIN_B, "DEGIRO Transactiekosten en/of kosten van derden", "EUR", "-2,00"),
        _acc_row("13-11-2026", "16:45", "13-11-2026", FAKE_ISIN_B, "Koop 1 @ 98 EUR", "EUR", "-98,00"),
        _acc_row("01-11-2026", "09:00", "01-11-2026", "", "iDEAL storting", "EUR", "150,00"),
    ]
    out, report = _run(tmp_path, [], acc, rules)
    cash = out[out.action == "CASH"]
    assert len(cash) == 1
    assert cash.net_amount.iloc[0] == 50.00          # 150 - 98 - 2; the sweep is internal
    assert cash.datetime.iloc[0] == "2026-11-13 17:31:00"
    assert cash["isin"].iloc[0] == "" and cash["quantity"].iloc[0] == 0
    assert any("cash balance" in n for n in report.notes)


def test_no_cash_row_without_account_rows(tmp_path, rules):
    tx = [_tx_row("01-03-2025", "09:30", FAKE_ISIN_B, 5, "-503,00", "a")]
    out, _ = _run(tmp_path, tx, [], rules)
    assert "CASH" not in set(out.action)