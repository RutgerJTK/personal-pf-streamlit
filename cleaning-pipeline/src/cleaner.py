"""
DEGIRO cleaner
==============

Turns the two Dutch DEGIRO exports (Transactions.csv + Account.csv) into one
anonymised file, ``portfolio_transactions.csv``, with exactly five columns:

    datetime      ISO 8601 string, YYYY-MM-DD HH:MM:SS
    isin          security identifier
    action        BUY | SELL | DIVIDEND | DEPOSIT
    quantity      shares, always >= 0 (0 for dividends)
    net_amount    EUR. BUY/SELL: absolute value including all fees and
                  AutoFX costs. DIVIDEND: after withholding tax (what hit
                  your account); can be negative on a reversal day.

Sources
-------
* BUY / SELL  -> Transactions.csv ("Totaal EUR" already includes fees + FX).
* DIVIDEND    -> Account.csv: dividend, dividend tax, capital repayments,
                 coupons, cash-in-lieu. Netted per ISIN per value date and
                 converted to EUR with DEGIRO's own rate (see _fx_lookup).
* DEPOSIT     -> Account.csv: iDEAL deposits (money you put in from outside).
                 isin is empty, quantity 0. Pending iDEAL reservations count
                 as deposits from the day they are booked.
* Corporate actions (rows without Order ID) are handled by the rules in
  shared-config/corporate_actions.json. Anything not covered stops the run.

Run from the repo root:  python cleaning-pipeline/src/cleaner.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = REPO_ROOT / "cleaning-pipeline" / "raw_data"
DEFAULT_OUT_DIR = REPO_ROOT / "cleaning-pipeline" / "output"
DEFAULT_RULES = REPO_ROOT / "shared-config" / "corporate_actions.json"

OUTPUT_COLUMNS = ["datetime", "isin", "action", "quantity", "net_amount"]

# --- Expected DEGIRO headers (Dutch). A mismatch stops the run. -------------
TX_HEADER = [
    "Datum", "Tijd", "Product", "ISIN", "Beurs", "Uitvoeringsplaats", "Aantal",
    "Koers", "", "Lokale waarde", "", "Waarde EUR", "Wisselkoers",
    "AutoFX Kosten", "Transactiekosten en/of kosten van derden EUR",
    "Totaal EUR", "Order ID",
]
TX_COLS = [
    "date", "time", "product", "isin", "exchange", "venue", "qty", "price",
    "price_cur", "local_value", "local_cur", "value_eur", "fx", "autofx_cost",
    "fees", "total_eur", "order_id",
]
ACC_HEADER = [
    "Datum", "Tijd", "Valutadatum", "Product", "ISIN", "Omschrijving", "FX",
    "Mutatie", "", "Saldo", "", "Order Id",
]
# In Account.csv the currency sits under "Mutatie" and the amount in the
# unnamed column next to it; same for "Saldo".
ACC_COLS = [
    "date", "time", "value_date", "product", "isin", "description", "fx",
    "cur", "amount", "bal_cur", "balance", "order_id",
]

# Account.csv descriptions that count as dividend income (net of tax).
DIVIDEND_DESCRIPTIONS = {
    "Dividend",
    "Dividendbelasting",
    "Kapitaalsuitkering",           # return of capital
    "Coupon",
    "Interest Income Distribution",  # DEGIRO re-books some dividends as this
    "Verrekening van Aandelen",     # cash in lieu of spun-off fractional shares
}
# Account.csv rows that move your own money into the account. The reservation
# rows are booked +X when you start an iDEAL payment and -X again when it
# settles (together with the real "iDEAL Deposit" +X), so they net to zero
# once settled and only a still-pending payment remains.
DEPOSIT_DESCRIPTIONS = {"iDEAL Deposit", "iDEAL storting", "Reservation iDEAL"}
# Real deposits below this (EUR) are iDEAL verification payments (e.g. the
# 1 cent test deposit when the account was opened) and are not tracked.
MIN_DEPOSIT_EUR = 1.00
# Account.csv rows with an ISIN that are knowingly not income/trades.
KNOWN_NON_INCOME_PREFIXES = (
    "Koop ", "Verkoop ", "Valuta ", "DEGIRO Transactiekosten",
    "DEGIRO Aansluitingskosten", "DEGIRO Real-Time", "ADR/GDR Externe Kosten",
    "Conversie geldmarktfonds", "Koersverandering geldmarktfonds", "Degiro Cash Sweep",
    "SPLIT AANPASSING", "SPIN-OFF", "STOCK DIVIDEND", "PRODUCTWIJZIGING",
    "WIJZIGING ISIN", "CLAIMEMISSIE", "AFWIKKELING HEFBOOMPRODUCT",
)
MAX_FX_GAP_DAYS = 90


class CleanerError(Exception):
    """Raised for anything the cleaner refuses to guess about."""


@dataclass
class Report:
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def note(self, msg: str) -> None:
        self.notes.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)


# =============================================================================
# Reading + parsing
# =============================================================================
def parse_decimal(value: str) -> float:
    """'1.234,56' -> 1234.56, '12,3550' -> 12.355, '' -> NaN."""
    s = value.strip()
    if not s:
        return float("nan")
    if "," in s:
        s = s.replace(".", "").replace(",", ".")
    return float(s)


def _read_export(path: Path, header: list[str], cols: list[str], report: Report) -> pd.DataFrame:
    if not path.exists():
        raise CleanerError(f"File not found: {path}")
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        raise CleanerError(f"{path.name} is empty")
    if rows[0] != header:
        raise CleanerError(
            f"{path.name}: header differs from what the cleaner expects.\n"
            f"  expected: {header}\n  found:    {rows[0]}"
        )
    body = []
    for line_no, row in enumerate(rows[1:], start=2):
        if not any(c.strip() for c in row):
            continue
        if len(row) != len(header):
            raise CleanerError(f"{path.name} line {line_no}: {len(row)} fields, expected {len(header)}")
        body.append(row)
    df = pd.DataFrame(body, columns=cols)
    # DEGIRO occasionally emits a wrapped fragment of a product name with no date.
    stray = df["date"].str.strip() == ""
    if stray.any():
        report.note(f"{path.name}: dropped {int(stray.sum())} stray row(s) without a date (broken line wrap)")
        df = df[~stray]
    return df.reset_index(drop=True)


def _to_dt(date_s: pd.Series, time_s: pd.Series) -> pd.Series:
    return pd.to_datetime(date_s + " " + time_s, format="%d-%m-%Y %H:%M", errors="raise")


def _iso_date(date_s: pd.Series) -> pd.Series:
    return pd.to_datetime(date_s, format="%d-%m-%Y", errors="raise").dt.strftime("%Y-%m-%d")


# =============================================================================
# Corporate-action rules
# =============================================================================
def load_rules(path: Path) -> dict:
    if not path.exists():
        raise CleanerError(f"Corporate action rules not found: {path}")
    rules = json.loads(path.read_text(encoding="utf-8"))
    for key in ("isin_changes", "keep_as_trade", "ignore"):
        rules.setdefault(key, [])
    for r in rules["isin_changes"]:
        for k in ("date", "old_isin", "new_isin"):
            if k not in r:
                raise CleanerError(f"isin_changes rule missing '{k}': {r}")
        r.setdefault("old_per_new", 1)
    for r in rules["keep_as_trade"] + rules["ignore"]:
        for k in ("date", "isin"):
            if k not in r:
                raise CleanerError(f"rule missing '{k}': {r}")
    return rules


# =============================================================================
# BUY / SELL
# =============================================================================
def build_trades(tx: pd.DataFrame, rules: dict, report: Report) -> pd.DataFrame:
    df = tx.copy()
    df["qty_n"] = df["qty"].map(parse_decimal)
    df["total_n"] = df["total_eur"].map(parse_decimal)
    df["dt"] = _to_dt(df["date"], df["time"])
    df["iso_date"] = _iso_date(df["date"])
    # File is newest-first; reverse so equal timestamps stay in booking order.
    df = df.iloc[::-1].reset_index(drop=True)

    orderless = df["order_id"].str.strip() == ""
    trades = df[~orderless].copy()
    ca = df[orderless].copy()

    if (trades["qty_n"] == 0).any():
        bad = trades[trades["qty_n"] == 0][["date", "product", "order_id"]]
        raise CleanerError(f"Trades with quantity 0 and an Order ID:\n{bad.to_string()}")
    if trades["total_n"].isna().any():
        raise CleanerError("Trades with an empty 'Totaal EUR'; cannot compute net amount")

    # ---- corporate-action rows -------------------------------------------
    used = {"isin_changes": set(), "keep_as_trade": set(), "ignore": set()}
    keep_rows, unmatched = [], []
    for idx, row in ca.iterrows():
        if row["qty_n"] == 0 and (pd.isna(row["total_n"]) or row["total_n"] == 0):
            report.note(f"ignored empty corporate-action placeholder: {row['iso_date']} {row['product']}")
            continue
        matched = False
        for i, r in enumerate(rules["isin_changes"]):
            if row["iso_date"] == r["date"] and row["isin"] in (r["old_isin"], r["new_isin"]):
                used["isin_changes"].add(i)
                matched = True
                break
        if not matched:
            for i, r in enumerate(rules["keep_as_trade"]):
                if row["iso_date"] == r["date"] and row["isin"] == r["isin"]:
                    used["keep_as_trade"].add(i)
                    keep_rows.append(idx)
                    matched = True
                    break
        if not matched:
            for i, r in enumerate(rules["ignore"]):
                if row["iso_date"] == r["date"] and row["isin"] == r["isin"]:
                    used["ignore"].add(i)
                    matched = True
                    break
        if not matched:
            unmatched.append(row)

    if unmatched:
        lines = "\n".join(
            f"  {r['iso_date']}  {r['isin']}  qty={r['qty_n']:g}  total_eur={r['total_n']}  {r['product']}"
            for r in unmatched
        )
        raise CleanerError(
            "Corporate-action rows without a rule in corporate_actions.json:\n"
            f"{lines}\n"
            "Add an 'isin_changes', 'keep_as_trade' or 'ignore' entry for each (date + ISIN)."
        )
    for kind, rule_list in (("isin_changes", rules["isin_changes"]), ("keep_as_trade", rules["keep_as_trade"]), ("ignore", rules["ignore"])):
        for i, r in enumerate(rule_list):
            if i not in used[kind]:
                report.warn(f"unused rule in {kind}: {r['date']} {r.get('isin', r.get('old_isin'))}")

    if keep_rows:
        trades = pd.concat([trades, ca.loc[keep_rows]], ignore_index=True)

    trades["action"] = np.where(trades["qty_n"] > 0, "BUY", "SELL")
    trades["quantity"] = trades["qty_n"].abs()
    trades["net_amount"] = trades["total_n"].abs()

    # ---- ISIN successions / reverse splits (apply to trades up to the date) -
    for r in sorted(rules["isin_changes"], key=lambda r: r["date"]):
        mask = (trades["isin"] == r["old_isin"]) & (trades["iso_date"] <= r["date"])
        trades.loc[mask, "quantity"] = trades.loc[mask, "quantity"] / float(r["old_per_new"])
        trades.loc[mask, "isin"] = r["new_isin"]
        report.note(f"applied {r['old_isin']} -> {r['new_isin']} (1 new = {r['old_per_new']} old) to {int(mask.sum())} trade(s)")

    trades["datetime"] = trades["dt"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return trades[OUTPUT_COLUMNS].copy()


# =============================================================================
# DIVIDENDS
# =============================================================================
def _fx_table(acc: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """DEGIRO's own conversion rates: foreign currency units per 1 EUR."""
    fx_rows = acc[
        acc["description"].isin(["Valuta Creditering", "Valuta Debitering"])
        & (acc["fx"].str.strip() != "")
        & (acc["cur"] != "EUR")
    ].copy()
    fx_rows["dt"] = _to_dt(fx_rows["date"], fx_rows["time"])
    fx_rows["rate"] = fx_rows["fx"].map(parse_decimal)
    table = {}
    for cur, grp in fx_rows.groupby("cur"):
        grp = grp.sort_values("dt")
        table[cur] = (grp["dt"].values.astype("datetime64[s]"), grp["rate"].values)
    return table


def _fx_lookup(table: dict, cur: str, ts: pd.Timestamp) -> float:
    """
    Rate of the first DEGIRO currency conversion of ``cur`` at or after ``ts``
    (that is the conversion that actually moved the dividend into EUR), else the
    closest one before. Errors if nothing within MAX_FX_GAP_DAYS.
    """
    if cur not in table:
        raise CleanerError(f"No FX conversions for currency {cur} found in Account.csv")
    times, rates = table[cur]
    t = np.datetime64(ts.to_datetime64(), "s")
    i = int(np.searchsorted(times, t, side="left"))
    gap = np.timedelta64(MAX_FX_GAP_DAYS, "D")
    if i < len(times) and times[i] - t <= gap:
        return float(rates[i])
    if i > 0 and t - times[i - 1] <= gap:
        return float(rates[i - 1])
    raise CleanerError(f"No {cur} conversion within {MAX_FX_GAP_DAYS} days of {ts}")


def build_dividends(acc: pd.DataFrame, report: Report) -> pd.DataFrame:
    df = acc.copy()

    # Safety net: rows tied to a security that we neither use nor knowingly skip.
    with_isin = df[df["isin"].str.strip() != ""]
    known = with_isin["description"].isin(DIVIDEND_DESCRIPTIONS) | with_isin["description"].str.startswith(KNOWN_NON_INCOME_PREFIXES)
    unknown = with_isin[~known]
    if len(unknown):
        kinds = unknown["description"].str.replace(r"\d[\d.,]*", "N", regex=True).value_counts()
        report.warn("Account.csv rows with an ISIN and an unrecognised description were ignored:\n    " + "\n    ".join(f"{n}x {d}" for d, n in kinds.items()))

    div = df[df["description"].isin(DIVIDEND_DESCRIPTIONS) & (df["isin"].str.strip() != "")].copy()
    if div.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    div["amount_n"] = div["amount"].map(parse_decimal)
    if div["amount_n"].isna().any():
        raise CleanerError("Dividend rows with an empty amount")
    div["booked"] = _to_dt(div["date"], div["time"])
    div["pay_date"] = pd.to_datetime(div["value_date"], format="%d-%m-%Y", errors="raise")

    grouped = (
        div.groupby(["isin", "pay_date", "cur"], as_index=False)
        .agg(net_local=("amount_n", "sum"), booked=("booked", "max"))
    )
    table = _fx_table(acc)
    net_eur = []
    for _, g in grouped.iterrows():
        if g["cur"] == "EUR":
            net_eur.append(g["net_local"])
        else:
            net_eur.append(g["net_local"] / _fx_lookup(table, g["cur"], g["booked"]))
    grouped["net_amount"] = np.round(net_eur, 2)

    zero = grouped["net_amount"].abs() < 0.005
    if zero.any():
        report.note(f"dropped {int(zero.sum())} dividend group(s) that net to zero (bookings and reversals)")
    grouped = grouped[~zero]
    neg = grouped[grouped["net_amount"] < 0]
    if len(neg):
        report.warn(f"{len(neg)} dividend group(s) are negative (a reversal booked on a different value date than the original)")

    out = pd.DataFrame({
        "datetime": grouped["pay_date"].dt.strftime("%Y-%m-%d") + " 00:00:00",
        "isin": grouped["isin"].values,
        "action": "DIVIDEND",
        "quantity": 0.0,
        "net_amount": grouped["net_amount"].values,
    })
    return out


# =============================================================================
# DEPOSITS
# =============================================================================
def build_deposits(acc: pd.DataFrame, report: Report) -> pd.DataFrame:
    """Daily net of own-money inflows (EUR). Zero-net days (settled reservations) are dropped."""
    dep = acc[acc["description"].isin(DEPOSIT_DESCRIPTIONS)].copy()
    if dep.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    if (dep["cur"] != "EUR").any():
        raise CleanerError("Deposit rows in a currency other than EUR; not supported")
    dep["amount_n"] = dep["amount"].map(parse_decimal)
    if dep["amount_n"].isna().any():
        raise CleanerError("Deposit rows with an empty amount")
    is_real = dep["description"].isin({"iDEAL Deposit", "iDEAL storting"})
    tiny = is_real & (dep["amount_n"].abs() < MIN_DEPOSIT_EUR)
    if tiny.any():
        report.note(f"ignored {int(tiny.sum())} test deposit(s) under EUR {MIN_DEPOSIT_EUR:.2f} "
                    f"(total EUR {dep.loc[tiny, 'amount_n'].sum():.2f})")
        dep = dep[~tiny]
    dep["day"] = _iso_date(dep["date"])
    daily = dep.groupby("day", as_index=False)["amount_n"].sum()
    daily["amount_n"] = daily["amount_n"].round(2)
    daily = daily[daily["amount_n"].abs() >= 0.005]
    pending = dep.loc[dep["description"] == "Reservation iDEAL", "amount_n"].sum()
    if abs(pending) >= 0.005:
        report.note(f"EUR {pending:,.2f} of iDEAL payments still pending (counted as deposited)")
    return pd.DataFrame({
        "datetime": daily["day"] + " 00:00:00",
        "isin": "",
        "action": "DEPOSIT",
        "quantity": 0.0,
        "net_amount": daily["amount_n"].values,
    })


# =============================================================================
# Validation + orchestration
# =============================================================================
def validate_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Raise if any ISIN ever holds a negative number of shares. Returns open positions."""
    t = df[df["action"].isin(["BUY", "SELL"])].copy()
    t["signed"] = np.where(t["action"] == "BUY", t["quantity"], -t["quantity"])
    t = t.sort_values("datetime", kind="stable")
    t["held"] = t.groupby("isin")["signed"].cumsum()
    bad = t[t["held"] < -1e-4]
    if len(bad):
        raise CleanerError(
            "Holdings go negative (a buy is missing or a corporate action is unhandled):\n"
            + bad[["datetime", "isin", "action", "quantity", "held"]].head(10).to_string()
        )
    final = t.groupby("isin")["signed"].sum()
    return final[final.abs() > 1e-4]


def clean(account_path: Path, transactions_path: Path, rules_path: Path = DEFAULT_RULES) -> tuple[pd.DataFrame, Report]:
    report = Report()
    tx = _read_export(transactions_path, TX_HEADER, TX_COLS, report)
    acc = _read_export(account_path, ACC_HEADER, ACC_COLS, report)
    rules = load_rules(rules_path)

    trades = build_trades(tx, rules, report)
    dividends = build_dividends(acc, report)
    deposits = build_deposits(acc, report)

    frames = [f for f in (trades, dividends, deposits) if not f.empty]
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=OUTPUT_COLUMNS)
    out = out.astype({"quantity": "float64", "net_amount": "float64"})
    out = out.sort_values("datetime", kind="stable").reset_index(drop=True)
    out["quantity"] = out["quantity"].round(6)
    out["net_amount"] = out["net_amount"].round(2)

    open_positions = validate_positions(out)
    report.note(
        f"{(out.action == 'BUY').sum()} buys, {(out.action == 'SELL').sum()} sells, "
        f"{(out.action == 'DIVIDEND').sum()} dividend payments, "
        f"{(out.action == 'DEPOSIT').sum()} deposit days (EUR {out.loc[out.action == 'DEPOSIT', 'net_amount'].sum():,.2f}); "
        f"{len(open_positions)} open position(s)"
    )
    assert list(out.columns) == OUTPUT_COLUMNS
    return out, report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Clean DEGIRO exports into portfolio_transactions.csv")
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    args = p.parse_args(argv)

    try:
        out, report = clean(args.raw_dir / "Account.csv", args.raw_dir / "Transactions.csv", args.rules)
    except CleanerError as exc:
        print(f"\nERROR: {exc}\n", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    target = args.out_dir / "portfolio_transactions.csv"
    out.to_csv(target, index=False)

    for n in report.notes:
        print(f"  - {n}")
    for w in report.warnings:
        print(f"  ! {w}")
    print(f"\nWrote {len(out)} rows to {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())