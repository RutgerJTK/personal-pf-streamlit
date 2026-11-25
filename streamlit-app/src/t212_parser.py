"""
Trading 212 history export -> the app's standard transaction format.

Pure pandas, no Streamlit, no disk, no network: the generic dashboard calls
this on files uploaded in the browser and keeps the result in session state.

Output (same five columns as the DEGIRO cleaner, so engine.py works on it):

    datetime    pandas Timestamp
    isin        security identifier ('' for cash rows)
    action      BUY | SELL | DIVIDEND | DEPOSIT | FEE
    quantity    shares, >= 0 (0 for non-trades)
    net_amount  EUR. BUY/SELL: absolute 'Total' (incl. FX fee and taxes).
                DIVIDEND: 'Total' (net, after withholding tax).
                DEPOSIT: + money in, - money out (withdrawals, card spending).
                FEE: signed, + = income (interest on cash, cashback).

Format notes
------------
* Trading 212 exports at most 12 months per file, so several files can be
  uploaded together. Rows are de-duplicated on the 'ID' column.
* Columns are matched by NAME, never by position: Trading 212 has reordered
  and added columns over the years. Older files put the currency in the header
  ('Total (EUR)'); those are normalised to 'Total' + 'Currency (Total)'.
* Only accounts in EUR are supported for now.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field

import pandas as pd

OUTPUT_COLUMNS = ["datetime", "isin", "action", "quantity", "net_amount"]
REQUIRED = ["Action", "Time", "Total"]
ACCOUNT_CURRENCY = "EUR"

# Exact action names (lower case) -> (our action, sign applied to |Total|).
# Buys/sells/dividends are matched by pattern in _classify().
CASH_ACTIONS = {
    "deposit": ("DEPOSIT", 1.0),
    "withdrawal": ("DEPOSIT", -1.0),
    "card debit": ("DEPOSIT", -1.0),          # Trading 212 card spends from the same cash
    "card credit": ("DEPOSIT", 1.0),          # refunds on card payments
    "interest on cash": ("FEE", 1.0),
    "lending interest": ("FEE", 1.0),
    "spending cashback": ("FEE", 1.0),
}
# Internal bookkeeping with no effect on the EUR balance or holdings.
IGNORED_ACTIONS = {"currency conversion"}
# Actions that change the number of shares without a normal buy/sell.
CORPORATE_HINTS = ("split", "spin", "merger", "rights", "custody", "stock distribution",
                   "result adjustment", "delisting")


class ParseError(ValueError):
    """The file can't be used at all. The message is safe to show to the user."""


@dataclass
class ParseReport:
    files: int = 0
    rows_read: int = 0
    duplicates: int = 0
    skipped: dict[str, int] = field(default_factory=dict)       # action -> count
    corporate: dict[str, int] = field(default_factory=dict)     # action -> count
    warnings: list[str] = field(default_factory=list)


@dataclass
class ParseResult:
    transactions: pd.DataFrame          # OUTPUT_COLUMNS
    securities: pd.DataFrame            # indexed by isin: ticker, name, currency
    cash: float                         # EUR cash left, derived from the flows
    report: ParseReport


def _num(s: pd.Series) -> pd.Series:
    """'1,234.50' / '' -> float (NaN for blanks)."""
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False).str.strip()
                         .replace({"": None, "nan": None}), errors="coerce")


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Old exports: 'Total (EUR)' -> 'Total' plus a 'Currency (Total)' column."""
    df = df.rename(columns=lambda c: str(c).strip().lstrip("﻿"))
    for col in list(df.columns):
        m = re.fullmatch(r"(.+?) \(([A-Z]{3})\)", col)
        if m and not col.startswith("Currency ("):
            base, cur = m.groups()
            if base not in df.columns:
                df = df.rename(columns={col: base})
                cur_col = f"Currency ({base})"
                if cur_col not in df.columns:
                    df[cur_col] = cur
    return df


def _read_one(name: str, raw: bytes | str) -> pd.DataFrame:
    text = raw.decode("utf-8-sig", errors="replace") if isinstance(raw, bytes) else raw
    try:
        df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ParseError(f"{name}: not a readable CSV file.") from exc
    df = _normalise_columns(df)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ParseError(
            f"{name}: this doesn't look like a Trading 212 history export "
            f"(missing column(s): {', '.join(missing)})."
        )
    for col in ["ISIN", "Ticker", "Name", "ID", "No. of shares", "Currency (Total)",
                "Currency (Price / share)"]:
        if col not in df.columns:
            df[col] = ""
    return df


def _parse_time(s: pd.Series) -> pd.Series:
    """ISO 'YYYY-MM-DD HH:MM:SS[.fff]' (current) or 'DD/MM/YYYY HH:MM' (old)."""
    s = s.str.strip()
    iso = pd.to_datetime(s, format="ISO8601", errors="coerce")
    old = pd.to_datetime(s.where(iso.isna()), dayfirst=True, errors="coerce")
    return iso.fillna(old)


def _classify(action: str) -> tuple[str | None, float]:
    """Return (our action, sign) or (None, 0) for rows we don't use."""
    a = action.strip().lower()
    if re.search(r"\bbuy\b", a):
        return "BUY", 1.0
    if re.search(r"\bsell\b", a):
        return "SELL", 1.0
    if a.startswith("dividend"):
        return "DIVIDEND", 1.0
    return CASH_ACTIONS.get(a, (None, 0.0))


def parse_trading212(files: list[tuple[str, bytes | str]]) -> ParseResult:
    """Parse one or more Trading 212 history exports. Raises ParseError when
    nothing usable can be read; smaller problems go to result.report.warnings."""
    if not files:
        raise ParseError("No files uploaded.")
    report = ParseReport(files=len(files))
    frames = [_read_one(name, raw) for name, raw in files]
    raw = pd.concat(frames, ignore_index=True)
    report.rows_read = len(raw)

    # --- De-duplicate overlapping exports -------------------------------------
    has_id = raw["ID"].str.strip() != ""
    keyed = raw[has_id].drop_duplicates(subset=["Action", "ID"])
    unkeyed = raw[~has_id].drop_duplicates()
    df = pd.concat([keyed, unkeyed], ignore_index=True)
    report.duplicates = len(raw) - len(df)

    df["_time"] = _parse_time(df["Time"])
    bad_time = df["_time"].isna()
    if bad_time.any():
        report.warnings.append(f"{int(bad_time.sum())} row(s) with an unreadable date were skipped.")
        df = df[~bad_time]
    if df.empty:
        raise ParseError("The file(s) contain no rows with a valid date.")

    df["_total"] = _num(df["Total"])
    df["_qty"] = _num(df["No. of shares"]).fillna(0.0)
    cls = df["Action"].map(_classify)
    df["_action"] = cls.str[0]
    df["_sign"] = cls.str[1]

    # --- Rows we don't turn into transactions ---------------------------------
    unused = df[df["_action"].isna()]
    for action, n in unused["Action"].value_counts().items():
        a = action.strip().lower()
        if a in IGNORED_ACTIONS:
            continue
        if any(h in a for h in CORPORATE_HINTS):
            report.corporate[action] = int(n)
        else:
            report.skipped[action] = int(n)
    df = df[df["_action"].notna()].copy()
    if df.empty:
        raise ParseError("No buys, sells, dividends or deposits found in the file(s).")

    # --- Account currency -----------------------------------------------------
    cur = df["Currency (Total)"].str.strip().str.strip('"').replace("", ACCOUNT_CURRENCY)
    other = sorted(set(cur) - {ACCOUNT_CURRENCY})
    if other:
        raise ParseError(
            f"Only euro accounts are supported for now; this export has amounts in "
            f"{', '.join(other)}."
        )

    missing_total = df["_total"].isna()
    if missing_total.any():
        report.warnings.append(f"{int(missing_total.sum())} row(s) without a Total were skipped.")
        df = df[~missing_total]

    is_trade = df["_action"].isin(["BUY", "SELL"])
    no_isin = is_trade & (df["ISIN"].str.strip() == "")
    if no_isin.any():
        report.warnings.append(f"{int(no_isin.sum())} trade(s) without an ISIN were skipped.")
        df = df[~no_isin]
        is_trade = df["_action"].isin(["BUY", "SELL"])

    amount = df["_total"].abs()
    out = pd.DataFrame({
        "datetime": df["_time"].values,
        "isin": df["ISIN"].str.strip().values,
        "action": df["_action"].values,
        "quantity": df["_qty"].abs().where(is_trade, 0.0).values,
        "net_amount": (amount * df["_sign"]).where(df["_action"] != "DIVIDEND", df["_total"]).values,
    })
    out.loc[~out["action"].isin(["BUY", "SELL", "DIVIDEND"]), "isin"] = ""
    out = out.sort_values("datetime", kind="stable").reset_index(drop=True)

    # --- Sanity: never sell more than held (missing older export?) ------------
    t = out[out["action"].isin(["BUY", "SELL"])]
    signed = t["quantity"].where(t["action"] == "BUY", -t["quantity"])
    running = signed.groupby(t["isin"]).cumsum()
    short = sorted(t.loc[running < -1e-4, "isin"].unique())
    if short:
        report.warnings.append(
            "More shares sold than bought for " + ", ".join(short) + ". An export covering "
            "an earlier period is probably missing; upload all years back to the account opening."
        )

    # --- Securities seen in the file -------------------------------------------
    trades = df[df["ISIN"].str.strip() != ""].sort_values("_time")
    securities = (
        trades.groupby(trades["ISIN"].str.strip())
        .agg(ticker=("Ticker", "last"), name=("Name", "last"),
             currency=("Currency (Price / share)", "last"))
    )
    securities.index.name = "isin"
    securities = securities.apply(lambda c: c.str.strip().str.strip('"'))

    flow = out.loc[out["action"] != "BUY", "net_amount"].sum() - out.loc[out["action"] == "BUY", "net_amount"].sum()
    return ParseResult(out[OUTPUT_COLUMNS], securities, round(float(flow), 2), report)