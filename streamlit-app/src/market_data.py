"""
Security metadata (shared-config/isin_map.json) and live prices (Yahoo Finance).

Prices are best-effort: any ticker that fails is simply missing from the
result, and the Holdings page falls back to cost basis for that position.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import streamlit as st

ISIN_MAP_PATH = Path(__file__).resolve().parents[2] / "shared-config" / "isin_map.json"

# yfinance logs every symbol without data as an ERROR. Spare tickers that don't
# exist are expected here, and the pages report missing prices themselves.
logging.getLogger("yfinance").setLevel(logging.CRITICAL)


@st.cache_data(ttl=3600)
def load_isin_map() -> dict:
    """ISIN -> metadata dict. Keys starting with '_' are comments and dropped."""
    try:
        raw = json.loads(ISIN_MAP_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[meta] could not read {ISIN_MAP_PATH}: {exc}")
        return {}
    return {k: v for k, v in raw.items() if not k.startswith("_")}


# A listing whose last trade is older than this, or whose price has not moved
# for FLAT_BARS trading days, is treated as dead and the next ticker is tried.
# Yahoo keeps serving the last price of illiquid / delisted lines (that is how
# BITC and GOLD ended up far below the DEGIRO price).
MAX_QUOTE_AGE_DAYS = 6
FLAT_BARS = 5
# Listings of the same ISIN that differ by more than this are flagged on the page.
DISAGREE_PCT = 0.03


def _quote(ticker: str) -> dict | None:
    """
    Latest Yahoo quote for one symbol, or None when it fails or looks stale:
    last, prev (previous session's close), currency, as_of (date of the last
    session), traded_today (whether that session is today in the exchange's
    own time zone).
    """
    import yfinance as yf  # imported lazily so the rest of the app works without it

    try:
        tk = yf.Ticker(ticker)
        h = tk.history(period="1mo", interval="1d", auto_adjust=False, actions=False)
        if h is None or h.empty or "Close" not in h:
            # Yahoo has no prices under this symbol (doesn't exist or delisted). Expected
            # for spare tickers in isin_map.json, so not worth a log line; asking for its
            # currency would only raise KeyError: 'currency'.
            return None
        cur = (getattr(tk, "history_metadata", None) or {}).get("currency")
        if not cur:
            cur = tk.fast_info.currency
    except Exception as exc:  # network, rate-limited, ...
        print(f"[prices] {ticker}: {type(exc).__name__}: {exc}")
        return None
    if not cur:
        return None
    closes = h["Close"].astype(float).dropna()
    closes = closes[closes > 0]
    if closes.empty:
        return None
    idx = pd.DatetimeIndex(closes.index)
    tz = idx.tz
    as_of = (idx.tz_localize(None) if tz is not None else idx)[-1].normalize()
    today = pd.Timestamp.now(tz=tz).tz_localize(None).normalize() if tz is not None \
        else pd.Timestamp.today().normalize()
    age = (today - as_of).days
    if age > MAX_QUOTE_AGE_DAYS:
        print(f"[prices] {ticker}: last trade {as_of.date()} is {age} days old, skipped")
        return None
    if len(closes) >= FLAT_BARS and closes.iloc[-FLAT_BARS:].nunique() == 1:
        print(f"[prices] {ticker}: price unchanged for {FLAT_BARS} days, skipped as stale")
        return None
    last = float(closes.iloc[-1])
    prev = float(closes.iloc[-2]) if len(closes) > 1 else None
    if cur in ("GBp", "GBX"):  # London quotes in pence
        last, prev, cur = last / 100, (prev / 100 if prev else prev), "GBP"
    return {"last": last, "prev": prev, "currency": cur.upper(), "as_of": as_of,
            "traded_today": as_of == today}


def _eur_rate(currency: str, cache: dict) -> tuple[float, float] | None:
    """(now, previous close): units of `currency` per 1 EUR."""
    if currency == "EUR":
        return 1.0, 1.0
    if currency not in cache:
        q = _quote(f"EUR{currency}=X")
        cache[currency] = (q["last"], q["prev"] or q["last"]) if q else None
    return cache[currency]


@st.cache_data(ttl=900, show_spinner="Fetching live prices…")
def fetch_quotes(tickers_by_isin: dict[str, tuple[str, ...]]) -> pd.DataFrame:
    """
    Live prices in EUR, indexed by isin, with columns:
    ticker, currency, price_local, price_eur, prev_close_eur, as_of,
    traded_today, check (other listings that disagree, else empty).

    Each ISIN's tickers are tried in order; the first with a fresh price wins.
    The others are still fetched to cross-check it.

    prev_close_eur is what the position was worth per share at the previous
    close, so price_eur - prev_close_eur is today's change like DEGIRO shows it:
    if the listing has not traded yet today, only the currency moved.
    Cached for 15 minutes.
    """
    fx: dict[str, tuple[float, float] | None] = {}
    rows = []
    for isin, tickers in tickers_by_isin.items():
        found = []
        for t in tickers:
            q = _quote(t)
            rates = _eur_rate(q["currency"], fx) if q else None
            if q and rates:
                found.append((t, q, rates))
        if not found:
            continue
        t, q, (rate, rate_prev) = found[0]
        price_eur = q["last"] / rate
        prev_local = q["prev"] if q["traded_today"] else q["last"]
        others = [(t2, q2["last"] / r2[0]) for t2, q2, r2 in found[1:]]
        off = [f"{t2} €{p:,.2f}" for t2, p in others if abs(p / price_eur - 1) > DISAGREE_PCT]
        rows.append({
            "isin": isin, "ticker": t, "currency": q["currency"], "price_local": q["last"],
            "price_eur": price_eur,
            "prev_close_eur": prev_local / rate_prev if prev_local else None,
            "as_of": q["as_of"], "traded_today": q["traded_today"],
            "check": (f"{t} €{price_eur:,.2f} vs " + ", ".join(off)) if off else "",
        })
    cols = ["isin", "ticker", "currency", "price_local", "price_eur", "prev_close_eur",
            "as_of", "traded_today", "check"]
    return pd.DataFrame(rows, columns=cols).set_index("isin")

# =============================================================================
# Price history (Benchmark page)
# =============================================================================
BENCHMARKS_PATH = ISIN_MAP_PATH.parent / "benchmarks.json"


@st.cache_data(ttl=3600)
def load_benchmarks() -> dict:
    """Benchmark name -> {tickers, group, note, default}. Keys starting with '_' are comments."""
    try:
        raw = json.loads(BENCHMARKS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[bench] could not read {BENCHMARKS_PATH}: {exc}")
        return {}
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def _history(ticker: str, start: str, adjusted: bool) -> tuple[pd.Series, str] | None:
    """Daily closes (tz-naive dates) and currency for one Yahoo symbol, or None."""
    import yfinance as yf

    try:
        tk = yf.Ticker(ticker)
        try:  # repair=True lets yfinance fix known Yahoo split / 100x errors (needs scipy)
            h = tk.history(start=start, interval="1d", auto_adjust=adjusted, actions=False, repair=True)
        except Exception:
            h = tk.history(start=start, interval="1d", auto_adjust=adjusted, actions=False)
        if h is None or h.empty or "Close" not in h:
            return None
        cur = (getattr(tk, "history_metadata", None) or {}).get("currency")
        if not cur:
            cur = tk.fast_info.currency
    except Exception as exc:  # network, delisted, rate-limited, ...
        print(f"[history] {ticker}: {type(exc).__name__}: {exc}")
        return None
    s = h["Close"].astype(float)
    idx = pd.DatetimeIndex(s.index)
    s.index = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
    s = s[~s.index.duplicated(keep="last")].dropna()
    s = s[s > 0]
    if s.empty or not cur:
        return None
    if cur in ("GBp", "GBX"):  # London quotes in pence
        s, cur = s / 100, "GBP"
    return s, cur.upper()


def _fx_history(cur: str, start: str, cache: dict) -> pd.Series | None:
    """Units of `cur` per 1 EUR, daily."""
    if cur not in cache:
        res = _history(f"EUR{cur}=X", start, adjusted=False)
        cache[cur] = res[0] if res else None
    return cache[cur]


@st.cache_data(ttl=6 * 3600, show_spinner="Fetching price history…")
def fetch_history(candidates: dict[str, tuple[str, ...]], start: str,
                  adjusted: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Daily closes converted to EUR, one column per key of `candidates`.

    candidates  key -> Yahoo symbols, tried in order; the first with data wins.
    adjusted    True: dividends reinvested (total return, for benchmarks).
                False: split-adjusted only (for valuing your own shares, whose
                dividends are already in the transaction log).

    Returns (prices, info). info is indexed by key with columns ticker,
    currency, first_date. Keys that fail everywhere are missing from both.
    Cached for 6 hours.
    """
    fx: dict[str, pd.Series | None] = {}
    cols, info = {}, []
    for key, tickers in candidates.items():
        for t in tickers:
            res = _history(t, start, adjusted)
            if not res:
                continue
            s, cur = res
            if cur != "EUR":
                rate = _fx_history(cur, start, fx)
                if rate is None:
                    continue
                rate = rate.reindex(rate.index.union(s.index)).sort_index().ffill().bfill().reindex(s.index)
                s = s / rate
            cols[key] = s
            info.append({"key": key, "ticker": t, "currency": cur, "first_date": s.index.min()})
            break
    prices = pd.DataFrame(cols).sort_index()
    meta_cols = ["key", "ticker", "currency", "first_date"]
    return prices, pd.DataFrame(info, columns=meta_cols).set_index("key")