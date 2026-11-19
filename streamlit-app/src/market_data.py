"""
Security metadata (shared-config/isin_map.json) and live prices (Yahoo Finance).

Prices are best-effort: any ticker that fails is simply missing from the
result, and the Holdings page falls back to cost basis for that position.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ISIN_MAP_PATH = Path(__file__).resolve().parents[2] / "shared-config" / "isin_map.json"


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


def _quote(ticker: str) -> tuple[float, float | None, str, pd.Timestamp] | None:
    """(last price, previous close, currency, date of last trade) for one Yahoo symbol, or None."""
    import yfinance as yf  # imported lazily so the rest of the app works without it

    try:
        tk = yf.Ticker(ticker)
        h = tk.history(period="1mo", interval="1d", auto_adjust=False, actions=False)
        cur = (getattr(tk, "history_metadata", None) or {}).get("currency")
        if not cur:
            cur = tk.fast_info.currency
    except Exception as exc:  # network, delisted, rate-limited, ...
        print(f"[prices] {ticker}: {type(exc).__name__}: {exc}")
        return None
    if h is None or h.empty or "Close" not in h or not cur:
        return None
    closes = h["Close"].astype(float).dropna()
    closes = closes[closes > 0]
    if closes.empty:
        return None
    idx = pd.DatetimeIndex(closes.index)
    as_of = (idx.tz_localize(None) if idx.tz is not None else idx)[-1].normalize()
    age = (pd.Timestamp.today().normalize() - as_of).days
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
    return last, prev, cur.upper(), as_of


def _eur_rate(currency: str, cache: dict) -> float | None:
    """Units of `currency` per 1 EUR."""
    if currency == "EUR":
        return 1.0
    if currency not in cache:
        q = _quote(f"EUR{currency}=X")
        cache[currency] = q[0] if q else None
    return cache[currency]


@st.cache_data(ttl=900, show_spinner="Fetching live prices…")
def fetch_quotes(tickers_by_isin: dict[str, tuple[str, ...]]) -> pd.DataFrame:
    """
    Live prices in EUR, indexed by isin, with columns:
    ticker, currency, price_local, price_eur, prev_close_eur, as_of.
    Each ISIN's tickers are tried in order; the first with a fresh price wins.
    Cached for 15 minutes.
    """
    fx: dict[str, float | None] = {}
    rows = []
    for isin, tickers in tickers_by_isin.items():
        for t in tickers:
            q = _quote(t)
            if not q:
                continue
            last, prev, cur, as_of = q
            rate = _eur_rate(cur, fx)
            if not rate:
                continue
            rows.append({
                "isin": isin, "ticker": t, "currency": cur, "price_local": last,
                "price_eur": last / rate,
                "prev_close_eur": prev / rate if prev else None,
                "as_of": as_of,
            })
            break
    cols = ["isin", "ticker", "currency", "price_local", "price_eur", "prev_close_eur", "as_of"]
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