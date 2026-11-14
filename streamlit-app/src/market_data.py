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


def _quote(ticker: str) -> tuple[float, float | None, str] | None:
    """(last price, previous close, currency) for one Yahoo symbol, or None."""
    import yfinance as yf  # imported lazily so the rest of the app works without it

    try:
        fi = yf.Ticker(ticker).fast_info
        last = fi.last_price
        cur = fi.currency
        try:
            prev = fi.previous_close
        except Exception:
            prev = None
    except Exception as exc:  # network, delisted, rate-limited, ...
        print(f"[prices] {ticker}: {type(exc).__name__}: {exc}")
        return None
    if last is None or pd.isna(last) or last <= 0 or not cur:
        return None
    if cur in ("GBp", "GBX"):  # London quotes in pence
        last, prev, cur = last / 100, (prev / 100 if prev else prev), "GBP"
    return float(last), (float(prev) if prev and not pd.isna(prev) else None), cur.upper()


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
    ticker, currency, price_local, price_eur, prev_close_eur.
    Each ISIN's tickers are tried in order; the first that answers wins.
    Cached for 15 minutes.
    """
    fx: dict[str, float | None] = {}
    rows = []
    for isin, tickers in tickers_by_isin.items():
        for t in tickers:
            q = _quote(t)
            if not q:
                continue
            last, prev, cur = q
            rate = _eur_rate(cur, fx)
            if not rate:
                continue
            rows.append({
                "isin": isin, "ticker": t, "currency": cur, "price_local": last,
                "price_eur": last / rate,
                "prev_close_eur": prev / rate if prev else None,
            })
            break
    cols = ["isin", "ticker", "currency", "price_local", "price_eur", "prev_close_eur"]
    return pd.DataFrame(rows, columns=cols).set_index("isin")