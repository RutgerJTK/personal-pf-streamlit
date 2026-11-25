"""
Find Yahoo Finance symbols for unknown ISINs, fetch valuation ratios for
stocks and ETFs, and measure beta from price history.

Used by the generic dashboard, where the portfolio is not known in advance
(the personal pages use the hand-made shared-config/isin_map.json instead).

Caching: every cached function here is keyed only on ISINs, tickers or
symbols, never on anyone's holdings or amounts, so the shared Streamlit cache
cannot leak one user's data to another. Keep it that way.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import streamlit as st

# Yahoo exchange code -> (suffix, trading currency). Order inside each currency
# is the preference: the most liquid / best-covered listing first.
EXCHANGES = {
    # USD
    "NMS": ("", "USD"), "NGM": ("", "USD"), "NCM": ("", "USD"), "NYQ": ("", "USD"),
    "PCX": ("", "USD"), "ASE": ("", "USD"), "BTS": ("", "USD"),
    # EUR
    "AMS": (".AS", "EUR"), "GER": (".DE", "EUR"), "PAR": (".PA", "EUR"),
    "MIL": (".MI", "EUR"), "BRU": (".BR", "EUR"), "MCE": (".MC", "EUR"),
    "LIS": (".LS", "EUR"), "HEL": (".HE", "EUR"), "VIE": (".VI", "EUR"),
    "FRA": (".F", "EUR"), "STU": (".SG", "EUR"), "MUN": (".MU", "EUR"),
    "DUS": (".DU", "EUR"), "HAM": (".HM", "EUR"), "BER": (".BE", "EUR"),
    "ISE": (".IR", "EUR"),
    # Other
    "LSE": (".L", "GBP"), "EBS": (".SW", "CHF"), "TOR": (".TO", "CAD"),
    "CPH": (".CO", "DKK"), "STO": (".ST", "SEK"), "OSL": (".OL", "NOK"),
}
_RANK = {code: i for i, code in enumerate(EXCHANGES)}
# Guesses when Yahoo search finds nothing: suffixes to try per trading currency.
GUESS_SUFFIXES = {"USD": [""], "EUR": [".AS", ".DE", ".PA", ".MI", ".BR", ".MC"],
                  "GBP": [".L"], "GBX": [".L"], "CHF": [".SW"], "CAD": [".TO"]}
SECURITY_TYPES = {"EQUITY": "Stock", "ETF": "ETF", "MUTUALFUND": "ETF"}


@st.cache_data(ttl=7 * 24 * 3600, show_spinner=False)
def search_isin(isin: str) -> list[dict]:
    """Yahoo search hits for one ISIN: symbol, exchange, type, name. [] on failure.
    Cached for a week and shared by everyone (keyed on the ISIN only)."""
    import yfinance as yf

    try:
        quotes = yf.Search(isin, max_results=10, news_count=0, lists_count=0,
                           raise_errors=False).quotes or []
    except Exception as exc:  # network, rate limit, API change
        print(f"[lookup] {isin}: {type(exc).__name__}: {exc}")
        return []
    hits = []
    for q in quotes:
        sym = q.get("symbol")
        if not sym:
            continue
        hits.append({
            "symbol": sym,
            "exchange": q.get("exchange", ""),
            "type": q.get("quoteType", ""),
            "name": q.get("longname") or q.get("shortname") or "",
        })
    return hits


def rank_candidates(isin: str, t212_ticker: str, currency: str, hits: list[dict]) -> list[str]:
    """
    Yahoo symbols to try for one ISIN, best first.

    Preference: listings in the currency the user traded in (Trading 212's
    'Currency (Price / share)'), then symbols whose base equals the Trading 212
    ticker, then the exchange order in EXCHANGES. Guesses built from the
    Trading 212 ticker go last, so prices still work when search fails.
    """
    currency = (currency or "").upper().replace("GBX", "GBP")
    base = (t212_ticker or "").upper()

    def score(h: dict) -> tuple:
        _, cur = EXCHANGES.get(h["exchange"], ("", None))
        return (cur != currency, h["symbol"].split(".")[0].upper() != base,
                _RANK.get(h["exchange"], 99))

    ordered = [h["symbol"] for h in sorted(hits, key=score)]
    if base:
        for suffix in GUESS_SUFFIXES.get(currency, [""]):
            ordered.append(f"{base}{suffix}")
    seen, out = set(), []
    for s in ordered:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def security_type(hits: list[dict], name: str) -> str:
    """'Stock' or 'ETF', from Yahoo's quote type, else guessed from the name."""
    for h in hits:
        if h["type"] in SECURITY_TYPES:
            return SECURITY_TYPES[h["type"]]
    n = (name or "").upper()
    return "ETF" if any(w in n for w in ("ETF", "UCITS", "ISHARES", "VANGUARD", "(ACC)", "(DIST)")) else "Stock"


def _clean(x) -> float | None:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


@st.cache_data(ttl=24 * 3600, show_spinner="Fetching fundamentals…")
def fetch_fundamentals(symbols: tuple[str, ...]) -> pd.DataFrame:
    """
    Trailing P/E and P/B per Yahoo symbol (for stocks), plus sector.
    Missing values stay NaN. ETFs: see etf_valuation(). Beta: see betas().
    Cached for a day, keyed on the symbols only.
    """
    import yfinance as yf

    rows = []
    for sym in symbols:
        try:
            info = yf.Ticker(sym).info or {}
        except Exception as exc:
            print(f"[fundamentals] {sym}: {type(exc).__name__}: {exc}")
            info = {}
        rows.append({
            "symbol": sym,
            "pe": _clean(info.get("trailingPE")),
            "pb": _clean(info.get("priceToBook")),
            "sector": info.get("sector") or None,
        })
    return pd.DataFrame(rows, columns=["symbol", "pe", "pb", "sector"]).set_index("symbol")


# =============================================================================
# ETF valuation (P/E, P/B of the fund's holdings)
# =============================================================================
PROXIES_PATH = Path(__file__).resolve().parents[2] / "shared-config" / "etf_proxies.json"


@st.cache_data(ttl=3600)
def load_etf_proxies() -> list[dict]:
    try:
        return json.loads(PROXIES_PATH.read_text(encoding="utf-8")).get("rules", [])
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[proxies] could not read {PROXIES_PATH}: {exc}")
        return []


def etf_proxy(names: list[str], rules: list[dict]) -> dict | None:
    """First proxy rule whose 'match' words appear in any of the ETF's names."""
    text = " ".join(n for n in names if n).lower()
    for rule in rules:
        if any(m in text for m in rule.get("match", [])):
            return rule
    return None


def fund_ratios_from(pe, pb) -> tuple[float | None, float | None]:
    """
    Normalise Yahoo's fund ratios. Yahoo sometimes reports a fund's 'P/E' and
    'P/B' as their reciprocals (earnings yield 0.04 instead of P/E 25). No
    equity fund has a P/E below 1, so a P/E below 1 means both are inverted.
    Without a P/E, a P/B below 0.5 is taken as inverted too.
    """
    pe, pb = _clean(pe), _clean(pb)
    inverted = (pe is not None and 0 < pe < 1) or (pe is None and pb is not None and 0 < pb < 0.5)
    if inverted:
        pe = 1 / pe if pe else pe
        pb = 1 / pb if pb else pb
    return (pe if pe and pe > 0 else None), (pb if pb and pb > 0 else None)


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def fetch_fund_ratios(symbol: str) -> dict:
    """{'pe', 'pb'} of an ETF's holdings from Yahoo's fund data; None values if absent.
    Cached for a day, keyed on the symbol only."""
    import yfinance as yf

    pe = pb = None
    try:
        eq = yf.Ticker(symbol).funds_data.equity_holdings
        if eq is not None and not eq.empty:
            col = [c for c in eq.columns if c not in ("Average", "Category Average")]
            vals = eq.set_index("Average")[col[0]] if col else None
            if vals is not None:
                pe, pb = vals.get("Price/Earnings"), vals.get("Price/Book")
    except Exception as exc:  # no fund data for this symbol, network, API change
        print(f"[funds] {symbol}: {type(exc).__name__}: {exc}")
    pe, pb = fund_ratios_from(pe, pb)
    if pe is None and pb is None:  # some ETFs carry the ratios in .info instead
        try:
            info = yf.Ticker(symbol).info or {}
            pe, pb = fund_ratios_from(info.get("trailingPE"), info.get("priceToBook"))
        except Exception:
            pass
    return {"pe": pe, "pb": pb}


def etf_valuation(symbol: str | None, names: list[str], rules: list[dict]) -> dict:
    """P/E and P/B for one held ETF: its own Yahoo fund data, with anything
    missing filled from a US proxy tracking the same index. 'source' says
    which was used."""
    own = fetch_fund_ratios(symbol) if symbol else {"pe": None, "pb": None}
    if own["pe"] is not None and own["pb"] is not None:
        return own | {"source": f"{symbol} fund data"}
    rule = etf_proxy(names, rules)
    proxy = fetch_fund_ratios(rule["proxy"]) if rule else {"pe": None, "pb": None}
    out = {k: own[k] if own[k] is not None else proxy[k] for k in ("pe", "pb")}
    has_own = any(own[k] is not None for k in ("pe", "pb"))
    has_proxy = any(own[k] is None and proxy[k] is not None for k in ("pe", "pb"))
    if has_own and has_proxy:
        source = f"{symbol} fund data + {rule['proxy']}"
    elif has_own:
        source = f"{symbol} fund data"
    elif has_proxy:
        source = f"via {rule['proxy']} ({rule['index']})"
    elif rule:
        source = f"no data ({rule['proxy']} has none)"
    else:
        source = "no data, index not recognised"
    return out | {"source": source}


# =============================================================================
# Beta, measured from price history
# =============================================================================
MIN_WEEKS = 26


def betas(prices: pd.DataFrame, bench: pd.Series, weeks: int = 156) -> pd.DataFrame:
    """
    Beta and correlation of each column of `prices` against `bench`, from the
    last `weeks` Friday-to-Friday returns (both in EUR). Weekly returns avoid
    mismatches between markets that close at different times. Columns with
    fewer than MIN_WEEKS overlapping weeks get NaN.
    Returns a frame indexed like prices.columns with beta, corr, weeks.
    """
    b = bench.dropna()
    lv = prices.join(b.rename("__bench__"), how="outer").sort_index().ffill()
    w = lv.resample("W-FRI").last().pct_change(fill_method=None).iloc[1:].tail(weeks)
    rows = {}
    for c in prices.columns:
        pair = w[[c, "__bench__"]].dropna()
        pair = pair[(pair[c] != 0) | (pair["__bench__"] != 0)]
        var = pair["__bench__"].var()
        ok = len(pair) >= MIN_WEEKS and var and var > 0
        rows[c] = {
            "beta": pair[c].cov(pair["__bench__"]) / var if ok else float("nan"),
            "corr": pair[c].corr(pair["__bench__"]) if ok else float("nan"),
            "weeks": len(pair),
        }
    return pd.DataFrame.from_dict(rows, orient="index", columns=["beta", "corr", "weeks"])


def portfolio_ratios(values: pd.Series, fundamentals: pd.DataFrame) -> dict:
    """
    Value-weighted portfolio P/E, P/B and beta.

    values        market value per position (any index), aligned with
    fundamentals  rows for the same positions, columns pe, pb, beta.

    P/E and P/B are combined as weighted harmonic means (total value over total
    earnings / book value), which is how index providers do it; a plain average
    of P/Es overweights expensive stocks. Beta is a plain weighted average.
    Each ratio uses only the positions that have it; 'coverage' says what
    share of the portfolio's value that is.
    """
    out = {}
    total = float(values.sum())
    for col in ("pe", "pb", "beta"):
        x = fundamentals[col]
        ok = x.notna() & values.notna() & (values > 0)
        if col in ("pe", "pb"):
            ok &= x > 0
        v = values[ok]
        cov = float(v.sum()) / total if total else 0.0
        if v.empty:
            out[col], out[f"{col}_coverage"] = None, 0.0
            continue
        w = v / v.sum()
        if col == "beta":
            out[col] = float((w * x[ok]).sum())
        else:
            out[col] = float(1.0 / (w / x[ok]).sum())
        out[f"{col}_coverage"] = cov
    return out