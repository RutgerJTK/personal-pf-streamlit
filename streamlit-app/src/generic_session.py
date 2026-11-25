"""
State shared by the generic pages (Dashboard, Benchmark), per browser session.

The uploaded files live in st.session_state under a plain key, so they survive
switching pages (a file_uploader forgets its files when its page isn't shown).
Nothing is written to disk or to the shared Streamlit cache; going back to the
selection screen clears the session and with it the files.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

from security_lookup import first_with_prices, rank_candidates, search_isin, security_type
from t212_parser import ParseError, ParseResult, parse_trading212

FILES_KEY = "generic_files"            # list[(name, bytes)]
OVERRIDES_KEY = "ticker_overrides"     # isin -> Yahoo symbol typed by the user
SAMPLE_FILE = Path(__file__).resolve().parents[1] / "sample_data" / "trading212_sample.csv"
MAX_CANDIDATES = 4                     # Yahoo symbols checked per ISIN, in order, until one has prices
DASHBOARD_PAGE = "views/generic/dashboard.py"


def set_files(files: list[tuple[str, bytes]], sample: bool = False) -> None:
    st.session_state[FILES_KEY] = files
    st.session_state["generic_is_sample"] = sample


def load_sample() -> None:
    set_files([(SAMPLE_FILE.name, SAMPLE_FILE.read_bytes())], sample=True)


def files() -> list[tuple[str, bytes]]:
    return st.session_state.get(FILES_KEY) or []


def is_sample() -> bool:
    return bool(st.session_state.get("generic_is_sample"))


def clear() -> None:
    for k in (FILES_KEY, "generic_is_sample", OVERRIDES_KEY):
        st.session_state.pop(k, None)


def overrides() -> dict:
    return st.session_state.setdefault(OVERRIDES_KEY, {})


@dataclass
class Resolved:
    candidates: dict[str, tuple[str, ...]]   # isin -> (the Yahoo symbol to use,) or () if none found
    meta: dict[str, dict]                    # isin -> name, short, group, kind, currency
    hits: dict[str, list[dict]]              # isin -> raw Yahoo search hits


def resolve(parsed: ParseResult, isins) -> Resolved:
    """Yahoo symbols and display metadata for the given ISINs (user overrides win)."""
    over = overrides()
    sec = parsed.securities
    cands, meta, hits = {}, {}, {}
    with st.spinner("Matching your securities to price sources…"):
        for isin in isins:
            s = sec.loc[isin] if isin in sec.index else pd.Series({"ticker": "", "name": "", "currency": ""})
            hits[isin] = search_isin(isin)
            if over.get(isin):  # typed by the user: used as is, so a typo shows up as "not found"
                cands[isin] = (over[isin],)
            else:
                ranked = rank_candidates(isin, s["ticker"], s["currency"], hits[isin])[:MAX_CANDIDATES]
                found = first_with_prices(ranked)
                cands[isin] = (found,) if found else ()
            kind = security_type(hits[isin], s["name"])
            yahoo_name = next((h["name"] for h in hits[isin] if h["name"]), "")
            meta[isin] = {"name": s["name"] or yahoo_name or isin, "short": s["ticker"] or isin,
                          "group": kind, "kind": kind, "currency": s["currency"],
                          "yahoo_name": yahoo_name}
    return Resolved(cands, meta, hits)


def parsed_or_stop(show_link: bool = True) -> ParseResult:
    """The parsed upload, or a short message (and a link to the Dashboard) and stop."""
    if not files():
        st.info("Upload your Trading 212 export on the Dashboard first.")
        if show_link:
            st.page_link(DASHBOARD_PAGE, label="Go to the Dashboard", icon="📊")
        st.stop()
    try:
        return parse_trading212(files())
    except ParseError as exc:
        st.error(str(exc))
        st.stop()