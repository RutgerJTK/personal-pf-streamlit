"""
Generic dashboard: the landing page for anyone's portfolio.

Rules for everything under views/generic/:
  * never import s3_fetcher or read st.secrets["aws"];
  * user data lives in st.session_state only (per browser session), never on disk.
"""
import streamlit as st

st.title("📊 Portfolio Dashboard")