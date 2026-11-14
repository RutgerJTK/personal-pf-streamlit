import streamlit as st

from auth import render_logout, require_login

st.set_page_config(page_title="Portfolio Tracker", page_icon="📈", layout="wide")

# Nothing below this line runs, and no data is fetched,
# until the session is authenticated.
require_login()
render_logout()

pages = [
    st.Page("views/overview.py", title="Overview", icon="🏠", default=True),
    st.Page("views/holdings.py", title="Holdings", icon="💼"),
    st.Page("views/benchmark.py", title="Benchmark", icon="🏁"),
    st.Page("views/pnl.py", title="Realized vs Unrealized", icon="💶"),
    st.Page("views/dividends.py", title="Dividends", icon="📅"),
    st.Page("views/watchlist.py", title="Watchlist", icon="👀"),
]

pg = st.navigation(pages)
pg.run()