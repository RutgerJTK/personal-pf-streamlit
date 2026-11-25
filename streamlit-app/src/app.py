import streamlit as st

from auth import render_back_button, require_mode

st.set_page_config(page_title="Portfolio Tracker", page_icon="📈", layout="wide")

# Nothing below this line runs until a mode is chosen, and for "personal"
# until the password is accepted. In "generic" mode the personal pages are
# never registered, so they can't be opened by URL and no S3 data is fetched.
mode = require_mode()

if mode == "personal":
    pages = [
        st.Page("views/overview.py", title="Overview", icon="🏠", default=True),
        st.Page("views/holdings.py", title="Holdings", icon="💼"),
        st.Page("views/benchmark.py", title="Benchmark", icon="🏁"),
        st.Page("views/pnl.py", title="Realized vs Unrealized", icon="💶"),
        st.Page("views/dividends.py", title="Dividends", icon="📅"),
        st.Page("views/watchlist.py", title="Watchlist", icon="👀"),
    ]
else:
    pages = [
        st.Page("views/generic/dashboard.py", title="Dashboard", icon="📊", default=True),
        st.Page("views/generic/benchmark.py", title="Benchmark", icon="🏁", url_path="generic-benchmark"),
    ]

pg = st.navigation(pages)
render_back_button(home=pages[0])
pg.run()