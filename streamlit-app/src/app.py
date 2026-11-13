import streamlit as st

st.set_page_config(page_title="Portfolio Tracker", page_icon="📈", layout="wide")

# ---- AUTH GUARD GOES HERE (next step) ----
# Nothing below this line may run, and no data may be fetched,
# until the guard has confirmed the session is authenticated.

pages = [
    st.Page("views/overview.py", title="Overview", icon="🏠", default=True),
    st.Page("views/benchmark.py", title="Benchmark", icon="🏁"),
    st.Page("views/pnl.py", title="Realized vs Unrealized", icon="💶"),
    st.Page("views/dividends.py", title="Dividends", icon="📅"),
    st.Page("views/watchlist.py", title="Watchlist", icon="👀"),
]

pg = st.navigation(pages)
pg.run()