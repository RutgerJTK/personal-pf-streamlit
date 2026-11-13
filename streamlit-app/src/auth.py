import hashlib
import hmac
import time

import streamlit as st

_MAX_ATTEMPTS = 5
_LOCKOUT_SECONDS = 60


def _verify(password: str, stored: str) -> bool:
    """Check a password against a 'scrypt:<salt_hex>:<hash_hex>' string."""
    try:
        scheme, salt_hex, hash_hex = stored.split(":")
        if scheme != "scrypt":
            return False
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
    except ValueError:
        return False

    candidate = hashlib.scrypt(
        password.encode(), salt=salt, n=2**14, r=8, p=1, dklen=len(expected)
    )
    return hmac.compare_digest(candidate, expected)


def require_login() -> None:
    """Halt the script unless this session is authenticated. Fails closed."""
    if st.session_state.get("authenticated"):
        return

    try:
        stored = st.secrets["app_auth"]["password_hash"]
    except (KeyError, FileNotFoundError) as exc:
        # Detail goes to the server log only, never to the page
        print(f"[auth] secrets problem: {type(exc).__name__}: {exc}")
        st.error("Authentication is not configured. The app will not start.")
        st.stop()

    st.title("🔐 Portfolio Tracker")

    with st.form("login_form"):
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Unlock")

    if submitted:
        locked_for = int(st.session_state.get("locked_until", 0) - time.time())
        if locked_for > 0:
            st.error(f"Too many attempts. Try again in {locked_for} seconds.")
        elif _verify(password, stored):
            st.session_state.authenticated = True
            st.session_state.pop("failed_attempts", None)
            st.session_state.pop("locked_until", None)
            st.rerun()
        else:
            attempts = st.session_state.get("failed_attempts", 0) + 1
            st.session_state.failed_attempts = attempts
            time.sleep(1)
            if attempts >= _MAX_ATTEMPTS:
                st.session_state.locked_until = time.time() + _LOCKOUT_SECONDS
                st.session_state.failed_attempts = 0
            st.error("Incorrect password.")

    st.stop()


def render_logout() -> None:
    """Sidebar logout button. Call only after require_login()."""
    with st.sidebar:
        if st.button("🔒 Log out", use_container_width=True):
            st.session_state.clear()
            st.rerun()