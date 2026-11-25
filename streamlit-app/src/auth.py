"""
Portfolio selection and login.

The app has two modes, kept in ``st.session_state.mode``:

    "personal"  Rutger's own portfolio. Password-gated (scrypt hash in
                secrets under [app_auth]). Gets all the S3-backed pages.
    "generic"   Anyone's portfolio, from files they upload in the browser.
                No password, and no access to the personal pages or S3.

Until a mode is chosen (and, for personal, the password is accepted),
require_mode() shows the selection screen and halts the script.
"""
import hashlib
import hmac
import time

import streamlit as st

_MAX_ATTEMPTS = 5
_LOCKOUT_SECONDS = 60

# Kept when going back to the selection screen, so the lockout can't be
# reset by bouncing between screens.
_KEEP_ON_RESET = ("failed_attempts", "locked_until")


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


def _stored_hash() -> str | None:
    try:
        return st.secrets["app_auth"]["password_hash"]
    except (KeyError, FileNotFoundError) as exc:
        # Detail goes to the server log only, never to the page
        print(f"[auth] secrets problem: {type(exc).__name__}: {exc}")
        return None


def _toggle_password_form() -> None:
    st.session_state.show_password = not st.session_state.get("show_password", False)


def _choose_generic() -> None:
    st.session_state.mode = "generic"
    st.session_state.pop("show_password", None)


def _password_form() -> None:
    """Password form for the personal portfolio. Sets mode on success."""
    stored = _stored_hash()
    if stored is None:
        st.error("The personal portfolio is not configured on this server.")
        return

    with st.form("login_form"):
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Unlock")

    if not submitted:
        return

    locked_for = int(st.session_state.get("locked_until", 0) - time.time())
    if locked_for > 0:
        st.error(f"Too many attempts. Try again in {locked_for} seconds.")
    elif _verify(password, stored):
        st.session_state.mode = "personal"
        st.session_state.authenticated = True
        st.session_state.pop("failed_attempts", None)
        st.session_state.pop("locked_until", None)
        st.session_state.pop("show_password", None)
        st.rerun()
    else:
        attempts = st.session_state.get("failed_attempts", 0) + 1
        st.session_state.failed_attempts = attempts
        time.sleep(1)
        if attempts >= _MAX_ATTEMPTS:
            st.session_state.locked_until = time.time() + _LOCKOUT_SECONDS
            st.session_state.failed_attempts = 0
        st.error("Incorrect password.")


def require_mode() -> str:
    """Return "personal" or "generic". Until one is chosen (and, for personal,
    unlocked), show the selection screen and halt. Fails closed."""
    mode = st.session_state.get("mode")
    if mode == "generic":
        return mode
    if mode == "personal" and st.session_state.get("authenticated"):
        return mode

    # Register the selection screen as the only page, with the menu hidden.
    # Without this the browser keeps showing the previous mode's page menu.
    st.navigation(
        [st.Page(_selection_screen, title="Portfolio Tracker", icon="📈", default=True)],
        position="hidden",
    ).run()
    st.stop()


def _selection_screen() -> None:
    st.title("📈 Portfolio Tracker")
    st.write("Which portfolio do you want to open?")

    st.button(
        "🔐 Personal portfolio",
        on_click=_toggle_password_form,
        width="stretch",
    )
    if st.session_state.get("show_password"):
        _password_form()

    st.button(
        "📂 Generic portfolio",
        on_click=_choose_generic,
        help="Upload your own broker export. Nothing is stored.",
        width="stretch",
    )


def render_back_button(home: st.Page) -> None:
    """Sidebar button back to the selection screen. Clears the session
    (logs out of personal, drops any uploaded data) except the lockout.

    ``home`` is the current mode's default page. Switching to it first puts
    the URL back on the root, so the other mode doesn't open on a page path
    it doesn't have.
    """
    with st.sidebar:
        if st.button("← Back to portfolio selection", width="stretch"):
            kept = {k: st.session_state[k] for k in _KEEP_ON_RESET if k in st.session_state}
            st.session_state.clear()
            st.session_state.update(kept)
            st.switch_page(home)