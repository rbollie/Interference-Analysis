"""
auth.py — Authentication module for FAA RF Interference Analysis Tool
Handles login, session management, and admin user management.
Passwords are bcrypt-hashed. Credentials stored in Streamlit secrets.
"""

import streamlit as st
import bcrypt
import json
import time
from datetime import datetime

# ─── Helpers ─────────────────────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    """Return a bcrypt hash of the plaintext password."""
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt(rounds=12)).decode()

def check_password(plain: str, hashed: str) -> bool:
    """Verify a plaintext password against a bcrypt hash."""
    try:
        return bcrypt.checkpw(plain.encode(), hashed.encode())
    except Exception:
        return False

def load_users() -> dict:
    """
    Load user credentials from st.secrets.
    Expected secrets structure:
        [users.admin]
        password_hash = "$2b$12$..."
        role = "admin"
        name = "Administrator"

        [users.jsmith]
        password_hash = "$2b$12$..."
        role = "user"
        name = "John Smith"
    Returns dict: {username: {password_hash, role, name}}
    """
    try:
        raw = st.secrets.get("users", {})
        users = {}
        for username, data in raw.items():
            users[username] = {
                "password_hash": data.get("password_hash", ""),
                "role": data.get("role", "user"),
                "name": data.get("name", username),
            }
        return users
    except Exception:
        return {}

def is_authenticated() -> bool:
    return st.session_state.get("auth_ok", False)

def current_user() -> dict:
    return st.session_state.get("auth_user", {})

def is_admin() -> bool:
    return current_user().get("role") == "admin"

def logout():
    for key in ["auth_ok", "auth_user", "auth_username"]:
        st.session_state.pop(key, None)
    st.rerun()

# ─── Login page ──────────────────────────────────────────────────────────────

def show_login_page():
    """Render the login gate. Returns True if authenticated."""

    # Center the login card
    _, col, _ = st.columns([1, 1.4, 1])
    with col:
        st.markdown("""
<div style='text-align:center; padding: 30px 0 10px 0;'>
  <span style='font-size:3em;'>✈️</span>
  <h2 style='color:white; margin:8px 0 2px 0;'>FAA RF Interference Tool</h2>
  <p style='color:#aaa; font-size:0.85em; margin:0;'>ITU-R WP 5D / 5B Policy Support</p>
</div>
""", unsafe_allow_html=True)

        st.markdown("---")

        # Check if any users are configured
        users = load_users()

        if not users:
            st.warning("⚠️ No users configured yet.")
            st.markdown("""
**First-time setup:**

1. Go to your app on **share.streamlit.io**
2. Click **⋮ → Settings → Secrets**
3. Add the following (replace with a real bcrypt hash):

```toml
ANTHROPIC_API_KEY = "sk-ant-..."

[users.admin]
password_hash = "PASTE_HASH_HERE"
role = "admin"
name = "Administrator"
```

To generate a bcrypt hash for your password, use the **Hash Generator** below.
""")
            st.subheader("🔑 Password Hash Generator")
            st.caption("Generate a hash to paste into your Streamlit secrets:")
            pw_gen = st.text_input("Enter password to hash:", type="password", key="hashgen")
            if st.button("Generate Hash") and pw_gen:
                h = hash_password(pw_gen)
                st.code(h, language=None)
                st.caption("Copy this hash into your secrets TOML as `password_hash`.")
            return False

        # Login form
        with st.form("login_form", clear_on_submit=False):
            st.subheader("🔐 Sign In")
            username = st.text_input("Username").strip().lower()
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign In", use_container_width=True)

        if submitted:
            if not username or not password:
                st.error("Please enter both username and password.")
            elif username not in users:
                time.sleep(0.5)  # slow brute force
                st.error("Invalid username or password.")
            elif not check_password(password, users[username]["password_hash"]):
                time.sleep(0.5)
                st.error("Invalid username or password.")
            else:
                st.session_state["auth_ok"] = True
                st.session_state["auth_username"] = username
                st.session_state["auth_user"] = {
                    "username": username,
                    "name": users[username]["name"],
                    "role": users[username]["role"],
                }
                st.rerun()

        st.markdown("""
<div style='text-align:center; margin-top:20px;'>
  <p style='color:#555; font-size:0.75em;'>
  Contact your administrator to request access.
  </p>
</div>
""", unsafe_allow_html=True)

    return False

# ─── Admin panel ─────────────────────────────────────────────────────────────

def show_admin_panel():
    """Full admin panel for user management."""
    st.title("⚙️ Admin Panel — User Management")
    st.markdown(f"*Logged in as **{current_user()['name']}** (admin)*")
    st.caption(
        "Streamlit Cloud cannot write files at runtime. Changes made here generate "
        "updated secrets TOML — copy and paste it into your app's Secrets settings to apply."
    )

    users = load_users()

    # ── Current users ─────────────────────────────────────────────────────────
    st.subheader("👥 Current Users")
    if users:
        rows = []
        for uname, u in users.items():
            rows.append({
                "Username": uname,
                "Name": u["name"],
                "Role": u["role"],
                "Password Hash": u["password_hash"][:20] + "…",
            })
        import pandas as pd
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("No users configured.")

    st.markdown("---")

    # ── Add new user ──────────────────────────────────────────────────────────
    st.subheader("➕ Add New User")
    with st.form("add_user_form"):
        col1, col2 = st.columns(2)
        with col1:
            new_username = st.text_input("Username (lowercase, no spaces)",
                placeholder="e.g., jsmith")
            new_name = st.text_input("Full Name", placeholder="e.g., John Smith")
        with col2:
            new_role = st.selectbox("Role", ["user", "admin"])
            new_password = st.text_input("Temporary Password", type="password",
                help="User should change this on first login — tell them separately")
            new_password2 = st.text_input("Confirm Password", type="password")

        add_submitted = st.form_submit_button("➕ Add User", type="primary")

    if add_submitted:
        new_username = new_username.strip().lower()
        if not new_username or not new_name or not new_password:
            st.error("All fields required.")
        elif new_password != new_password2:
            st.error("Passwords do not match.")
        elif new_username in users:
            st.error(f"Username '{new_username}' already exists.")
        elif " " in new_username or not new_username.replace("_","").replace("-","").isalnum():
            st.error("Username must be lowercase alphanumeric (hyphens/underscores OK).")
        else:
            users[new_username] = {
                "password_hash": hash_password(new_password),
                "role": new_role,
                "name": new_name,
            }
            st.success(f"User '{new_username}' added. Copy the secrets below into Streamlit Cloud.")
            _show_secrets_toml(users)

    st.markdown("---")

    # ── Remove user ───────────────────────────────────────────────────────────
    st.subheader("🗑️ Remove User")
    removable = [u for u in users if u != current_user()["username"]]
    if removable:
        with st.form("remove_user_form"):
            remove_target = st.selectbox("Select user to remove:", removable)
            remove_submitted = st.form_submit_button("🗑️ Remove User",
                type="primary")
        if remove_submitted:
            del users[remove_target]
            st.success(f"User '{remove_target}' removed. Copy the secrets below.")
            _show_secrets_toml(users)
    else:
        st.info("No other users to remove.")

    st.markdown("---")

    # ── Reset password ────────────────────────────────────────────────────────
    st.subheader("🔑 Reset a User's Password")
    with st.form("reset_pw_form"):
        col1, col2 = st.columns(2)
        with col1:
            reset_target = st.selectbox("User:", list(users.keys()))
            reset_pw = st.text_input("New Password", type="password")
        with col2:
            reset_pw2 = st.text_input("Confirm New Password", type="password")
        reset_submitted = st.form_submit_button("🔄 Reset Password")

    if reset_submitted:
        if not reset_pw:
            st.error("Enter a new password.")
        elif reset_pw != reset_pw2:
            st.error("Passwords do not match.")
        else:
            users[reset_target]["password_hash"] = hash_password(reset_pw)
            st.success(f"Password for '{reset_target}' reset. Copy the secrets below.")
            _show_secrets_toml(users)

    st.markdown("---")

    # ── Change own password ───────────────────────────────────────────────────
    st.subheader("🔐 Change Your Own Password")
    with st.form("change_own_pw"):
        own_current = st.text_input("Current Password", type="password")
        own_new = st.text_input("New Password", type="password")
        own_new2 = st.text_input("Confirm New Password", type="password")
        own_submitted = st.form_submit_button("Update My Password")

    if own_submitted:
        me = current_user()["username"]
        if not check_password(own_current, users[me]["password_hash"]):
            st.error("Current password incorrect.")
        elif own_new != own_new2:
            st.error("New passwords do not match.")
        elif len(own_new) < 8:
            st.error("Password must be at least 8 characters.")
        else:
            users[me]["password_hash"] = hash_password(own_new)
            st.success("Password updated. Copy the secrets TOML below and apply in Streamlit Cloud.")
            _show_secrets_toml(users)

    st.markdown("---")

    # ── Always show current full secrets ─────────────────────────────────────
    with st.expander("📋 View / Copy Full Current Secrets TOML"):
        st.caption("This is the complete secrets file for your current user list.")
        _show_secrets_toml(users, label="Current")

    # ── Audit log hint ────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📖 How to Apply Changes")
    st.markdown("""
1. Copy the **Secrets TOML** block generated above
2. Go to **share.streamlit.io** → your app → **⋮ → Settings → Secrets**
3. **Replace** the existing `[users.*]` sections with the new content
4. Keep your `ANTHROPIC_API_KEY` line — do not overwrite it
5. Click **Save** — the app restarts and new credentials are active within ~30 seconds
""")


def _show_secrets_toml(users: dict, label: str = "Updated"):
    """Render a copyable secrets TOML block for all users."""
    lines = [f"# {label} secrets — paste into Streamlit Cloud Settings → Secrets"]
    lines.append('# Keep your ANTHROPIC_API_KEY line above these sections\n')
    for uname, u in users.items():
        lines.append(f"[users.{uname}]")
        lines.append(f'password_hash = "{u["password_hash"]}"')
        lines.append(f'role = "{u["role"]}"')
        lines.append(f'name = "{u["name"]}"')
        lines.append("")
    toml_str = "\n".join(lines)
    st.code(toml_str, language="toml")
    st.caption("⬆️ Copy this entire block and paste it into Streamlit Cloud Secrets.")
