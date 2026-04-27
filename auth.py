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
    """Render the branded FAA login page matching the design spec."""

    # ── Full-page dark background ─────────────────────────────────────────────
    st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #020d1f 0%, #041530 40%, #071e3d 100%);
    min-height: 100vh;
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { display: none; }
footer { visibility: hidden; }
div[data-testid="column"] { padding: 0 !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* Left side text */
.faa-brand   { display:flex; align-items:center; gap:12px; margin-bottom:36px; }
.faa-title   { font-size:3.4em; font-weight:900; color:#fff; line-height:1.05; margin:0 0 10px 0; }
.faa-sub     { font-size:1.15em; font-weight:600; color:#3b9eff; margin:0 0 14px 0; }
.faa-desc    { color:#a8bdd0; font-size:0.93em; line-height:1.6; max-width:420px; }
.freq-label  { color:#3b9eff; font-size:0.78em; font-family:monospace; position:relative; }
.feature-row { display:flex; gap:24px; margin-top:44px; }
.feature-box { background:rgba(59,158,255,0.08); border:1px solid rgba(59,158,255,0.2);
               border-radius:10px; padding:18px 16px; flex:1; }
.feature-box svg { margin-bottom:10px; }
.feature-box h4  { color:#fff; font-size:0.95em; margin:0 0 6px 0; }
.feature-box p   { color:#7fa8c8; font-size:0.78em; margin:0; line-height:1.5; }

/* Login card */
.login-card {
    background: rgba(12, 28, 55, 0.85);
    border: 1px solid rgba(59,158,255,0.25);
    border-radius: 16px;
    padding: 40px 38px;
    backdrop-filter: blur(18px);
    box-shadow: 0 24px 64px rgba(0,0,0,0.55);
    max-width: 420px;
    margin: 0 auto;
}
.secure-badge {
    display:flex; align-items:center; justify-content:center;
    gap:8px; margin-bottom:22px;
    color:#3b9eff; font-size:0.75em; font-weight:700; letter-spacing:0.12em;
}
.secure-badge svg { opacity:0.85; }
.card-title   { color:#fff; font-size:1.6em; font-weight:800; text-align:center; margin:0 0 4px 0; }
.card-sub     { color:#7fa8c8; font-size:0.85em; text-align:center; margin:0 0 28px 0; }
.field-label  { color:#a8bdd0; font-size:0.8em; font-weight:600; margin:0 0 7px 0; letter-spacing:0.04em; }
.divider-or   { display:flex; align-items:center; gap:12px; margin:16px 0;
                color:#3a5070; font-size:0.8em; }
.divider-or::before, .divider-or::after {
    content:''; flex:1; height:1px; background:rgba(59,158,255,0.15); }
.access-btn   { background:rgba(20,45,80,0.7); border:1px solid rgba(59,158,255,0.2);
                border-radius:8px; padding:14px 18px; cursor:pointer; display:flex;
                align-items:center; gap:12px; color:#a8bdd0; font-size:0.85em; width:100%; }
.contact-link { color:#3b9eff; font-size:0.82em; text-align:center; margin-top:14px;
                display:flex; align-items:center; justify-content:center; gap:6px; }

/* Streamlit form overrides for dark theme */
div[data-testid="stForm"] { background: transparent !important; border:none !important; padding:0 !important; }
div.stTextInput > div > div { background: rgba(6,22,45,0.8) !important; border: 1px solid rgba(59,158,255,0.3) !important; border-radius:8px !important; }
div.stTextInput input { color:#fff !important; }
div.stTextInput input::placeholder { color:#3a5070 !important; }
div.stCheckbox label { color:#a8bdd0 !important; font-size:0.85em !important; }
div.stFormSubmitButton button[type=submit] {
    background: linear-gradient(90deg,#1a6fd4,#1e90ff) !important;
    color: white !important; font-weight:700 !important; font-size:1em !important;
    border-radius:9px !important; border:none !important; padding:14px !important;
    letter-spacing:0.02em !important; box-shadow:0 4px 20px rgba(30,144,255,0.35) !important;
}
div.stFormSubmitButton button:hover { opacity:0.9 !important; }

/* Footer */
.page-footer {
    display:flex; align-items:center; justify-content:space-between;
    border-top:1px solid rgba(59,158,255,0.12);
    padding:14px 48px; margin-top:32px;
    color:#3a5070; font-size:0.72em;
}
.page-footer .secure-tagline { display:flex; align-items:center; gap:8px; }

/* Orbital rings / decorative */
.orbital-wrap { position:relative; height:260px; margin:32px 0 0 0; overflow:visible; }
.orbital-svg  { position:absolute; left:-60px; top:-30px; opacity:0.5; }
</style>
""", unsafe_allow_html=True)

    # ── Page layout: left hero + right login ──────────────────────────────────
    hero_col, gap_col, card_col = st.columns([1.15, 0.08, 0.9])

    with hero_col:
        st.markdown("""
<div style="padding: 48px 12px 24px 48px;">

  <!-- FAA Logo + Name -->
  <div class="faa-brand">
    <svg width="48" height="48" viewBox="0 0 48 48" fill="none">
      <circle cx="24" cy="24" r="22" fill="#0a2044" stroke="#3b9eff" stroke-width="2"/>
      <path d="M12 28L24 12L36 28H28L24 36L20 28H12Z" fill="#3b9eff" opacity="0.9"/>
      <circle cx="24" cy="24" r="4" fill="#1e90ff"/>
    </svg>
    <div>
      <div style="color:#fff;font-weight:800;font-size:1em;line-height:1.1;">FAA</div>
      <div style="color:#7fa8c8;font-size:0.65em;line-height:1.2;">Federal Aviation<br>Administration</div>
    </div>
  </div>

  <!-- Hero headline -->
  <h1 class="faa-title">FAA RF<br>Interference Tool</h1>
  <p class="faa-sub">ITU-R Working Party Policy Support</p>
  <p class="faa-desc">Analyze and assess RF interference risks to support global<br>aviation spectrum protection and policy decisions.</p>

  <!-- Orbital decoration with frequency annotations -->
  <div class="orbital-wrap">
    <svg class="orbital-svg" width="480" height="280" viewBox="0 0 480 280" fill="none">
      <!-- Earth arc -->
      <ellipse cx="200" cy="360" rx="320" ry="220" stroke="#1a3a6a" stroke-width="1.5" fill="none" opacity="0.6"/>
      <!-- Orbit rings -->
      <ellipse cx="200" cy="140" rx="260" ry="90" stroke="#1e90ff" stroke-width="0.8" fill="none" opacity="0.3" transform="rotate(-15 200 140)"/>
      <ellipse cx="200" cy="140" rx="200" ry="65" stroke="#3b9eff" stroke-width="0.6" fill="none" opacity="0.25" transform="rotate(-8 200 140)"/>
      <!-- Satellite dot -->
      <circle cx="410" cy="68" r="4" fill="#3b9eff" opacity="0.85"/>
      <circle cx="410" cy="68" r="8" fill="#3b9eff" opacity="0.2"/>
      <!-- Aircraft silhouette (simple) -->
      <path d="M170 155 L200 148 L230 155 L222 160 L200 157 L178 160Z" fill="#7fc8ff" opacity="0.7"/>
      <path d="M192 148 L200 130 L208 148Z" fill="#7fc8ff" opacity="0.5"/>
      <!-- Grid lines (spectrum) -->
      <line x1="40" y1="200" x2="380" y2="200" stroke="#1a3a6a" stroke-width="0.5" opacity="0.6"/>
      <line x1="40" y1="220" x2="380" y2="220" stroke="#1a3a6a" stroke-width="0.5" opacity="0.4"/>
      <!-- Freq labels -->
      <text x="42" y="195" fill="#3b9eff" font-size="9" font-family="monospace" opacity="0.8">118.0 MHz</text>
      <text x="300" y="168" fill="#3b9eff" font-size="9" font-family="monospace" opacity="0.8">–PROTECTED</text>
      <text x="314" y="178" fill="#3b9eff" font-size="9" font-family="monospace" opacity="0.8">BANDS</text>
      <text x="260" y="230" fill="#3b9eff" font-size="9" font-family="monospace" opacity="0.8">5,091–5,091.5 MHz</text>
      <!-- Glow lines -->
      <line x1="200" y1="157" x2="200" y2="210" stroke="#1e90ff" stroke-width="0.5" opacity="0.4" stroke-dasharray="3 3"/>
      <circle cx="200" cy="210" r="20" fill="#1e90ff" opacity="0.06"/>
    </svg>
  </div>

  <!-- Feature cards row -->
  <div class="feature-row">
    <div class="feature-box">
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
        <rect x="2" y="14" width="3" height="6" rx="1" fill="#3b9eff"/>
        <rect x="7" y="10" width="3" height="10" rx="1" fill="#3b9eff" opacity="0.8"/>
        <rect x="12" y="6" width="3" height="14" rx="1" fill="#3b9eff" opacity="0.6"/>
        <rect x="17" y="8" width="3" height="12" rx="1" fill="#3b9eff" opacity="0.8"/>
      </svg>
      <h4>Signal Analysis</h4>
      <p>Detect and analyze RF emissions with precision and clarity.</p>
    </div>
    <div class="feature-box">
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
        <path d="M12 2L4 6v6c0 5.5 3.5 10.7 8 12 4.5-1.3 8-6.5 8-12V6L12 2z" stroke="#3b9eff" stroke-width="1.5" fill="none"/>
        <path d="M9 12l2 2 4-4" stroke="#3b9eff" stroke-width="1.5" stroke-linecap="round"/>
      </svg>
      <h4>Protected Band Assessment</h4>
      <p>Evaluate potential interference to aviation services and protected bands.</p>
    </div>
    <div class="feature-box">
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
        <rect x="3" y="3" width="18" height="18" rx="2" stroke="#3b9eff" stroke-width="1.5" fill="none"/>
        <path d="M7 17l3-5 3 3 3-6" stroke="#3b9eff" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
      <h4>Policy Support</h4>
      <p>Generate insights and reports to inform ITU-R working party policy decisions.</p>
    </div>
  </div>

</div>
""", unsafe_allow_html=True)

    with card_col:
        st.markdown("""
<div style="padding: 48px 48px 24px 12px;">
<div class="login-card">

  <div class="secure-badge">
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
      <path d="M8 1L3 3.5v4C3 10.5 5.2 13.2 8 14c2.8-.8 5-3.5 5-6.5v-4L8 1z"
            stroke="#3b9eff" stroke-width="1.3" fill="rgba(59,158,255,0.1)"/>
      <path d="M6 8l1.5 1.5L10 6.5" stroke="#3b9eff" stroke-width="1.2" stroke-linecap="round"/>
    </svg>
    SECURE ACCESS
  </div>

  <h2 class="card-title">Welcome back</h2>
  <p class="card-sub">Sign in to continue to the tool</p>

</div>
</div>
""", unsafe_allow_html=True)

        # ── Form (must be Streamlit widgets — can't be pure HTML) ────────────
        users = load_users()

        if not users:
            # First-run setup
            st.markdown("""
<div style="padding:0 48px; max-width:420px; margin:0 auto;">
<div style="background:rgba(255,200,50,0.08);border:1px solid rgba(255,200,50,0.25);
     border-radius:10px;padding:16px 18px;color:#f0c050;font-size:0.85em;">
<b>⚠️ First-time setup</b><br>No users configured yet — add credentials in Streamlit Secrets.
</div></div>""", unsafe_allow_html=True)
            with st.expander("🔑 Setup instructions & hash generator"):
                st.markdown("""
**Add to Streamlit Cloud → Settings → Secrets:**
```toml
ANTHROPIC_API_KEY = "sk-ant-..."
[users.admin]
password_hash = "PASTE_HASH_HERE"
role = "admin"
name = "Administrator"
```""")
                pw_gen = st.text_input("Password to hash:", type="password", key="hashgen")
                if st.button("Generate Hash") and pw_gen:
                    st.code(hash_password(pw_gen), language=None)
            return False

        with st.container():
            st.markdown('<div style="padding:0 48px; max-width:420px; margin:-20px auto 0 auto;">', unsafe_allow_html=True)

            with st.form("login_form", clear_on_submit=False):
                st.markdown('<p class="field-label">Username</p>', unsafe_allow_html=True)
                username = st.text_input("", placeholder="Enter your username",
                    label_visibility="collapsed").strip().lower()

                st.markdown('<p class="field-label" style="margin-top:14px;">Password</p>',
                    unsafe_allow_html=True)
                password = st.text_input("", placeholder="Enter your password",
                    type="password", label_visibility="collapsed")

                rem_col, fp_col = st.columns([1, 1])
                with rem_col:
                    st.checkbox("Remember me", value=True)
                with fp_col:
                    st.markdown(
                        '<div style="text-align:right;padding-top:4px;">'
                        '<span style="color:#3b9eff;font-size:0.82em;cursor:pointer;">'
                        'Forgot password?</span></div>', unsafe_allow_html=True)

                submitted = st.form_submit_button("Sign In  →",
                    use_container_width=True, type="primary")

            if submitted:
                if not username or not password:
                    st.error("Please enter both username and password.")
                elif username not in users:
                    time.sleep(0.5)
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

            st.markdown('<div class="divider-or">or</div>', unsafe_allow_html=True)

            st.markdown("""
<div class="access-btn">
  <svg width="22" height="22" viewBox="0 0 22 22" fill="none">
    <circle cx="11" cy="7" r="4" stroke="#7fa8c8" stroke-width="1.4"/>
    <path d="M3 19c0-4 3.6-7 8-7s8 3 8 7" stroke="#7fa8c8" stroke-width="1.4" stroke-linecap="round"/>
    <circle cx="17" cy="5" r="3" fill="#0a2044" stroke="#3b9eff" stroke-width="1.2"/>
    <path d="M16 5h2M17 4v2" stroke="#3b9eff" stroke-width="1" stroke-linecap="round"/>
  </svg>
  <div>
    <div style="color:#c8daea;font-weight:600;font-size:0.88em;">Need access?</div>
    <div style="color:#5a7a99;font-size:0.77em;">Request an account</div>
  </div>
</div>

<div class="contact-link" style="margin-top:14px;">
  <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
    <rect x="1" y="3" width="12" height="9" rx="1.5" stroke="#3b9eff" stroke-width="1.2"/>
    <path d="M1 4.5l6 4 6-4" stroke="#3b9eff" stroke-width="1.2" stroke-linecap="round"/>
  </svg>
  <span style="color:#3b9eff;">Contact administrator</span>
</div>
""", unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("""
<div class="page-footer">
  <div class="secure-tagline">
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
      <path d="M7 1L2 3.5v3.5C2 9.8 4.2 12.2 7 13c2.8-.8 5-3.2 5-6V3.5L7 1z"
            stroke="#3b9eff" stroke-width="1.2" fill="none"/>
    </svg>
    <span style="color:#4a7a9b;">Secure &nbsp;•&nbsp; Compliant &nbsp;•&nbsp; Reliable</span>
  </div>
  <span style="color:#2a4a6a;">Supporting safe and efficient access to the radio spectrum for global aviation.</span>
  <span style="color:#2a4a6a;">© 2024 Federal Aviation Administration</span>
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
