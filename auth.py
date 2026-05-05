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
    """Render the animated metallic silver & black FAA login page."""

    st.markdown("""
<style>
/* ── Base reset ─────────────────────────────────────────────────── */
[data-testid="stAppViewContainer"] {
    background: #000 !important;
    min-height: 100vh;
    overflow: hidden;
}
[data-testid="stHeader"], [data-testid="stToolbar"] { display: none !important; }
footer { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* Ensure Streamlit widgets are always above all decorative layers */
[data-testid="stVerticalBlock"],
[data-testid="column"],
div[data-testid="stForm"],
div.stTextInput,
div.stCheckbox,
div.stFormSubmitButton,
div.stButton,
.stExpander { position: relative; z-index: 100 !important; }

/* ── Animated metallic background ──────────────────────────────── */
.faa-bg {
    position: fixed; inset: 0; z-index: 0;
    pointer-events: none;
    background:
        radial-gradient(ellipse at 20% 50%, rgba(180,180,180,0.08) 0%, transparent 60%),
        radial-gradient(ellipse at 80% 20%, rgba(220,220,220,0.06) 0%, transparent 55%),
        linear-gradient(160deg, #0a0a0a 0%, #141414 30%, #0d0d0d 60%, #111 100%);
    animation: bgShimmer 8s ease-in-out infinite alternate;
}
@keyframes bgShimmer {
    0%   { background-position: 0% 50%; }
    100% { background-position: 100% 50%; }
}

/* Animated particle grid */
.faa-grid {
    position: fixed; inset: 0; z-index: 1;
    pointer-events: none;
    background-image:
        linear-gradient(rgba(180,180,180,0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(180,180,180,0.04) 1px, transparent 1px);
    background-size: 60px 60px;
    animation: gridMove 20s linear infinite;
}
@keyframes gridMove {
    0%   { background-position: 0 0, 0 0; }
    100% { background-position: 60px 60px, 60px 60px; }
}

/* Sweeping metallic light bar */
.faa-sweep {
    position: fixed;
    top: 0; left: -200%;
    width: 60%; height: 100%;
    background: linear-gradient(
        105deg,
        transparent 0%,
        rgba(200,200,200,0.015) 40%,
        rgba(220,220,220,0.05) 50%,
        rgba(200,200,200,0.015) 60%,
        transparent 100%
    );
    animation: sweep 6s ease-in-out infinite;
    z-index: 2; pointer-events: none;
}
@keyframes sweep {
    0%   { left: -200%; }
    100% { left: 200%; }
}

/* Floating orbs */
.orb1, .orb2, .orb3 {
    position: fixed; border-radius: 50%;
    filter: blur(80px); opacity: 0.12;
    z-index: 1; pointer-events: none;
}
.orb1 {
    width: 500px; height: 500px;
    background: radial-gradient(circle, #b0b0b0, #555);
    top: -100px; left: -100px;
    animation: float1 12s ease-in-out infinite alternate;
}
.orb2 {
    width: 400px; height: 400px;
    background: radial-gradient(circle, #d0d0d0, #444);
    bottom: -80px; right: -80px;
    animation: float2 15s ease-in-out infinite alternate;
}
.orb3 {
    width: 300px; height: 300px;
    background: radial-gradient(circle, #999, #333);
    top: 40%; left: 50%;
    animation: float3 10s ease-in-out infinite alternate;
}
@keyframes float1 { from { transform: translate(0,0) scale(1); } to { transform: translate(80px,60px) scale(1.1); } }
@keyframes float2 { from { transform: translate(0,0) scale(1); } to { transform: translate(-60px,-40px) scale(1.2); } }
@keyframes float3 { from { transform: translate(-50%,-50%) scale(1); } to { transform: translate(-50%,-50%) scale(0.8); } }

/* ── Layout ─────────────────────────────────────────────────────── */
.page-wrap {
    position: relative; z-index: 10;
    display: flex; align-items: center;
    min-height: 100vh; padding: 0 5%;
    gap: 0;
}
.hero-side {
    flex: 1.1; padding: 60px 40px 40px 40px;
    animation: fadeSlideLeft 1s ease-out;
}
.card-side {
    flex: 0.9; display: flex; justify-content: center;
    align-items: center; padding: 40px 20px;
    animation: fadeSlideRight 1s ease-out;
}
@keyframes fadeSlideLeft  { from { opacity:0; transform:translateX(-40px); } to { opacity:1; transform:translateX(0); } }
@keyframes fadeSlideRight { from { opacity:0; transform:translateX(40px);  } to { opacity:1; transform:translateX(0); } }

/* ── Hero typography ─────────────────────────────────────────────── */
.hero-logo {
    display: flex; align-items: center; gap: 14px; margin-bottom: 42px;
    animation: fadeIn 1.2s ease-out 0.2s both;
}
.logo-icon {
    width: 52px; height: 52px; border-radius: 12px;
    background: linear-gradient(135deg, #2a2a2a, #444);
    border: 1px solid rgba(200,200,200,0.25);
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.1);
}
.logo-text-faa { color: #e8e8e8; font-weight: 900; font-size: 1.1em; letter-spacing: 0.04em; }
.logo-text-sub  { color: #888; font-size: 0.65em; line-height: 1.3; margin-top: 1px; }

.hero-title {
    font-size: 3.2em; font-weight: 900; line-height: 1.05;
    margin: 0 0 12px 0;
    background: linear-gradient(135deg, #f0f0f0 0%, #b8b8b8 40%, #e0e0e0 70%, #c8c8c8 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
    text-shadow: none;
    animation: fadeIn 1s ease-out 0.3s both;
}
.hero-sub {
    font-size: 1.05em; font-weight: 500; color: #999;
    letter-spacing: 0.06em; margin: 0 0 14px 0;
    animation: fadeIn 1s ease-out 0.45s both;
    text-transform: uppercase;
}
.hero-desc {
    color: #666; font-size: 0.9em; line-height: 1.7;
    max-width: 400px; margin: 0 0 40px 0;
    animation: fadeIn 1s ease-out 0.55s both;
}

/* Metallic divider */
.metal-divider {
    width: 60px; height: 2px; margin: 18px 0 24px 0;
    background: linear-gradient(90deg, transparent, #aaa, transparent);
    animation: fadeIn 1s ease-out 0.5s both;
}

/* Feature cards */
.features-row {
    display: flex; gap: 16px; margin-top: 8px;
    animation: fadeIn 1s ease-out 0.7s both;
}
.feat-card {
    background: linear-gradient(145deg, rgba(30,30,30,0.9), rgba(18,18,18,0.95));
    border: 1px solid rgba(200,200,200,0.1);
    border-radius: 10px; padding: 16px 14px; flex: 1;
    transition: transform 0.2s, border-color 0.2s;
    box-shadow: 0 2px 12px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.05);
}
.feat-card:hover {
    transform: translateY(-3px);
    border-color: rgba(200,200,200,0.25);
}
.feat-card h4 { color: #ccc; font-size: 0.9em; margin: 8px 0 5px 0; }
.feat-card p  { color: #666; font-size: 0.75em; margin: 0; line-height: 1.5; }

/* ── Login card ──────────────────────────────────────────────────── */
.login-card {
    background: linear-gradient(160deg, rgba(22,22,22,0.97), rgba(12,12,12,0.99));
    border: 1px solid rgba(200,200,200,0.15);
    border-radius: 18px; padding: 42px 38px;
    width: 100%; max-width: 420px;
    box-shadow:
        0 30px 80px rgba(0,0,0,0.7),
        0 0 0 1px rgba(255,255,255,0.04) inset,
        0 1px 0 rgba(255,255,255,0.1) inset;
    backdrop-filter: blur(20px);
    position: relative; overflow: hidden;
}
/* Metallic top edge shimmer */
.login-card::before {
    content: '';
    position: absolute; top: 0; left: 10%; right: 10%; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(220,220,220,0.4), transparent);
}
/* Subtle animated inner glow */
.login-card::after {
    content: '';
    position: absolute; top: -50%; left: -50%; width: 200%; height: 200%;
    background: radial-gradient(circle at 50% 0%, rgba(180,180,180,0.04), transparent 60%);
    animation: cardGlow 4s ease-in-out infinite alternate;
    pointer-events: none;
}
@keyframes cardGlow {
    from { opacity: 0.5; }
    to   { opacity: 1; }
}

.secure-badge {
    display: flex; align-items: center; justify-content: center;
    gap: 7px; margin-bottom: 24px;
    color: #888; font-size: 0.72em; font-weight: 700;
    letter-spacing: 0.18em; text-transform: uppercase;
}
.secure-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: #888;
    animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.5; transform: scale(0.7); }
}

.card-title { color: #e0e0e0; font-size: 1.55em; font-weight: 800; text-align: center; margin: 0 0 4px 0; }
.card-sub   { color: #555; font-size: 0.83em; text-align: center; margin: 0 0 28px 0; }

.field-label {
    color: #888; font-size: 0.78em; font-weight: 600;
    letter-spacing: 0.06em; text-transform: uppercase;
    margin: 0 0 7px 0;
}

.divider-or {
    display: flex; align-items: center; gap: 12px;
    margin: 16px 0; color: #333; font-size: 0.8em;
}
.divider-or::before, .divider-or::after {
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(200,200,200,0.12), transparent);
}

/* Streamlit widget overrides */
div[data-testid="stForm"] { background: transparent !important; border: none !important; padding: 0 !important; }
div.stTextInput > div > div {
    background: rgba(10,10,10,0.9) !important;
    border: 1px solid rgba(200,200,200,0.15) !important;
    border-radius: 9px !important;
    transition: border-color 0.2s;
}
div.stTextInput > div > div:focus-within {
    border-color: rgba(200,200,200,0.45) !important;
    box-shadow: 0 0 0 3px rgba(180,180,180,0.08) !important;
}
div.stTextInput input { color: #ddd !important; }
div.stTextInput input::placeholder { color: #444 !important; }
div.stCheckbox label { color: #666 !important; font-size: 0.85em !important; }
div.stFormSubmitButton button[type=submit] {
    background: linear-gradient(135deg, #3a3a3a 0%, #555 40%, #3a3a3a 100%) !important;
    color: #e8e8e8 !important;
    font-weight: 700 !important; font-size: 1em !important;
    border-radius: 10px !important; border: 1px solid rgba(220,220,220,0.2) !important;
    padding: 14px !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.12) !important;
    transition: all 0.2s !important;
    letter-spacing: 0.03em !important;
}
div.stFormSubmitButton button:hover {
    background: linear-gradient(135deg, #484848 0%, #666 40%, #484848 100%) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 30px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.15) !important;
}

.access-btn {
    background: linear-gradient(135deg, rgba(20,20,20,0.9), rgba(10,10,10,0.95));
    border: 1px solid rgba(200,200,200,0.1);
    border-radius: 9px; padding: 14px 18px; cursor: pointer;
    display: flex; align-items: center; gap: 12px;
    color: #666; font-size: 0.84em; width: 100%;
    transition: border-color 0.2s, transform 0.2s;
}
.access-btn:hover { border-color: rgba(200,200,200,0.25); transform: translateY(-1px); }

/* ── Footer ──────────────────────────────────────────────────────── */
.page-footer {
    position: fixed; bottom: 0; left: 0; right: 0; z-index: 20;
    display: flex; align-items: center; justify-content: space-between;
    padding: 10px 48px;
    background: rgba(5,5,5,0.85);
    border-top: 1px solid rgba(200,200,200,0.06);
    backdrop-filter: blur(12px);
    color: #333; font-size: 0.7em;
}

/* Animated scanning lines (radar sweep effect) */
.scan-line {
    position: fixed; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent 0%, rgba(180,180,180,0.15) 50%, transparent 100%);
    pointer-events: none; z-index: 3;
    animation: scanDown 8s linear infinite;
}
@keyframes scanDown {
    0%   { top: -2%; opacity: 0; }
    5%   { opacity: 1; }
    95%  { opacity: 1; }
    100% { top: 102%; opacity: 0; }
}
.scan-line:nth-child(2) { animation-delay: -4s; opacity: 0.5; }

/* ── Flying planes ──────────────────────────────────────────────── */
.plane {
    position: fixed; z-index: 4; pointer-events: none;
    opacity: 0; will-change: transform, left, right;
}
.plane1 { top: 62%; animation: fly1 22s linear 1.5s infinite; }
.plane2 { top: 18%; animation: fly2 32s linear 9s infinite; }
.plane3 { top: 40%; animation: fly3 42s linear 20s infinite; }

@keyframes fly1 {
    0%   { left:-200px; opacity:0;    transform:scale(0.8); }
    4%   { opacity:0.7; }
    88%  { opacity:0.55; }
    96%  { opacity:0; }
    100% { left:115vw; opacity:0; transform:scale(0.8); }
}
@keyframes fly2 {
    0%   { left:-140px; opacity:0;    transform:scale(0.45); }
    5%   { opacity:0.4; }
    90%  { opacity:0.3; }
    97%  { opacity:0; }
    100% { left:115vw; opacity:0; transform:scale(0.45); }
}
@keyframes fly3 {
    0%   { left:115vw; opacity:0;    transform:scale(0.28) scaleX(-1); }
    5%   { opacity:0.25; }
    90%  { opacity:0.18; }
    97%  { opacity:0; }
    100% { left:-200px; opacity:0; transform:scale(0.28) scaleX(-1); }
}

@keyframes fadeIn { from { opacity:0; transform:translateY(10px); } to { opacity:1; transform:translateY(0); } }
</style>

<!-- Animated background layers -->
<div class="faa-bg"></div>
<div class="faa-grid"></div>
<div class="orb1"></div>
<div class="orb2"></div>
<div class="orb3"></div>
<div class="faa-sweep"></div>
<div class="scan-line"></div>
<div class="scan-line"></div>

<!-- Flying planes — inline SVG aircraft silhouettes -->
<div class="plane plane1">
  <svg width="160" height="44" viewBox="0 0 160 44" fill="none" xmlns="http://www.w3.org/2000/svg">
    <!-- Contrail -->
    <defs>
      <linearGradient id="trail1" x1="0" y1="0" x2="1" y2="0">
        <stop offset="0%" stop-color="rgba(200,200,200,0)"/>
        <stop offset="100%" stop-color="rgba(200,200,200,0.22)"/>
      </linearGradient>
    </defs>
    <rect x="0" y="17" width="110" height="1.2" fill="url(#trail1)"/>
    <rect x="0" y="26" width="90" height="0.8" fill="url(#trail1)" opacity="0.5"/>
    <!-- Fuselage -->
    <path d="M112 21 L148 21 Q158 21 158 22.5 Q158 24 148 24 L112 24 Z" fill="#c0c0c0"/>
    <!-- Nose -->
    <path d="M148 21 Q160 22.5 148 24 Z" fill="#d8d8d8"/>
    <!-- Tail fin -->
    <path d="M114 21 L118 12 L122 21 Z" fill="#b0b0b0"/>
    <!-- Horizontal stabiliser -->
    <path d="M112 22.5 L106 18 L114 22.5 L106 27 Z" fill="#b8b8b8" opacity="0.8"/>
    <!-- Wings -->
    <path d="M132 22.5 L126 10 L144 22.5 L126 35 Z" fill="#bfbfbf"/>
    <!-- Engine pods -->
    <ellipse cx="131" cy="12.5" rx="4" ry="2" fill="#aaa"/>
    <ellipse cx="131" cy="32.5" rx="4" ry="2" fill="#aaa"/>
    <!-- Window strip -->
    <rect x="136" y="21.5" width="8" height="1.5" rx="0.7" fill="rgba(240,240,255,0.5)"/>
    <rect x="128" y="21.5" width="5" height="1.5" rx="0.7" fill="rgba(240,240,255,0.4)"/>
  </svg>
</div>

<div class="plane plane2">
  <svg width="100" height="28" viewBox="0 0 100 28" fill="none" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <linearGradient id="trail2" x1="0" y1="0" x2="1" y2="0">
        <stop offset="0%" stop-color="rgba(200,200,200,0)"/>
        <stop offset="100%" stop-color="rgba(200,200,200,0.18)"/>
      </linearGradient>
    </defs>
    <rect x="0" y="11" width="68" height="0.9" fill="url(#trail2)"/>
    <rect x="0" y="16" width="55" height="0.6" fill="url(#trail2)" opacity="0.4"/>
    <!-- Fuselage -->
    <path d="M70 13 L92 13 Q99 13 99 14.5 Q99 16 92 16 L70 16 Z" fill="#c8c8c8"/>
    <path d="M92 13 Q100 14.5 92 16 Z" fill="#ddd"/>
    <!-- Tail -->
    <path d="M71 13 L74 7 L77 13 Z" fill="#b0b0b0"/>
    <path d="M70 14.5 L66 11.5 L71 14.5 L66 17.5 Z" fill="#b8b8b8" opacity="0.8"/>
    <!-- Wings -->
    <path d="M83 14.5 L78 6 L90 14.5 L78 23 Z" fill="#c0c0c0"/>
    <ellipse cx="81" cy="7.5" rx="3" ry="1.5" fill="#aaa"/>
    <ellipse cx="81" cy="21.5" rx="3" ry="1.5" fill="#aaa"/>
  </svg>
</div>

<div class="plane plane3">
  <svg width="70" height="20" viewBox="0 0 70 20" fill="none" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <linearGradient id="trail3" x1="0" y1="0" x2="1" y2="0">
        <stop offset="0%" stop-color="rgba(200,200,200,0)"/>
        <stop offset="100%" stop-color="rgba(200,200,200,0.14)"/>
      </linearGradient>
    </defs>
    <rect x="0" y="8" width="46" height="0.7" fill="url(#trail3)"/>
    <!-- Fuselage -->
    <path d="M48 9 L64 9 Q70 9 70 10.5 Q70 12 64 12 L48 12 Z" fill="#bfbfbf"/>
    <path d="M64 9 Q70 10.5 64 12 Z" fill="#d0d0d0"/>
    <path d="M49 9 L51 4.5 L54 9 Z" fill="#aaa"/>
    <path d="M48 10.5 L44 8 L49 10.5 L44 13 Z" fill="#b4b4b4" opacity="0.8"/>
    <path d="M59 10.5 L54 4 L64 10.5 L54 17 Z" fill="#b8b8b8"/>
  </svg>
</div>
<div class="orb2"></div>
<div class="orb3"></div>
<div class="faa-sweep"></div>
<div class="scan-line"></div>
<div class="scan-line"></div>
""", unsafe_allow_html=True)

    # ── Left hero panel ────────────────────────────────────────────────────────
    hero_col, card_col = st.columns([1.15, 0.95])

    with hero_col:
        st.markdown("""
<div class="hero-side">

  <div class="hero-logo">
    <div class="logo-icon">
      <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
        <path d="M14 2L5 7v7c0 6.5 4 12 9 13.5C19 25.5 23 20 23 14V7L14 2z"
              fill="none" stroke="rgba(200,200,200,0.6)" stroke-width="1.5"/>
        <path d="M10 14l3 3 6-6" stroke="#aaa" stroke-width="1.5" stroke-linecap="round"/>
      </svg>
    </div>
    <div>
      <div class="logo-text-faa">FAA</div>
      <div class="logo-text-sub">Federal Aviation<br>Administration</div>
    </div>
  </div>

  <h1 class="hero-title">FAA RF<br>Interference Tool</h1>
  <p class="hero-sub">ITU-R Working Party Policy Support</p>
  <div class="metal-divider"></div>
  <p class="hero-desc">Analyze and assess RF interference risks to support global aviation spectrum protection and policy decisions.</p>

  <div class="features-row">
    <div class="feat-card">
      <svg width="22" height="22" viewBox="0 0 22 22" fill="none">
        <rect x="1" y="13" width="3" height="7" rx="1" fill="#777"/>
        <rect x="6" y="9" width="3" height="11" rx="1" fill="#777" opacity="0.8"/>
        <rect x="11" y="5" width="3" height="15" rx="1" fill="#aaa" opacity="0.9"/>
        <rect x="16" y="7" width="3" height="13" rx="1" fill="#777" opacity="0.8"/>
      </svg>
      <h4>Signal Analysis</h4>
      <p>Detect and analyze RF emissions with precision and clarity.</p>
    </div>
    <div class="feat-card">
      <svg width="22" height="22" viewBox="0 0 22 22" fill="none">
        <path d="M11 1L3 5.5v5C3 15.5 6.5 19.5 11 21c4.5-1.5 8-5.5 8-10.5v-5L11 1z"
              stroke="#aaa" stroke-width="1.3" fill="none"/>
        <path d="M8 11l2 2 4-4" stroke="#aaa" stroke-width="1.3" stroke-linecap="round"/>
      </svg>
      <h4>Protected Band Assessment</h4>
      <p>Evaluate interference to aviation services and protected bands.</p>
    </div>
    <div class="feat-card">
      <svg width="22" height="22" viewBox="0 0 22 22" fill="none">
        <rect x="2" y="2" width="18" height="18" rx="2" stroke="#aaa" stroke-width="1.3" fill="none"/>
        <path d="M6 16l3-5 3 3 3-6" stroke="#aaa" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
      <h4>Policy Support</h4>
      <p>Generate insights to inform ITU-R working party decisions.</p>
    </div>
  </div>

</div>
""", unsafe_allow_html=True)

    # ── Right card panel ───────────────────────────────────────────────────────
    with card_col:
        st.markdown("""
<div class="card-side">
<div class="login-card">
  <div class="secure-badge">
    <div class="secure-dot"></div>
    SECURE ACCESS
    <div class="secure-dot"></div>
  </div>
  <h2 class="card-title">Welcome back</h2>
  <p class="card-sub">Sign in to continue to the tool</p>
</div></div>
""", unsafe_allow_html=True)

        users = load_users()

        if not users:
            st.markdown('<div style="padding:0 20px">', unsafe_allow_html=True)
            st.warning("⚠️ No users configured. Add credentials to Streamlit Secrets.")
            with st.expander("Setup instructions"):
                st.markdown("""
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
            st.markdown('</div>', unsafe_allow_html=True)
            return False

        with st.container():
            st.markdown('<div style="padding:0 28px; max-width:420px; margin:-24px auto 0 auto;">', unsafe_allow_html=True)

            with st.form("login_form", clear_on_submit=False):
                st.markdown('<p class="field-label">Username</p>', unsafe_allow_html=True)
                username = st.text_input("", placeholder="Enter your username",
                    label_visibility="collapsed").strip().lower()

                st.markdown('<p class="field-label" style="margin-top:14px;">Password</p>',
                    unsafe_allow_html=True)
                password = st.text_input("", placeholder="Enter your password",
                    type="password", label_visibility="collapsed")

                rc1, rc2 = st.columns([1, 1])
                with rc1:
                    st.checkbox("Remember me", value=True)
                with rc2:
                    st.markdown('<div style="text-align:right;padding-top:6px;"><span style="color:#666;font-size:0.8em;">Forgot password?</span></div>',
                        unsafe_allow_html=True)

                st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)
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
                    st.session_state["auth_ok"]       = True
                    st.session_state["auth_username"]  = username
                    st.session_state["auth_user"]      = {
                        "username": username,
                        "name":     users[username]["name"],
                        "role":     users[username]["role"],
                    }
                    st.rerun()

            st.markdown("""
<div class="divider-or">or</div>
<div class="access-btn">
  <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
    <circle cx="10" cy="6.5" r="3.5" stroke="#666" stroke-width="1.3"/>
    <path d="M3 17.5c0-3.5 3.1-6.5 7-6.5s7 3 7 6.5" stroke="#666" stroke-width="1.3" stroke-linecap="round"/>
    <circle cx="15.5" cy="4.5" r="2.5" fill="#111" stroke="#aaa" stroke-width="1.1"/>
    <path d="M14.5 4.5h2M15.5 3.5v2" stroke="#aaa" stroke-width="0.9" stroke-linecap="round"/>
  </svg>
  <div>
    <div style="color:#bbb;font-weight:600;font-size:0.87em;">Need access?</div>
    <div style="color:#555;font-size:0.76em;">Request an account</div>
  </div>
</div>
<div style="text-align:center;margin-top:14px;color:#444;font-size:0.8em;">Contact administrator ✉</div>
""", unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

    # ── Footer ─────────────────────────────────────────────────────────────────
    st.markdown("""
<div class="page-footer">
  <span>🔒&nbsp; Secure &nbsp;•&nbsp; Compliant &nbsp;•&nbsp; Reliable</span>
  <span>Supporting safe and efficient access to the radio spectrum for global aviation.</span>
  <span>© 2024 Federal Aviation Administration</span>
</div>
""", unsafe_allow_html=True)

    return False
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
