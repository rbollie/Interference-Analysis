"""
=============================================================================
FAA RF Interference Analysis Tool
For ITU-R Policy Support — Aeronautical Spectrum Protection
=============================================================================
Purpose : Help RF engineers supporting FAA/NTIA policy assess interference
          threats to protected aeronautical frequencies before and during
          ITU-R WP 5D / 5B proceedings.
Libraries: itur (ITU-R propagation models), numpy, pandas, matplotlib
=============================================================================
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import itur
import itur.models.itu676 as itu676
import warnings
import anthropic
import math
from auth import (show_login_page, show_admin_panel, is_authenticated,
                  is_admin, current_user, logout)
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FAA RF Interference Analysis Tool",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# AUTH GATE — must pass before anything else renders
# ─────────────────────────────────────────────────────────────────────────────
if not is_authenticated():
    show_login_page()
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.explainer {
    background: #1a2a3a;
    border-left: 4px solid #00aaff;
    padding: 6px 12px;
    border-radius: 4px;
    color: #aaddff;
    font-size: 0.85em;
    margin-bottom: 10px;
}
.warn-box {
    background: #3a1a1a;
    border-left: 4px solid #ff4444;
    padding: 6px 12px;
    border-radius: 4px;
    color: #ffaaaa;
    font-size: 0.85em;
    margin-bottom: 10px;
}
.ok-box {
    background: #1a3a1a;
    border-left: 4px solid #44ff44;
    padding: 6px 12px;
    border-radius: 4px;
    color: #aaffaa;
    font-size: 0.85em;
    margin-bottom: 10px;
}
.metric-label { font-size: 0.75em; color: #888; }
</style>
""", unsafe_allow_html=True)

def ex(text):
    """Render a one-line blue explainer box."""
    st.markdown(f'<div class="explainer">💡 {text}</div>', unsafe_allow_html=True)

def warn(text):
    st.markdown(f'<div class="warn-box">⚠️ {text}</div>', unsafe_allow_html=True)

def ok(text):
    st.markdown(f'<div class="ok-box">✅ {text}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FAA PROTECTED BAND DATABASE
# ─────────────────────────────────────────────────────────────────────────────
FAA_BANDS = {
    "VOR / ILS Localizer": {
        "f_low_mhz": 108.0, "f_high_mhz": 117.975,
        "system": "VOR / ILS Localizer",
        "allocation": "ARNS",
        "service_category": "AM(R)S + ARNS",
        "in_threshold_db": -6,
        "aviation_safety_factor_db": 6,
        "effective_threshold_db": -12,
        "noise_floor_dbm": -120,
        "safety_of_life": True,
        "rr_1_59": True,
        "protection_basis": "I/N ≤ −6 dB + 6 dB aviation safety factor for precision approach",
        "spr_source": "Max EIRP toward airport; worst-case azimuth",
        "spr_path": "FSPL; distances >20 km; free-space worst-case",
        "spr_victim": "Receiver susceptibility mask per DO-196; −120 dBm noise floor",
        "notes": "En-route navigation and precision approach guidance. ILS CAT III approach — 6 dB additional safety factor applies.",
        "rtca_doc": "DO-196",
    },
    "ILS Glide Slope": {
        "f_low_mhz": 328.6, "f_high_mhz": 335.4,
        "system": "ILS Glide Slope",
        "allocation": "ARNS",
        "service_category": "ARNS",
        "in_threshold_db": -6,
        "aviation_safety_factor_db": 6,
        "effective_threshold_db": -12,
        "noise_floor_dbm": -115,
        "safety_of_life": True,
        "rr_1_59": True,
        "protection_basis": "I/N ≤ −6 dB + 6 dB aviation safety factor; CAT III precision approach",
        "spr_source": "Max EIRP interferer; worst-case signal characteristics",
        "spr_path": "FSPL worst-case; free-space attenuation above 300 MHz",
        "spr_victim": "Receiver susceptibility per DO-148; noise floor −115 dBm",
        "notes": "Vertical guidance for precision landings (CAT I/II/III). Safety factor mandatory for CAT III.",
        "rtca_doc": "DO-148",
    },
    "DME / TACAN": {
        "f_low_mhz": 960.0, "f_high_mhz": 1215.0,
        "system": "DME / TACAN / SSR / TCAS",
        "allocation": "ARNS + ANS",
        "service_category": "ARNS + ANS",
        "in_threshold_db": -6,
        "aviation_safety_factor_db": 0,
        "effective_threshold_db": -6,
        "epfd_threshold_dbw_m2_mhz": -121.5,
        "noise_floor_dbm": -106,
        "safety_of_life": True,
        "rr_1_59": True,
        "protection_basis": "I/N ≤ −6 dB; epfd ≤ −121.5 dBW/m² in any 1 MHz band (from slide)",
        "spr_source": "Max Tx power; worst-case antenna gain toward aircraft",
        "spr_path": "FSPL; distance separation >20 km worst-case",
        "spr_victim": "Receiver susceptibility mask; antenna gain; noise power per DO-189",
        "notes": "Distance measuring, ATC surveillance, collision avoidance. epfd limit applies to satellite downlinks.",
        "rtca_doc": "DO-189 / DO-185B",
    },
    "ADS-B / Mode-S (1090 MHz)": {
        "f_low_mhz": 1085.0, "f_high_mhz": 1095.0,
        "system": "ADS-B / Mode-S Transponder",
        "allocation": "ARNS + ANS",
        "service_category": "ANS (Aeronautical Navigation Service)",
        "in_threshold_db": -10,
        "aviation_safety_factor_db": 0,
        "effective_threshold_db": -10,
        "noise_floor_dbm": -100,
        "safety_of_life": True,
        "rr_1_59": True,
        "protection_basis": "I/N ≤ −10 dB; ASR protection level per system protection table",
        "spr_source": "Max EIRP; worst-case signal characteristics",
        "spr_path": "FSPL worst-case; distances >20 km",
        "spr_victim": "Receiver susceptibility per DO-260B; noise floor −100 dBm",
        "notes": "1090 MHz squitter — global ATC surveillance backbone. ASR threshold −10 dB applies.",
        "rtca_doc": "DO-260B",
    },
    "GNSS L5 / ARNS": {
        "f_low_mhz": 1164.0, "f_high_mhz": 1215.0,
        "system": "GNSS L5 / Galileo E5",
        "allocation": "ARNS + RNSS",
        "service_category": "RNSS + ARNS",
        "in_threshold_db": -10,
        "aviation_safety_factor_db": 6,
        "effective_threshold_db": -16,
        "delta_t_t_pct_aggregate": 6.0,
        "noise_floor_dbm": -130,
        "safety_of_life": True,
        "rr_1_59": True,
        "protection_basis": "I/N ≤ −10 dB wideband; ΔT/T ≤ 6% single-entry for RNSS feeder links; +6 dB aviation safety factor for safety-of-life applications",
        "spr_source": "Max EIRP; worst-case signal characteristics",
        "spr_path": "FSPL worst-case; distances >20 km; aggregate sources",
        "spr_victim": "Receiver susceptibility mask per DO-292; noise floor −130 dBm",
        "notes": "Safety-of-life GNSS signal; aviation approach procedures. ΔT/T = 6% for RNSS feeder links.",
        "rtca_doc": "DO-292",
    },
    "GPS L1 / GNSS": {
        "f_low_mhz": 1559.0, "f_high_mhz": 1610.0,
        "system": "GPS L1 / GLONASS / Galileo E1",
        "allocation": "RNSS + ARNS",
        "service_category": "RNSS + ARNS",
        "in_threshold_db": -10,
        "aviation_safety_factor_db": 6,
        "effective_threshold_db": -16,
        "psd_threshold_dbw_mhz": -146.5,
        "noise_floor_dbm": -130,
        "safety_of_life": True,
        "rr_1_59": True,
        "protection_basis": "I < −146.5 dBW/MHz (L1 SBAS Type 1 acquisition); I/N ≈ −5 dB + 6 dB safety margin (wideband RFI); L2 SBAS Gnd Ref: I < −147.5 dBW/MHz, I/N ≈ −6 dB",
        "spr_source": "Max EIRP; worst-case antenna gain; signal characteristics",
        "spr_path": "FSPL worst-case; aggregate effect of multiple sources",
        "spr_victim": "Receiver susceptibility per DO-235B/DO-253; noise floor −130 dBm",
        "notes": "Primary GNSS band. SBAS/WAAS critical. PSD limit −146.5 dBW/MHz applies to wideband RFI per system protection table.",
        "rtca_doc": "DO-235B / DO-253",
    },
    "En-Route Radar": {
        "f_low_mhz": 2700.0, "f_high_mhz": 2900.0,
        "system": "ATC En-Route Surveillance Radar (ARSR / ASR)",
        "allocation": "ARNS + RN",
        "service_category": "ARNS (Safety Service per RR 1.59)",
        "in_threshold_db": -6,
        "aviation_safety_factor_db": 0,
        "effective_threshold_db": -6,
        "noise_floor_dbm": -100,
        "safety_of_life": True,
        "rr_1_59": True,
        "protection_basis": "ARSR: I/N ≤ −6 dB; ASR: I/N ≤ −10 dB (per system protection levels table)",
        "spr_source": "Max EIRP; worst-case azimuth toward radar",
        "spr_path": "FSPL; worst-case for distances >20 km",
        "spr_victim": "Radar receiver susceptibility; noise power; antenna gain",
        "notes": "ARSR (long-range) I/N = −6 dB; ASR (short-range, airport) I/N = −10 dB. Both from FAA system protection table.",
        "rtca_doc": "N/A (ITU-R M.1849)",
    },
    "Radio Altimeter": {
        "f_low_mhz": 4200.0, "f_high_mhz": 4400.0,
        "system": "Radio Altimeter (Rad Alt)",
        "allocation": "ARNS",
        "service_category": "ARNS (Safety Service per RR 1.59)",
        "in_threshold_db": -6,
        "aviation_safety_factor_db": 6,
        "effective_threshold_db": -12,
        "noise_floor_dbm": -90,
        "safety_of_life": True,
        "rr_1_59": True,
        "protection_basis": "I/N ≤ −6 dB + 6 dB aviation safety factor for CAT III precision approach and landing",
        "spr_source": "Max EIRP interferer (e.g. 5G base station); OOB/spurious emissions",
        "spr_path": "FSPL + possible blocking; distances >20 km worst-case; aggregate effect",
        "spr_victim": "LNA susceptibility mask; blocking threshold; noise floor −90 dBm",
        "notes": "Critical for CAT III landings, TAWS, GPWS, helicopter ops. 5G adjacent band (3.7–3.98 GHz) blocking risk.",
        "rtca_doc": "DO-155 / ETSO-C87",
    },
    "ARNS 5 GHz": {
        "f_low_mhz": 5000.0, "f_high_mhz": 5150.0,
        "system": "ARNS / Future Aeronautical Systems",
        "allocation": "ARNS",
        "service_category": "ARNS",
        "in_threshold_db": -6,
        "aviation_safety_factor_db": 6,
        "effective_threshold_db": -12,
        "noise_floor_dbm": -95,
        "safety_of_life": True,
        "rr_1_59": True,
        "protection_basis": "I/N ≤ −6 dB + 6 dB aviation safety factor; under pressure from IMT-2030 WP 5D studies",
        "spr_source": "Potential IMT base station EIRP; OOB emissions from adjacent IMT bands",
        "spr_path": "FSPL worst-case; P.528 for airborne victim",
        "spr_victim": "Future ARNS receiver; noise floor −95 dBm",
        "notes": "Protected for future aeronautical use; under pressure from IMT. WRC-27 WP 5D AI 1.2 threat.",
        "rtca_doc": "N/A",
    },
    "Airborne Weather Radar": {
        "f_low_mhz": 9000.0, "f_high_mhz": 9500.0,
        "system": "Airborne / Surface Movement Radar",
        "allocation": "ARNS + RN",
        "service_category": "ARNS (Safety Service per RR 1.59)",
        "in_threshold_db": -6,
        "aviation_safety_factor_db": 0,
        "effective_threshold_db": -6,
        "noise_floor_dbm": -95,
        "safety_of_life": True,
        "rr_1_59": True,
        "protection_basis": "I/N ≤ −6 dB; safety service per RR 1.59",
        "spr_source": "Max EIRP; worst-case signal characteristics",
        "spr_path": "FSPL worst-case; gaseous absorption significant at X-band",
        "spr_victim": "Radar receiver susceptibility; noise floor −95 dBm",
        "notes": "X-band weather radar and airport surface detection. Governed by ITU-R M.1849.",
        "rtca_doc": "DO-220",
    },
    "MLS (Microwave Landing System)": {
        "f_low_mhz": 5030.0, "f_high_mhz": 5091.0,
        "system": "Microwave Landing System (MLS)",
        "allocation": "ARNS",
        "service_category": "ARNS (Safety Service per RR 1.59)",
        "in_threshold_db": -6,
        "aviation_safety_factor_db": 6,
        "effective_threshold_db": -12,
        "pfd_threshold_dbw_m2_khz": -124.5,
        "noise_floor_dbm": -110,
        "safety_of_life": True,
        "rr_1_59": True,
        "protection_basis": "pfd ≤ −124.5 dBW/m² in 150 kHz band (from FAA system protection table); I/N ≤ −6 dB + 6 dB safety factor",
        "spr_source": "Max EIRP; signal characteristics",
        "spr_path": "FSPL; worst-case approach geometry",
        "spr_victim": "MLS receiver susceptibility; noise floor −110 dBm",
        "notes": "Precision approach system. PFD limit −124.5 dBW/m² in 150 kHz band.",
        "rtca_doc": "N/A",
    },
    "L-band AMS(R)S": {
        "f_low_mhz": 1525.0, "f_high_mhz": 1559.0,
        "system": "L-band Aeronautical Mobile Satellite (Route) Service",
        "allocation": "AMS(R)S",
        "service_category": "AMS(R)S — Aeronautical Mobile Satellite (Route) Service",
        "in_threshold_db": -6,
        "aviation_safety_factor_db": 0,
        "effective_threshold_db": -6,
        "delta_t_t_pct_aggregate": 20.0,
        "delta_t_t_pct_single": 6.0,
        "noise_floor_dbm": -120,
        "safety_of_life": True,
        "rr_1_59": True,
        "protection_basis": "ΔT/T ≤ 20% aggregate, ΔT/T ≤ 6% single-entry (from FAA system protection table)",
        "spr_source": "Satellite downlink EIRP; terrestrial co-frequency EIRP",
        "spr_path": "Slant path (P.619); FSPL worst-case for terrestrial",
        "spr_victim": "Aircraft terminal; noise temperature; antenna gain",
        "notes": "INMARSAT/IRIDIUM safety comms. ΔT/T metric used (not I/N). 20% aggregate = 6% single-entry per protection table.",
        "rtca_doc": "N/A",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# CORE CALCULATION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def noise_floor_dbm(bandwidth_hz: float, noise_figure_db: float) -> float:
    """Thermal noise floor: N = -174 + 10log10(BW) + NF"""
    return -174.0 + 10.0 * np.log10(bandwidth_hz) + noise_figure_db

def free_space_path_loss_db(freq_mhz: float, dist_km: float) -> float:
    """FSPL(dB) = 20log10(d_km) + 20log10(f_MHz) + 32.44"""
    return 20.0 * np.log10(dist_km) + 20.0 * np.log10(freq_mhz) + 32.44

def received_power_dbm(tx_power_dbm, tx_gain_dbi, path_loss_db,
                        rx_gain_dbi, cable_loss_db=0.0) -> float:
    """Friis: Pr = Pt + Gt - L - cable + Gr"""
    return tx_power_dbm + tx_gain_dbi - path_loss_db + rx_gain_dbi - cable_loss_db

def in_ratio_db(interference_dbm: float, noise_floor_dbm: float) -> float:
    """I/N = interference power relative to receiver noise floor."""
    return interference_dbm - noise_floor_dbm

def protection_margin_db(in_ratio: float, threshold_db: float) -> float:
    """Margin = threshold - I/N.  Positive = protected, negative = violated."""
    return threshold_db - in_ratio

def eirp_dbm(tx_power_dbm: float, tx_gain_dbi: float) -> float:
    """EIRP = transmit power + antenna gain (dBm + dBi)."""
    return tx_power_dbm + tx_gain_dbi

def pfd_dbm_per_m2(eirp_dbm_val: float, dist_km: float, freq_mhz: float) -> float:
    """Power flux density at receiver: PFD = EIRP - FSPL - 10log10(4π/λ²)... simplified."""
    fspl = free_space_path_loss_db(freq_mhz, dist_km)
    wavelength_m = 300.0 / freq_mhz
    return eirp_dbm_val - fspl - 10.0 * np.log10(4.0 * np.pi / wavelength_m**2)

# ─── Simplified ITU-R P.452 terrestrial interference model ───────────────────
def p452_basic_loss_db(freq_mhz: float, dist_km: float,
                        tx_height_m: float, rx_height_m: float,
                        terrain_type: str = "suburban") -> float:
    """
    Simplified P.452 basic transmission loss (line-of-sight + diffraction).
    Full P.452 requires terrain profiles; this uses FSPL + empirical corrections.
    For rigorous analysis use the ITU-R SoftTools P.452 implementation.
    """
    fspl = free_space_path_loss_db(freq_mhz, dist_km)
    # Effective earth radius factor
    k = 4.0 / 3.0  # standard effective earth radius
    Re = 6371.0 * k  # km
    # Fresnel zone clearance check (simplified)
    d1, d2 = dist_km / 2, dist_km / 2
    f_hz = freq_mhz * 1e6
    r1 = np.sqrt(3e8 * d1 * 1e3 * d2 * 1e3 / (f_hz * dist_km * 1e3))
    # Height gain correction (log)
    ht_correction = -10.0 * np.log10(tx_height_m / 10.0) - 10.0 * np.log10(rx_height_m / 10.0)
    terrain_corr = {"open": 0, "suburban": 8, "urban": 18, "dense_urban": 26}
    tc = terrain_corr.get(terrain_type, 8)
    return fspl + tc + ht_correction * 0.5

# ─── Simplified ITU-R P.528 aeronautical propagation ─────────────────────────
def p528_aero_path_loss_db(freq_mhz: float, ground_dist_km: float,
                            aircraft_alt_km: float,
                            time_percent: float = 50.0) -> float:
    """
    Simplified P.528 aeronautical path loss (airborne receiver).
    P.528 accounts for: FSPL + atmospheric absorption + troposcatter.
    Uses itur gaseous attenuation for atmospheric component.
    Full P.528 requires detailed marine-layer / refractive corrections.
    """
    # Slant distance
    slant_km = np.sqrt(ground_dist_km**2 + aircraft_alt_km**2)
    fspl = free_space_path_loss_db(freq_mhz, slant_km)

    # Gaseous attenuation via itur P.676
    try:
        elev_angle_deg = np.degrees(np.arctan(aircraft_alt_km / max(ground_dist_km, 0.1)))
        # itur gaseous_attenuation_terrestrial_path for horizontal path approximation
        if freq_mhz >= 1000:  # Above 1 GHz, gaseous matters more
            gamma = itu676.gaseous_attenuation_terrestrial_path(
                freq_mhz / 1e3,  # GHz
                p=50,
                T=15,
                H=50,
                P=1013.25,
                d=slant_km,
                mode="approx"
            )
            gas_atten_db = float(gamma.value) if hasattr(gamma, 'value') else float(gamma)
        else:
            gas_atten_db = 0.0
    except Exception:
        gas_atten_db = 0.0

    # Time-variability factor (simplified log-normal)
    t_factor = 0.0
    if time_percent < 50:
        t_factor = -5.0 * np.log10(time_percent / 50.0)

    return fspl + gas_atten_db + t_factor

# ─── Monte Carlo Aggregate Interference ──────────────────────────────────────
def monte_carlo_aggregate(
    n_interferers: int,
    tx_power_dbm: float,
    tx_gain_dbi: float,
    freq_mhz: float,
    deployment_radius_km: float,
    rx_noise_floor_dbm: float,
    in_threshold_db: float,
    n_trials: int = 5000,
    model: str = "FSPL",
    terrain_type: str = "suburban",
    aircraft_alt_km: float = 0.0,
) -> dict:
    """
    Monte Carlo simulation of aggregate interference from N random interferers.
    Each trial: randomly place N transmitters, compute total I at victim receiver.
    Returns distribution statistics and exceedance probability.
    """
    violations = 0
    in_values = []
    agg_power_list = []

    for _ in range(n_trials):
        # Random distances: uniform in area → radius ~ sqrt(uniform) * max_radius
        r = np.sqrt(np.random.uniform(0.01**2, deployment_radius_km**2, n_interferers))
        # Small random power variation ±2 dB (log-normal transmitter variation)
        pwr_variation = np.random.normal(0, 2.0, n_interferers)
        effective_eirp = eirp_dbm(tx_power_dbm, tx_gain_dbi) + pwr_variation

        # Path loss per interferer
        path_losses = np.array([
            p528_aero_path_loss_db(freq_mhz, ri, aircraft_alt_km)
            if (model == "P.528" and aircraft_alt_km > 0)
            else p452_basic_loss_db(freq_mhz, ri, 30.0, 5.0, terrain_type)
            if model == "P.452"
            else free_space_path_loss_db(freq_mhz, ri)
            for ri in r
        ])

        rx_powers_dbm = effective_eirp - path_losses
        # Linear sum of interferers
        agg_power_mw = np.sum(10.0 ** (rx_powers_dbm / 10.0))
        agg_power_dbm = 10.0 * np.log10(max(agg_power_mw, 1e-30))

        i_n = in_ratio_db(agg_power_dbm, rx_noise_floor_dbm)
        in_values.append(i_n)
        agg_power_list.append(agg_power_dbm)
        if i_n > in_threshold_db:
            violations += 1

    in_values = np.array(in_values)
    return {
        "in_mean_db": float(np.mean(in_values)),
        "in_p50_db": float(np.percentile(in_values, 50)),
        "in_p95_db": float(np.percentile(in_values, 95)),
        "in_p99_db": float(np.percentile(in_values, 99)),
        "violation_probability": violations / n_trials,
        "in_values": in_values,
        "agg_power_list": np.array(agg_power_list),
        "n_trials": n_trials,
    }

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/17/FAA_seal.svg/200px-FAA_seal.svg.png", width=80)
st.sidebar.title("FAA RF Interference\nAnalysis Tool")
st.sidebar.markdown("*For ITU-R WP 5D / 5B Policy Support*")

# ── User info bar ─────────────────────────────────────────────────────────────
user = current_user()
role_badge = "🔴 Admin" if is_admin() else "🟢 User"
st.sidebar.markdown(
    f"<div style='background:#1a2a1a;border-left:3px solid #44bb44;"
    f"padding:6px 10px;border-radius:4px;font-size:0.82em;color:#aaffaa'>"
    f"👤 <b>{user.get('name','')}</b> &nbsp;|&nbsp; {role_badge}</div>",
    unsafe_allow_html=True
)
if st.sidebar.button("🚪 Sign Out", use_container_width=True):
    logout()

st.sidebar.markdown("---")

tab_names = [
    "📡 Protected Bands",
    "🔗 Link Budget",
    "📊 Noise & I/N",
    "🌐 Propagation",
    "🎲 Monte Carlo",
    "📋 Contribution Summary",
    "📚 Tutorial",
    "🤖 Contribution Analyzer",
    "🎓 RF Training",
    "📓 Meeting Notes",
    "🔬 Code Analyzer",
    "📖 Glossary",
    "📻 Microwave Link Budget",
]

# Admin-only tab
if is_admin():
    tab_names.append("⚙️ Admin Panel")
selected_tab = st.sidebar.radio("Module", tab_names)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Key References**
- ITU-R P.452 — Terrestrial interference
- ITU-R P.528 — Aeronautical propagation  
- ITU-R M.1642 — IMT → ARNS methodology
- RTCA DO-235B — GNSS protection
- RR No. 4.10 — Safety-of-life protection
""")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — PROTECTED BANDS DATABASE
# ─────────────────────────────────────────────────────────────────────────────
if selected_tab == "📡 Protected Bands":
    st.title("📡 FAA Protected Frequency Bands")
    ex("Each band carries a primary or secondary allocation under the ITU Radio Regulations — knowing the allocation type and footnote structure determines which regulatory instruments you can invoke.")

    # Spectrum overview chart
    st.subheader("Spectrum Overview")
    ex("Frequency proximity matters: receiver front-end selectivity is finite — a new allocation 200 MHz away at C-band can still cause LNA desensitization via blocking, even with no spectral overlap.")

    band_colors = [
        "#4fc3f7","#81c784","#ffb74d","#e57373",
        "#ce93d8","#4db6ac","#fff176","#ff8a65",
        "#a5d6a7","#90caf9","#f48fb1","#80cbc4",
    ]

    band_list = list(FAA_BANDS.items())
    n_bands   = len(band_list)

    # ── Swimlane layout: one row per band ─────────────────────────────────────
    # Each band gets its own horizontal row — completely eliminates overlap
    ROW_H    = 0.72        # height of each row
    ROW_GAP  = 0.18        # gap between rows
    ROW_STEP = ROW_H + ROW_GAP
    FIG_H    = n_bands * ROW_STEP + 0.8

    fig, ax = plt.subplots(figsize=(17, FIG_H))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    log_min, log_max = np.log10(90), np.log10(12000)
    min_log_w = (log_max - log_min) * 0.018  # min visible bar width in log space

    # Draw subtle frequency grid lines first
    for kf in [100, 200, 300, 500, 1000, 2000, 3000, 5000, 10000]:
        ax.axvline(kf, color='#1e1e2e', linewidth=0.8, zorder=0)

    for i, (name, b) in enumerate(band_list):
        col  = band_colors[i % len(band_colors)]
        row_y = (n_bands - 1 - i) * ROW_STEP   # top band at top

        # Alternating row background for readability
        row_bg = "#111118" if i % 2 == 0 else "#0e0e15"
        ax.fill_betweenx(
            [row_y, row_y + ROW_H],
            10**log_min, 10**log_max,
            color=row_bg, linewidth=0, zorder=1
        )

        # Band bar in log space
        log_fl = np.log10(b["f_low_mhz"])
        log_fh = np.log10(b["f_high_mhz"])
        if (log_fh - log_fl) < min_log_w:
            lm = (log_fl + log_fh) / 2
            log_fl = lm - min_log_w / 2
            log_fh = lm + min_log_w / 2

        bar_bot = row_y + ROW_H * 0.15
        bar_top = row_y + ROW_H * 0.85

        ax.fill_betweenx([bar_bot, bar_top], 10**log_fl, 10**log_fh,
                         color=col, alpha=0.90, linewidth=0, zorder=3)
        # White border on bar
        ax.plot([10**log_fl, 10**log_fh, 10**log_fh, 10**log_fl, 10**log_fl],
                [bar_bot, bar_bot, bar_top, bar_top, bar_bot],
                color='white', linewidth=0.5, alpha=0.4, zorder=4)

        # Band number badge inside bar (if wide enough)
        log_mid = (log_fl + log_fh) / 2
        bar_center_y = (bar_bot + bar_top) / 2
        ax.text(10**log_mid, bar_center_y, str(i+1),
                ha='center', va='center',
                fontsize=7.5, fontweight='bold',
                color='#0e1117', zorder=5)

        # LEFT label panel: band name
        label_x = 10**log_min * 1.02
        ax.text(label_x, bar_center_y + 0.06, name,
                ha='left', va='center',
                fontsize=9, fontweight='bold', color=col,
                zorder=5)

        # Frequency range under name
        ax.text(label_x, bar_center_y - 0.10,
                f"{b['f_low_mhz']:.0f} – {b['f_high_mhz']:.0f} MHz",
                ha='left', va='center',
                fontsize=8, color='#888888', zorder=5)

        # RIGHT label panel: I/N threshold + allocation
        thresh_eff = b.get("effective_threshold_db", b["in_threshold_db"])
        sf         = b.get("aviation_safety_factor_db", 0)
        thresh_str = f"I/N {b['in_threshold_db']} dB"
        if sf > 0:
            thresh_str += f"  (+{sf} dB)"
        ax.text(10**log_max * 0.985, bar_center_y + 0.06,
                thresh_str,
                ha='right', va='center',
                fontsize=8.5, fontweight='bold',
                color='#ffcc66' if sf > 0 else '#aaffaa', zorder=5)
        ax.text(10**log_max * 0.985, bar_center_y - 0.10,
                b.get("service_category", b["allocation"]).split(" —")[0][:22],
                ha='right', va='center',
                fontsize=7.5, color='#777777', zorder=5)

        # Thin separator line between rows
        ax.axhline(row_y, color='#1a1a2a', linewidth=0.5, zorder=2)

    # Axis formatting
    ax.set_xscale("log")
    ax.set_xlim(10**log_min, 10**log_max)
    ax.set_ylim(-0.1, n_bands * ROW_STEP + 0.1)
    ax.set_yticks([])

    # Custom x-axis ticks at key aeronautical frequencies
    key_ticks = [100, 200, 330, 500, 960, 1090, 1176, 1575, 2000, 2800, 4300, 5000, 9375]
    ax.set_xticks(key_ticks)
    ax.set_xticklabels([str(k) for k in key_ticks],
                       fontsize=8, color='#888888', rotation=45, ha='right')

    ax.tick_params(axis='x', colors='#444', pad=4, length=3)
    ax.spines['bottom'].set_color('#333')
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_title("FAA Protected Aeronautical Frequency Bands",
                 color='white', fontsize=13, fontweight='bold', pad=10, loc='left')
    ax.set_xlabel("Frequency (MHz) — log scale", color='#666', fontsize=9, labelpad=6)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    st.pyplot(fig)

    # Compact legend strip
    leg_cols = st.columns(min(6, n_bands))
    for i, (name, b_leg) in enumerate(band_list):
        col_leg = leg_cols[i % 6]
        col_leg.markdown(
            f"<div style='border-left:3px solid {band_colors[i % len(band_colors)]};padding:2px 6px;"
            f"margin:1px 0;font-size:0.75em;color:#ccc'>"
            f"<b style='color:{band_colors[i % len(band_colors)]}'>{i+1}</b> {name.split('/')[0].strip()}"
            f"</div>",
            unsafe_allow_html=True
        )
    st.markdown("")

    st.subheader("Band Details")
    ex("I/N threshold is defined such that the total noise floor rise stays below ~1 dB — at I/N = −6 dB the noise power increases by 10·log(1 + 10^(−0.6)) ≈ 0.97 dB; at −10 dB it is 0.41 dB.")

    selected_band = st.selectbox("Select a band for details:", list(FAA_BANDS.keys()))
    b = FAA_BANDS[selected_band]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Lower Freq", f"{b['f_low_mhz']} MHz",
        help="Bottom edge of the protected band in Megahertz (MHz) — new allocations must not encroach below this frequency.")
    col2.metric("Upper Freq", f"{b['f_high_mhz']} MHz",
        help="Top edge of the protected band in Megahertz (MHz) — out-of-band emissions from systems above this frequency can still cause Low Noise Amplifier (LNA) desensitization or blocking in the protected receiver.")
    col3.metric("Bandwidth", f"{b['f_high_mhz'] - b['f_low_mhz']:.1f} MHz",
        help="Total protected bandwidth in Megahertz (MHz). Wider bandwidth raises the thermal noise floor (noise power = kTBF), making the receiver less sensitive — but also more spectrum to defend.")
    col4.metric("I/N Threshold", f"{b['in_threshold_db']} dB",
        help="Interference-to-Noise (I/N) ratio threshold — the maximum tolerable interference power relative to the receiver noise floor. Exceeding this threshold is the basis for citing harmful interference under Radio Regulations No. 4.10.")

    col5, col6, col7 = st.columns(3)
    col5.metric("Allocation", b["allocation"],
        help="ITU Radio Regulations allocation type. ARNS = Aeronautical Radionavigation Service. RNSS = Radionavigation Satellite Service. ANS = Aeronautical Navigation Service. Primary allocation has stronger protection than secondary.")
    col6.metric("Noise Floor (est.)", f"{b['noise_floor_dbm']} dBm",
        help="Estimated receiver thermal noise floor (kTBF). Verify exact value against applicable RTCA Minimum Operational Performance Standard (MOPS) before citing in a contribution.")
    col7.metric("RTCA Standard", b["rtca_doc"],
        help="Radio Technical Commission for Aeronautics (RTCA) Minimum Operational Performance Standard governing this receiver type. Primary source for protection criteria cited in ITU-R contributions.")

    # New fields from slides
    col8, col9, col10 = st.columns(3)
    safety_factor = b.get("aviation_safety_factor_db", 0)
    effective_thresh = b.get("effective_threshold_db", b["in_threshold_db"])
    col8.metric("Aviation Safety Factor", f"{safety_factor} dB" if safety_factor > 0 else "0 dB (N/A)",
        help="Per FAA slides: 'Some aeronautical applications (e.g. precision approach and landing) are critical, meriting an additional safety factor of not less than 6 dB.' Applies on top of the base I/N threshold.")
    col9.metric("Effective Threshold (incl. safety factor)", f"{effective_thresh} dB",
        help="Base I/N threshold plus the aviation safety factor. This is the operationally applied protection criterion for safety-critical precision approach and landing applications.")
    col10.metric("Service Category (ITU-R)", b.get("service_category", b["allocation"]),
        help="ITU service category. AM(R)S = Aeronautical Mobile (Route) Service — safety and regularity. AMS(R)S = satellite variant. ARNS = Aeronautical Radionavigation Service.")

    st.markdown(f"**System:** {b['system']}")
    st.markdown(f"**Notes:** {b['notes']}")
    st.markdown(f"**Protection Basis:** {b.get('protection_basis', '—')}")

    # Special protection values (ΔT/T, PFD, epfd)
    special_vals = []
    if b.get("delta_t_t_pct_aggregate"):
        special_vals.append(f"ΔT/T aggregate: {b['delta_t_t_pct_aggregate']}%")
    if b.get("delta_t_t_pct_single"):
        special_vals.append(f"ΔT/T single-entry: {b['delta_t_t_pct_single']}%")
    if b.get("psd_threshold_dbw_mhz"):
        special_vals.append(f"PSD threshold: {b['psd_threshold_dbw_mhz']} dBW/MHz")
    if b.get("epfd_threshold_dbw_m2_mhz"):
        special_vals.append(f"epfd limit: {b['epfd_threshold_dbw_m2_mhz']} dBW/m²/MHz")
    if b.get("pfd_threshold_dbw_m2_khz"):
        special_vals.append(f"pfd limit: {b['pfd_threshold_dbw_m2_khz']} dBW/m² in 150 kHz")
    if special_vals:
        st.markdown("**Additional Protection Criteria:** " + "  |  ".join(special_vals))

    if b.get("rr_1_59"):
        st.markdown("🔴 **Safety Service (RR 1.59)** — any radiocommunication service used permanently or temporarily for the safeguarding of human life and property. Protected under RR No. 4.10.")

    # Source-Path-Receiver model
    with st.expander("🔍 Source-Path-Receiver (SPR) Analysis Framework for this band"):
        ex("The SPR model is the formal FAA methodology for aviation interference analysis (per FAA slides). Worst-case limits must be applied on all three elements.")
        spr1, spr2, spr3 = st.columns(3)
        with spr1:
            st.markdown(f"""
<div style='background:#2a1a0a;border-left:4px solid #ff8844;padding:10px;border-radius:4px'>
<b style='color:#ff8844'>📡 SOURCE (Interferer)</b><br>
<span style='color:#ffddaa;font-size:0.85em'>{b.get('spr_source','Max Tx power; worst-case antenna gain; signal characteristics')}</span>
</div>""", unsafe_allow_html=True)
        with spr2:
            st.markdown(f"""
<div style='background:#0a0a2a;border-left:4px solid #4488ff;padding:10px;border-radius:4px'>
<b style='color:#4488ff'>🌐 PATH (Propagation)</b><br>
<span style='color:#aaddff;font-size:0.85em'>{b.get('spr_path','FSPL worst-case; free-space attenuation especially >1 GHz; distance separation >20 km')}</span>
</div>""", unsafe_allow_html=True)
        with spr3:
            st.markdown(f"""
<div style='background:#0a2a0a;border-left:4px solid #44bb44;padding:10px;border-radius:4px'>
<b style='color:#44bb44'>📻 VICTIM (Receiver)</b><br>
<span style='color:#aaffaa;font-size:0.85em'>{b.get('spr_victim','Receiver susceptibility mask; antenna gain; receiver noise power')}</span>
</div>""", unsafe_allow_html=True)
        st.markdown("")
        st.caption("Per FAA guidance: use worst-case limits on all aspects. Aggregate effect of multiple interference sources must be considered. Max interference threshold is used as protection criteria taking into account all environmental conditions.")

    # Full table — updated with new columns
    st.subheader("All Bands Summary Table")
    ex("Now includes aviation safety factor and effective threshold per FAA system protection levels. Noise floor estimates are representative — verify against applicable RTCA MOPS.")
    rows = []
    for name, b_row in FAA_BANDS.items():
        rows.append({
            "Band": name,
            "Low (MHz)": b_row["f_low_mhz"],
            "High (MHz)": b_row["f_high_mhz"],
            "BW (MHz)": round(b_row["f_high_mhz"] - b_row["f_low_mhz"], 3),
            "Allocation": b_row["allocation"],
            "I/N Threshold (dB)": b_row["in_threshold_db"],
            "Aviation Safety Factor (dB)": b_row.get("aviation_safety_factor_db", 0),
            "Effective Threshold (dB)": b_row.get("effective_threshold_db", b_row["in_threshold_db"]),
            "Noise Floor (dBm)": b_row["noise_floor_dbm"],
            "RR 1.59 Safety Service": "✅" if b_row.get("rr_1_59") else "—",
            "RTCA": b_row["rtca_doc"],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — LINK BUDGET
# ─────────────────────────────────────────────────────────────────────────────
elif selected_tab == "🔗 Link Budget":
    st.title("🔗 Link Budget Calculator")
    ex("The Friis transmission equation in log form: Pr = Pt + Gt − Lcable − FSPL + Gr. Every term must use consistent reference points — EIRP subsumes Pt + Gt − Lcable into a single radiated power figure.")

    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.subheader("Transmitter (Interferer)")
        ex("For a worst-case bound, use the maximum authorized EIRP from the relevant ITU-R Recommendation or WRC agenda item — proponents often use median or typical values, which is a common optimism to challenge.")
        tx_power_dbm = st.number_input("Tx Power (dBm)", value=43.0, step=1.0,
            help="e.g., 43 dBm = 20W typical LTE base station")
        tx_gain_dbi = st.number_input("Tx Antenna Gain (dBi)", value=15.0, step=0.5,
            help="Directional gain toward victim receiver")
        cable_loss = st.number_input("Cable / Feeder Loss (dB)", value=2.0, step=0.5,
            help="Losses between PA output and antenna port")
        tx_height_m = st.number_input("Tx Height (m AGL)", value=30.0, step=5.0)

        st.subheader("Channel")
        ex("FSPL assumes isotropic radiation into free space with no terrain, atmosphere, or clutter — it is the most optimistic model (least loss) and therefore represents the worst-case interference scenario.")
        propagation_model = st.selectbox("Propagation Model",
            ["Free Space (FSPL)", "P.452 (Terrestrial)", "P.528 (Aeronautical)"])
        freq_mhz = st.number_input("Frequency (MHz)", value=4300.0, step=10.0,
            help="Center frequency of the interfering emission")
        dist_km = st.number_input("Distance (km)", value=5.0, step=0.5)

        if propagation_model == "P.452 (Terrestrial)":
            terrain_type = st.selectbox("Terrain / Clutter Type",
                ["open", "suburban", "urban", "dense_urban"])
            rx_height_m = st.number_input("Rx Height (m AGL)", value=5.0, step=1.0)
        elif propagation_model == "P.528 (Aeronautical)":
            aircraft_alt_km = st.number_input("Aircraft Altitude (km)", value=3.0, step=0.5)
            rx_height_m = 5.0
            terrain_type = "suburban"
        else:
            rx_height_m = 5.0
            terrain_type = "suburban"
            aircraft_alt_km = 0.0

    with col_r:
        st.subheader("Receiver (Victim)")
        ex("Use 0 dBi receive antenna gain for worst-case (isotropic toward the interferer) unless you have a validated antenna pattern model — assumed directivity away from the interferer is a common proponent tactic to challenge.")
        rx_gain_dbi = st.number_input("Rx Antenna Gain (dBi)", value=0.0, step=0.5,
            help="Toward interferer; 0 dBi = worst case omnidirectional")

        st.markdown("**— or select a protected FAA band —**")
        band_select = st.selectbox("Auto-fill from FAA band:", ["(manual)"] + list(FAA_BANDS.keys()))
        if band_select != "(manual)":
            b = FAA_BANDS[band_select]
            rx_noise_floor_dbm_input = b["noise_floor_dbm"]
            in_threshold_db = b["in_threshold_db"]
            st.info(f"Loaded: Noise floor = {rx_noise_floor_dbm_input} dBm | I/N threshold = {in_threshold_db} dB")
        else:
            rx_bw_mhz = st.number_input("Rx Bandwidth (MHz)", value=100.0, step=10.0)
            rx_nf_db = st.number_input("Receiver Noise Figure (dB)", value=5.0, step=0.5)
            rx_noise_floor_dbm_input = noise_floor_dbm(rx_bw_mhz * 1e6, rx_nf_db)
            in_threshold_db = st.number_input("I/N Threshold (dB)", value=-6.0, step=1.0,
                help="Protection criterion: -6 dB typical aeronautical, -10 dB for GNSS")

    # Compute
    st.markdown("---")
    if st.button("⚡ Run Link Budget", type="primary"):
        eir = eirp_dbm(tx_power_dbm, tx_gain_dbi)

        if propagation_model == "Free Space (FSPL)":
            pl = free_space_path_loss_db(freq_mhz, dist_km)
            model_label = "FSPL"
        elif propagation_model == "P.452 (Terrestrial)":
            pl = p452_basic_loss_db(freq_mhz, dist_km, tx_height_m, rx_height_m, terrain_type)
            model_label = f"P.452 ({terrain_type})"
        else:
            ac_alt = aircraft_alt_km if 'aircraft_alt_km' in dir() else 3.0
            pl = p528_aero_path_loss_db(freq_mhz, dist_km, ac_alt)
            model_label = f"P.528 (alt={ac_alt} km)"

        rx_pwr = received_power_dbm(tx_power_dbm, tx_gain_dbi, pl, rx_gain_dbi, cable_loss)
        i_n = in_ratio_db(rx_pwr, rx_noise_floor_dbm_input)
        margin = protection_margin_db(i_n, in_threshold_db)

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("EIRP", f"{eir:.1f} dBm")
        col2.metric("Path Loss", f"{pl:.1f} dB", help=model_label)
        col3.metric("Rx Power", f"{rx_pwr:.1f} dBm")
        col4.metric("I/N", f"{i_n:.1f} dB")
        col5.metric("Protection Margin", f"{margin:.1f} dB",
                    delta=f"{margin:.1f} dB",
                    delta_color="normal" if margin >= 0 else "inverse")

        ex("Protection Margin = threshold − computed I/N. A negative margin means the aggregate noise floor rise exceeds the ITU-R criterion and you have grounds to cite harmful interference under RR No. 4.10.")

        if margin >= 10:
            ok(f"PROTECTED with {margin:.1f} dB margin — system is well-protected at this distance.")
        elif margin >= 0:
            warn(f"MARGINALLY PROTECTED with only {margin:.1f} dB margin — consider conservative assumptions.")
        else:
            warn(f"THRESHOLD VIOLATED by {abs(margin):.1f} dB — this scenario poses an interference risk to the protected service.")

        # Waterfall chart
        st.subheader("Link Budget Waterfall")
        ex("The waterfall visualization follows the Friis equation left to right: Pt → EIRP (add Gt, subtract cable) → Pr (subtract path loss, add Gr). The gap between Pr and the noise floor reference is your link margin.")
        stages = ["Tx Power", "Tx Gain", "Cable Loss", "Path Loss", "Rx Gain", "Rx Power"]
        values = [tx_power_dbm, tx_gain_dbi, -cable_loss, -pl, rx_gain_dbi, None]

        running = tx_power_dbm
        bar_bottoms, bar_heights, bar_colors = [], [], []
        for i, v in enumerate(values[1:-1], 1):
            if v >= 0:
                bar_bottoms.append(running)
                bar_heights.append(v)
                bar_colors.append("#44bb44")
            else:
                bar_bottoms.append(running + v)
                bar_heights.append(-v)
                bar_colors.append("#bb4444")
            running += v

        fig2, ax2 = plt.subplots(figsize=(10, 4))
        fig2.patch.set_facecolor("#0e1117")
        ax2.set_facecolor("#0e1117")

        # Tx power base bar
        ax2.bar(0, tx_power_dbm, color="#4488ff", label="Tx Power")
        for i, (b_val, h_val, c) in enumerate(zip(bar_bottoms, bar_heights, bar_colors)):
            ax2.bar(i + 1, h_val, bottom=b_val, color=c)

        # Final Rx power
        ax2.bar(len(stages) - 1, rx_pwr, color="#ffaa00")
        ax2.axhline(rx_noise_floor_dbm_input, color='cyan', linestyle='--', linewidth=1.5,
                    label=f"Noise Floor ({rx_noise_floor_dbm_input:.0f} dBm)")
        ax2.axhline(rx_noise_floor_dbm_input + in_threshold_db, color='red',
                    linestyle=':', linewidth=1.5,
                    label=f"I/N Threshold ({in_threshold_db} dB above noise)")

        ax2.set_xticks(range(len(stages)))
        ax2.set_xticklabels(stages, color='white', fontsize=9)
        ax2.set_ylabel("Power (dBm)", color='white')
        ax2.tick_params(colors='white')
        ax2.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white')
        for sp in ax2.spines.values():
            sp.set_color('#444')
        plt.tight_layout()
        st.pyplot(fig2)

        # Distance sweep
        st.subheader("Path Loss vs. Distance")
        ex("Required path loss = EIRP − noise floor − I/N threshold. This value on the FSPL curve gives the minimum separation distance — cite this in contributions as a coordination zone requirement.")
        dists = np.linspace(0.1, max(50, dist_km * 3), 200)
        pls_fspl = [free_space_path_loss_db(freq_mhz, d) for d in dists]
        pls_p452 = [p452_basic_loss_db(freq_mhz, d, tx_height_m, rx_height_m, terrain_type) for d in dists]

        fig3, ax3 = plt.subplots(figsize=(10, 4))
        fig3.patch.set_facecolor("#0e1117")
        ax3.set_facecolor("#0e1117")
        ax3.plot(dists, pls_fspl, color='#4488ff', label='FSPL (best case / worst interference)')
        ax3.plot(dists, pls_p452, color='#ffaa00', linestyle='--', label='P.452 simplified')

        # Required path loss for protection
        req_pl = eir - cable_loss + rx_gain_dbi - rx_noise_floor_dbm_input - in_threshold_db
        ax3.axhline(req_pl, color='red', linestyle=':', label=f"Required PL for I/N≤{in_threshold_db} dB ({req_pl:.0f} dB)")
        ax3.axvline(dist_km, color='white', linestyle=':', alpha=0.5, label=f"Current dist ({dist_km} km)")
        ax3.set_xlabel("Distance (km)", color='white')
        ax3.set_ylabel("Path Loss (dB)", color='white')
        ax3.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white')
        ax3.tick_params(colors='white')
        for sp in ax3.spines.values():
            sp.set_color('#444')
        plt.tight_layout()
        st.pyplot(fig3)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — NOISE & I/N ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
elif selected_tab == "📊 Noise & I/N":
    st.title("📊 Noise Floor & I/N Analysis")
    ex("I/N quantifies the degradation of receiver sensitivity: total noise = N·(1 + I/N_linear). At I/N = −6 dB, sensitivity degrades by 0.97 dB; at 0 dB it degrades by 3 dB — unacceptable for safety-critical systems.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Receiver Noise Floor Calculator")
        ex("Thermal noise floor: N = kTB·NF, where k = 1.38×10⁻²³ J/K, T = 290K standard, B = noise bandwidth. The NF term captures all internal noise contributions referred to the input — any interference power above kTB·NF degrades SNR.")
        bw_mhz = st.number_input("Receiver Bandwidth (MHz)", value=100.0, min_value=0.001, step=10.0)
        nf_db = st.number_input("Receiver Noise Figure (dB)", value=5.0, step=0.5,
            help="Measure of receiver's internal noise amplification (0 dB = ideal)")
        temp_k = st.number_input("Noise Temperature (K)", value=290.0, step=10.0,
            help="Physical temperature; 290K is standard; use 255K for avionics")

        kT = 10.0 * np.log10(1.38e-23 * temp_k) + 30  # in dBm/Hz
        nf_dbm = kT + 10.0 * np.log10(bw_mhz * 1e6) + nf_db

        st.markdown("---")
        st.metric("Thermal Noise Density", f"{kT:.1f} dBm/Hz")
        st.metric("Noise Floor", f"{nf_dbm:.1f} dBm")
        ex("kT = −174 dBm/Hz at 290K is the fundamental thermal noise limit. Each decade of bandwidth adds 10 dB to the noise floor. NF is the receiver's contribution above kTB — a 1 dB NF increase directly reduces sensitivity by 1 dB.")

    with col2:
        st.subheader("I/N Threshold Selector")
        ex("GNSS threshold is −10 dB I/N per M.1477/M.1905 because GPS L1 arrives at approximately −130 dBm — only 25 dB above the noise floor of a typical receiver. Any noise floor rise materially degrades acquisition and tracking margin.")
        threshold_type = st.selectbox("System Type", [
            "Generic aeronautical ARNS (−6 dB)",
            "GNSS / GPS (−10 dB)",
            "ADS-B / SSR (−10 dB)",
            "Radio Altimeter (−6 dB)",
            "Custom",
        ])
        thresh_map = {
            "Generic aeronautical ARNS (−6 dB)": -6,
            "GNSS / GPS (−10 dB)": -10,
            "ADS-B / SSR (−10 dB)": -10,
            "Radio Altimeter (−6 dB)": -6,
        }
        if threshold_type == "Custom":
            threshold_db = st.number_input("Custom I/N Threshold (dB)", value=-6.0)
        else:
            threshold_db = thresh_map[threshold_type]
            st.metric("I/N Threshold", f"{threshold_db} dB")

        interference_dbm = st.number_input("Interference Power at Rx (dBm)", value=-100.0, step=1.0)
        i_n_calc = in_ratio_db(interference_dbm, nf_dbm)
        margin_calc = protection_margin_db(i_n_calc, threshold_db)

        st.metric("Computed I/N", f"{i_n_calc:.1f} dB")
        st.metric("Protection Margin", f"{margin_calc:.1f} dB",
                  delta=f"{margin_calc:.1f} dB",
                  delta_color="normal" if margin_calc >= 0 else "inverse")

        if margin_calc < 0:
            warn(f"Threshold violated by {abs(margin_calc):.1f} dB — this would be cited as harmful interference in a US contribution.")
        else:
            ok(f"Protected with {margin_calc:.1f} dB margin.")

    # I/N sensitivity heatmap
    st.subheader("I/N Sensitivity — Interference Power vs Bandwidth")
    ex("The contour line at your I/N threshold shows the boundary between compatible and incompatible operation. Note how wider bandwidth moves the boundary upward — a broader receiver requires proportionally stronger interference to violate the threshold.")
    bw_range = np.logspace(np.log10(0.1), np.log10(500), 40)
    int_range = np.linspace(-140, -60, 40)
    Z = np.zeros((len(int_range), len(bw_range)))
    for i, intpwr in enumerate(int_range):
        for j, bw in enumerate(bw_range):
            nf_j = kT + 10.0 * np.log10(bw * 1e6) + nf_db
            Z[i, j] = in_ratio_db(intpwr, nf_j)

    fig4, ax4 = plt.subplots(figsize=(10, 5))
    fig4.patch.set_facecolor("#0e1117")
    im = ax4.contourf(bw_range, int_range, Z, levels=20, cmap='RdYlGn_r')
    ax4.contour(bw_range, int_range, Z, levels=[threshold_db], colors='red', linewidths=2)
    ax4.set_xscale('log')
    ax4.set_xlabel("Receiver Bandwidth (MHz)", color='white')
    ax4.set_ylabel("Interference Power at Rx (dBm)", color='white')
    ax4.set_title(f"I/N (dB) — Red line = {threshold_db} dB threshold", color='white')
    ax4.tick_params(colors='white')
    for sp in ax4.spines.values():
        sp.set_color('#444')
    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label("I/N (dB)", color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
    plt.tight_layout()
    st.pyplot(fig4)

    st.markdown("---")
    # ── GNSS PROTECTION FRAMEWORK (M.1318 / M.1477 / M.1904 / M.1905) ────────
    st.header("🛰️ GNSS Protection Framework — ITU-R M.1318 / M.1477 / M.1904 / M.1905")
    ex("The M.1318/M.1477 framework is the legally grounded calculation chain for GNSS protection: receiver spec (a) minus safety margin (b) gives the maximum tolerable interference density (c) — all in dB(W/Hz) at the passive antenna terminal.")

    st.subheader("1️⃣  The c = a − b Framework (ITU-R M.1318 Annex 1)")
    ex("Note that 'a' is specified at the passive antenna terminal — not at the LNA output. This is important when comparing with interference analyses that compute received power: convert to power density using the receiver noise bandwidth.")
    st.latex(r"c = a - b")
    st.markdown("""
| Symbol | Parameter | Units | Description |
|---|---|---|---|
| **a** | Max aggregate non-RNSS interference power density | dB(W/Hz) | Specified by the RNSS receiver design — from receiver datasheet or ITU-R recommendation |
| **b** | Protection margin | dB | To ensure protection as provided by RR No. 4.10 — **6 dB aeronautical safety margin (M.1477)** |
| **c** | Tolerable interference power density at receiver | dB(W/Hz) | The level you must stay below in your interference analysis |
    """)

    st.subheader("🔢 c = a − b Calculator")
    ex("'a' comes from the RNSS receiver's minimum operational performance specification (MOPS) — typically −111.5 dB(W/Hz) for GPS/GNSS. This is not a system-level threshold; it is the worst-case receiver design point.")

    col_gnss1, col_gnss2 = st.columns(2)
    with col_gnss1:
        gnss_band = st.selectbox("GNSS Band (ITU-R M.1318):", [
            "L5 / E5 — 1164–1215 MHz",
            "L2 / E6 — 1215–1300 MHz",
            "L1 / E1 — 1559–1610 MHz",
            "L3 / future — 5010–5030 MHz",
            "Custom",
        ])
        # Pre-fill typical 'a' values per band from ITU-R M.1318
        a_defaults = {
            "L5 / E5 — 1164–1215 MHz": -111.5,
            "L2 / E6 — 1215–1300 MHz": -111.5,
            "L1 / E1 — 1559–1610 MHz": -111.5,
            "L3 / future — 5010–5030 MHz": -111.5,
            "Custom": -111.5,
        }
        param_a = st.number_input(
            "a — Max aggregate non-RNSS interference power density (dB(W/Hz))",
            value=a_defaults.get(gnss_band, -111.5), step=0.5,
            help="From ITU-R M.1318 Table / receiver spec. Typical GPS L1: −111.5 dB(W/Hz)"
        )
        interferer_bw_hz = st.number_input(
            "Interferer bandwidth (Hz) — for narrowband rule check",
            value=10e6, min_value=1.0, step=1000.0, format="%.0f",
            help="Enter the bandwidth of the interfering signal in Hz"
        )

    with col_gnss2:
        # Protection margin b — M.1477 defines 6 dB aeronautical safety margin
        margin_type = st.selectbox("b — Protection Margin Source:", [
            "M.1477 Annex 5 — Aeronautical safety margin (6 dB)",
            "M.1904 / M.1905 — GLONASS/RNSS safety margin (6 dB)",
            "M.1477 Annex 5 — Narrowband interferer ≤700 Hz (+10 dB additional)",
            "Custom margin",
        ])
        margin_map = {
            "M.1477 Annex 5 — Aeronautical safety margin (6 dB)": 6.0,
            "M.1904 / M.1905 — GLONASS/RNSS safety margin (6 dB)": 6.0,
            "M.1477 Annex 5 — Narrowband interferer ≤700 Hz (+10 dB additional)": 16.0,
            "Custom margin": None,
        }
        if margin_map[margin_type] is None:
            param_b = st.number_input("b — Custom protection margin (dB)", value=6.0, step=0.5)
        else:
            param_b = margin_map[margin_type]
            st.metric("b — Protection Margin", f"{param_b} dB")

        param_c = param_a - param_b
        st.metric("c — Tolerable interference level", f"{param_c:.1f} dB(W/Hz)")
        ex("c is defined at the passive antenna terminal, integrated over the noise bandwidth. If your analysis produces received interference power P_i (dBm), convert: c_computed = P_i(dBm) − 30 − 10·log10(B_Hz), then compare to c.")

    # Narrowband rule check — M.1477 Annex 5
    st.subheader("2️⃣  Narrowband Interferer Rule (ITU-R M.1477 Annex 5)")
    ex("The ≤700 Hz narrowband rule applies because a CW or very narrowband interferer concentrates all its energy into the receiver's phase-locked tracking loops, causing cycle slips and position errors disproportionate to its total power.")

    nb_threshold_hz = 700.0
    is_narrowband = interferer_bw_hz <= nb_threshold_hz
    col_nb1, col_nb2 = st.columns(2)
    with col_nb1:
        st.metric("Interferer Bandwidth", f"{interferer_bw_hz:.0f} Hz")
        st.metric("Narrowband Threshold", "700 Hz")
    with col_nb2:
        if is_narrowband:
            warn(f"⚠️ NARROWBAND RULE APPLIES — interferer BW ({interferer_bw_hz:.0f} Hz) ≤ 700 Hz. "
                 f"Additional +10 dB protection margin required per M.1477 Annex 5. "
                 f"Effective I/N threshold tightens to −20 dB (−10 dB standard + −10 dB additional).")
            effective_gnss_threshold = -20.0
        else:
            ok(f"Narrowband rule does NOT apply — interferer BW ({interferer_bw_hz:.0f} Hz) > 700 Hz. "
               f"Standard −10 dB I/N threshold applies.")
            effective_gnss_threshold = -10.0
        st.metric("Effective GNSS I/N Threshold", f"{effective_gnss_threshold} dB")

    # Full GNSS protection criteria table
    st.subheader("3️⃣  Complete GNSS Protection Criteria by Band")
    ex("RR No. 5.328 limits RNSS protection in 1164–1215 MHz: RNSS cannot claim protection from ARNS (DME beacons) per M.1318 Annex 1. Your interference analysis must account for DME as a pre-existing interference source in this band.")
    gnss_table = pd.DataFrame([
        ["L1 / E1 / B1", "1559–1610 MHz", "−111.5", "6 dB", "−10 dB", "−20 dB*",
         "M.1318, M.1477, M.1905", "Primary aviation GNSS; WAAS/SBAS; most protected"],
        ["L5 / E5a / B2a", "1164–1215 MHz", "−111.5", "6 dB", "−10 dB", "−20 dB*",
         "M.1318, M.1477, M.1904", "Safety-of-life signal; aviation approach procedures"],
        ["L2 / E6 / B3", "1215–1300 MHz", "−111.5", "6 dB", "−10 dB", "−20 dB*",
         "M.1318, M.1477", "Dual-freq integrity; shares with ARNS (RR 5.328 applies)"],
        ["L3 / future", "5010–5030 MHz", "−111.5", "6 dB", "−10 dB", "−20 dB*",
         "M.1318", "Under pressure from IMT-2030 studies in WP 5D"],
    ], columns=[
        "Signal", "Band (MHz)", "a (dB(W/Hz))", "b (Aero Margin)",
        "Std I/N Threshold", "Narrowband I/N*", "Key References", "Notes"
    ])
    st.dataframe(gnss_table, use_container_width=True)
    st.caption("* Narrowband threshold applies when interferer BW ≤ 700 Hz per ITU-R M.1477 Annex 5")

    # Recommendation summary box
    st.subheader("4️⃣  Recommendation Doctrine Summary")
    ex("M.1904 and M.1905 are critical because they establish the 6 dB safety margin as settled ITU-R doctrine — any proponent proposing less must argue against two explicit Recommendations, not just a methodology document.")
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.markdown("""
**ITU-R M.1318** — *Methodology for GNSS protection*
- Defines the c = a − b framework
- Covers L1, L2, L5, L3 bands
- Establishes aggregate interference methodology
- **Cite for:** methodology basis of any GNSS interference analysis

**ITU-R M.1477 Annex 5** — *Aeronautical safety margin*
- 6 dB safety margin for GNSS safety-of-life apps
- +10 dB additional margin for narrowband (≤700 Hz) interferers
- Lists factors requiring additional margins: terrain, weather, RNSS config
- **Cite for:** justifying b = 6 dB in the c = a − b calculation
        """)
    with col_r2:
        st.markdown("""
**ITU-R M.1904** — *GLONASS spaceborne receiver protection*
- Annex 1 Table 1 Note 3: safety margin = 6 dB
- Establishes GLONASS-specific protection baseline
- **Cite for:** GLONASS L1/L2 band contributions; reinforces 6 dB doctrine

**ITU-R M.1905** — *RNSS safety applications*
- Recommends safety margin be applied for all RNSS safety-of-life interference analyses
- Note 1: Aeronautical safety margin of 6 dB
- **Cite for:** Any contribution involving aviation use of GNSS — broadest coverage
- **Key strength:** Applies to ALL RNSS systems, not just GPS or GLONASS

---
**Together:** M.1318 (method) + M.1477 (margin) + M.1904 + M.1905 (doctrine) = complete GNSS defense stack
        """)

    # Interference budget worked example
    st.subheader("5️⃣  Worked Example — GPS L1 Protection Budget")
    ex("The M.1318 budget uses power spectral density (dB(W/Hz)) rather than integrated power (dBm) — this makes the criterion independent of receiver bandwidth and directly comparable across systems with different noise bandwidths.")
    with st.expander("📐 Show GPS L1 Protection Budget Walkthrough"):
        st.markdown("""
**Scenario:** New terrestrial service proposed near 1559–1610 MHz GPS L1 band.
Is the aggregate interference tolerable?

**Step 1 — Establish 'a' from receiver spec (M.1318)**
> Maximum aggregate non-RNSS interference power density for GPS L1:
> **a = −111.5 dB(W/Hz)**
> (From ITU-R M.1318 Annex 1 Table 1)

**Step 2 — Apply aeronautical safety margin 'b' (M.1477 Annex 5 + M.1905)**
> Aeronautical safety margin = **6 dB** (M.1477 Note 1, M.1905 Recommendation 2)
> This ensures protection as required by RR No. 4.10 (M.1318 Annex 1, Step 1b)
> **b = 6 dB**

**Step 3 — Compute tolerable interference level 'c'**
> c = a − b = −111.5 − 6 = **−117.5 dB(W/Hz)**

**Step 4 — Check narrowband rule (M.1477 Annex 5, Section 4)**
> If interfering signal BW ≤ 700 Hz: apply additional 10 dB margin
> Effective tolerable level = −117.5 − 10 = **−127.5 dB(W/Hz)** for narrowband

**Step 5 — Compare your computed interference to 'c'**
> Run your link budget → convert received interference power to power density (dB(W/Hz))
> If computed interference > c → threshold exceeded → **cite harmful interference under RR 4.10**
> If computed interference ≤ c → compatible → **document margin and conditions**

**Step 6 — State the regulatory consequence**
> "The computed aggregate non-RNSS interference power density of [X] dB(W/Hz) exceeds
> the tolerable level c = −117.5 dB(W/Hz) established by ITU-R M.1318 Annex 1, with
> the 6 dB aeronautical safety margin required by ITU-R M.1477 Annex 5, M.1904, and M.1905.
> This constitutes harmful interference to a safety-of-life service under RR No. 4.10.
> The United States opposes this proposal without additional protective measures."
        """)

    # Power density converter
    st.subheader("🔢 Power → Power Density Converter")
    ex("Power density = received power − 10·log10(noise BW). GPS L1 C/A noise BW ≈ 2 MHz; L5 ≈ 20.46 MHz. Using too wide a bandwidth here artificially lowers the density and makes interference appear more benign — check the proponent's assumed bandwidth.")
    col_pd1, col_pd2 = st.columns(2)
    with col_pd1:
        pwr_dbm_conv = st.number_input("Interference power at Rx (dBm)", value=-100.0, step=1.0)
        rx_bw_hz_conv = st.number_input("Receiver noise bandwidth (Hz)", value=20.46e6,
            step=1e6, format="%.0f",
            help="GPS L1 C/A: ~1 MHz; L5: ~20.46 MHz; use receiver spec")
    pwr_dbw = pwr_dbm_conv - 30
    pwr_density_dbwHz = pwr_dbw - 10*np.log10(rx_bw_hz_conv)
    with col_pd2:
        st.metric("Power density", f"{pwr_density_dbwHz:.1f} dB(W/Hz)")
        margin_vs_c = param_c - pwr_density_dbwHz
        st.metric("Margin vs tolerable level c", f"{margin_vs_c:.1f} dB",
                  delta_color="normal" if margin_vs_c >= 0 else "inverse")
        if margin_vs_c < 0:
            warn(f"Computed interference exceeds c by {abs(margin_vs_c):.1f} dB — cite M.1318 violation in contribution.")
        else:
            ok(f"Compatible with {margin_vs_c:.1f} dB margin against M.1318 tolerable level.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — PROPAGATION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
elif selected_tab == "🌐 Propagation":
    st.title("🌐 Propagation Analysis")
    ex("P.528 accounts for atmospheric refraction, radio horizon geometry, and troposcatter — all of which increase path loss relative to FSPL for slant paths. A proponent using FSPL for an airborne scenario is using an unrealistically pessimistic interference estimate.")

    freq_mhz_p = st.slider("Frequency (MHz)", min_value=100, max_value=10000, value=4300, step=50)
    max_dist_km = st.slider("Max Distance (km)", min_value=5, max_value=500, value=100)

    col_a, col_b = st.columns(2)
    with col_a:
        aircraft_alt_km_p = st.number_input("Aircraft Altitude (km) — for P.528", value=3.0, step=0.5)
        tx_h = st.number_input("Tx Height m (P.452)", value=30.0)
        rx_h = st.number_input("Rx Height m (P.452)", value=5.0)
    with col_b:
        terrain_p = st.selectbox("Terrain (P.452)", ["open", "suburban", "urban", "dense_urban"])
        time_pct = st.slider("Time % for P.528", min_value=1, max_value=99, value=50,
            help="50% = median; 1% = exceeded only 1% of the time (worst interference case)")

    dists_p = np.linspace(0.5, max_dist_km, 300)
    fspl_vals = [free_space_path_loss_db(freq_mhz_p, d) for d in dists_p]
    p452_vals = [p452_basic_loss_db(freq_mhz_p, d, tx_h, rx_h, terrain_p) for d in dists_p]
    p528_vals = [p528_aero_path_loss_db(freq_mhz_p, d, aircraft_alt_km_p, time_pct) for d in dists_p]

    fig5, ax5 = plt.subplots(figsize=(12, 5))
    fig5.patch.set_facecolor("#0e1117")
    ax5.set_facecolor("#0e1117")
    ax5.plot(dists_p, fspl_vals, color='#4488ff', linewidth=2, label='FSPL (optimistic — worst interference case)')
    ax5.plot(dists_p, p452_vals, color='#ffaa00', linewidth=2, linestyle='--',
             label=f'P.452 simplified ({terrain_p})')
    ax5.plot(dists_p, p528_vals, color='#44ffaa', linewidth=2, linestyle='-.',
             label=f'P.528 simplified (alt={aircraft_alt_km_p} km, {time_pct}%)')
    ax5.set_xlabel("Distance (km)", color='white', fontsize=11)
    ax5.set_ylabel("Basic Transmission Loss (dB)", color='white', fontsize=11)
    ax5.set_title(f"Path Loss Comparison at {freq_mhz_p} MHz", color='white')
    ax5.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=9)
    ax5.tick_params(colors='white')
    ax5.grid(color='#333', alpha=0.5)
    for sp in ax5.spines.values():
        sp.set_color('#444')
    plt.tight_layout()
    st.pyplot(fig5)

    ex("Conservative protection analysis uses the lowest plausible path loss (FSPL or P.452 open, time %=1) to maximize computed interference. If compatibility holds under these conditions, any realistic deployment will also be compatible.")

    # Atmospheric attenuation using itur P.676
    st.subheader("Atmospheric Gaseous Attenuation (ITU-R P.676 via itur)")
    ex("P.676 gaseous attenuation is negligible below ~3 GHz (<0.01 dB/km), becomes relevant above 6 GHz, and peaks at 60 GHz (O₂ resonance, ~15 dB/km). Do not cite it as a protection mechanism for L-band or C-band ground-to-air paths.")

    freq_range_ghz = np.linspace(0.1, min(freq_mhz_p / 1000 * 2, 10), 100)
    gas_atten_vals = []
    path_len_km = st.slider("Path length for P.676 (km)", 1, 100, 20)

    for f_ghz in freq_range_ghz:
        try:
            g = itu676.gaseous_attenuation_terrestrial_path(
                f_ghz, p=50, T=15, H=50, P=1013.25, d=path_len_km, mode="approx"
            )
            gas_atten_vals.append(float(g.value) if hasattr(g, 'value') else float(g))
        except Exception:
            gas_atten_vals.append(np.nan)

    fig6, ax6 = plt.subplots(figsize=(12, 4))
    fig6.patch.set_facecolor("#0e1117")
    ax6.set_facecolor("#0e1117")
    ax6.plot(freq_range_ghz * 1000, gas_atten_vals, color='#ff88aa', linewidth=2)
    ax6.axvline(freq_mhz_p, color='yellow', linestyle='--', alpha=0.7, label=f'Selected: {freq_mhz_p} MHz')
    ax6.set_xlabel("Frequency (MHz)", color='white')
    ax6.set_ylabel(f"Gaseous Attenuation (dB) over {path_len_km} km", color='white')
    ax6.set_title("ITU-R P.676 Atmospheric Attenuation (via itur)", color='white')
    ax6.legend(facecolor='#1a1a2e', labelcolor='white')
    ax6.tick_params(colors='white')
    ax6.grid(color='#333', alpha=0.5)
    for sp in ax6.spines.values():
        sp.set_color('#444')
    plt.tight_layout()
    st.pyplot(fig6)

    # Model selector guidance
    st.subheader("Which Model Should You Use?")
    ex("Model selection is a regulatory choice, not just a technical one — ITU-R contributions cite the specific model by name. A challenge to your model choice in a Working Party session can invalidate your entire analysis, so document your selection rationale explicitly.")
    guidance = pd.DataFrame([
        ["Terrestrial base station → ground receiver", "P.452", "Point-to-point interference, terrain profile needed for full implementation"],
        ["Terrestrial base station → airborne receiver", "P.528", "Specific to aeronautical scenarios; slant path + atmosphere"],
        ["Satellite → airborne receiver", "P.619 / P.618", "Earth-space path; use itur.atmospheric_attenuation_slant_path"],
        ["Quick worst-case bound", "FSPL", "Always optimistic (most interference); use to bound the problem first"],
        ["High freq (>6 GHz) atmospheric loss", "P.676", "Gaseous absorption becomes significant above ~6 GHz"],
    ], columns=["Scenario", "Model", "Notes"])
    st.table(guidance)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — MONTE CARLO
# ─────────────────────────────────────────────────────────────────────────────
elif selected_tab == "🎲 Monte Carlo":
    st.title("🎲 Monte Carlo Aggregate Interference")
    ex("Per SM.2028, violation probability >5% is the standard incompatibility criterion. The simulation draws random transmitter positions from a uniform spatial distribution within the deployment area — the resulting I/N distribution is a function of aggregate received power summed linearly across all N interferers.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Interferer Parameters")
        n_interferers = st.number_input("Number of Interferers (N)", value=10, min_value=1, max_value=500, step=5)
        ex("N is the most sensitive assumption in any Monte Carlo study — aggregate I/N scales approximately as 10·log10(N) at high densities. Always document the source of your N assumption and challenge unrealistically low values in others' contributions.")
        mc_tx_power = st.number_input("Tx Power per Unit (dBm)", value=43.0, step=1.0)
        mc_tx_gain = st.number_input("Tx Antenna Gain (dBi)", value=15.0, step=1.0)
        mc_freq = st.number_input("Frequency (MHz)", value=4300.0, step=10.0)
        mc_radius = st.number_input("Deployment Radius (km)", value=20.0, step=1.0,
            help="Interferers placed randomly within this radius of victim")
        mc_model = st.selectbox("Propagation Model", ["FSPL", "P.452", "P.528"])
        mc_terrain = st.selectbox("Terrain (P.452 only)", ["suburban", "open", "urban"])
        mc_alt = st.number_input("Aircraft Altitude km (P.528 only)", value=3.0, step=0.5)

    with col2:
        st.subheader("Victim Receiver")
        band_mc = st.selectbox("FAA Protected Band:", ["(manual)"] + list(FAA_BANDS.keys()))
        if band_mc != "(manual)":
            b_mc = FAA_BANDS[band_mc]
            mc_noise = b_mc["noise_floor_dbm"]
            mc_thresh = b_mc["in_threshold_db"]
            st.info(f"Noise floor: {mc_noise} dBm | I/N threshold: {mc_thresh} dB")
        else:
            mc_noise = st.number_input("Rx Noise Floor (dBm)", value=-90.0)
            mc_thresh = st.number_input("I/N Threshold (dB)", value=-6.0)

        n_trials = st.select_slider("Monte Carlo Trials", [500, 1000, 2000, 5000, 10000], value=2000)
        ex(f"More trials = smoother statistics. 2000–5000 is sufficient for most policy contributions.")

    if st.button("🎲 Run Monte Carlo Simulation", type="primary"):
        with st.spinner("Running simulation..."):
            results = monte_carlo_aggregate(
                n_interferers=int(n_interferers),
                tx_power_dbm=mc_tx_power,
                tx_gain_dbi=mc_tx_gain,
                freq_mhz=mc_freq,
                deployment_radius_km=mc_radius,
                rx_noise_floor_dbm=mc_noise,
                in_threshold_db=mc_thresh,
                n_trials=n_trials,
                model=mc_model,
                terrain_type=mc_terrain,
                aircraft_alt_km=mc_alt,
            )

        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Median I/N", f"{results['in_p50_db']:.1f} dB")
        col_b.metric("95th pct I/N", f"{results['in_p95_db']:.1f} dB")
        col_c.metric("99th pct I/N", f"{results['in_p99_db']:.1f} dB")
        col_d.metric("Violation Probability", f"{results['violation_probability']*100:.1f}%",
                     delta="High Risk" if results['violation_probability'] > 0.05 else "Acceptable",
                     delta_color="inverse" if results['violation_probability'] > 0.05 else "normal")

        ex("The 99th percentile bounds the upper tail of the I/N distribution — for a 5% violation criterion, you need the exceedance probability at the threshold, not just the percentile. Read the CCDF directly at the I/N threshold to get the violation probability.")

        if results['violation_probability'] > 0.05:
            warn(f"Violation probability {results['violation_probability']*100:.1f}% exceeds 5% — this scenario poses a credible interference threat and supports a protection requirement.")
        elif results['violation_probability'] > 0.01:
            warn(f"Violation probability {results['violation_probability']*100:.1f}% is marginal — conservative assumptions needed.")
        else:
            ok(f"Violation probability only {results['violation_probability']*100:.2f}% — compatible under these assumptions.")

        # Distribution plot
        fig7, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig7.patch.set_facecolor("#0e1117")

        ax7 = axes[0]
        ax7.set_facecolor("#0e1117")
        ax7.hist(results['in_values'], bins=60, color='#4488ff', alpha=0.75, density=True)
        ax7.axvline(mc_thresh, color='red', linewidth=2, linestyle='--', label=f'I/N Threshold ({mc_thresh} dB)')
        ax7.axvline(results['in_p50_db'], color='yellow', linewidth=1.5, linestyle=':', label=f'Median ({results["in_p50_db"]:.1f} dB)')
        ax7.axvline(results['in_p99_db'], color='orange', linewidth=1.5, linestyle=':', label=f'99th pct ({results["in_p99_db"]:.1f} dB)')
        ax7.set_xlabel("I/N (dB)", color='white')
        ax7.set_ylabel("Probability Density", color='white')
        ax7.set_title("I/N Distribution Across Trials", color='white')
        ax7.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=8)
        ax7.tick_params(colors='white')
        for sp in ax7.spines.values():
            sp.set_color('#444')

        # CCDF
        ax8 = axes[1]
        ax8.set_facecolor("#0e1117")
        sorted_in = np.sort(results['in_values'])
        ccdf = 1.0 - np.arange(1, len(sorted_in) + 1) / len(sorted_in)
        ax8.semilogy(sorted_in, ccdf, color='#44ffaa', linewidth=2)
        ax8.axvline(mc_thresh, color='red', linewidth=2, linestyle='--', label=f'Threshold ({mc_thresh} dB)')
        ax8.axhline(0.05, color='orange', linewidth=1, linestyle=':', label='5% exceedance')
        ax8.axhline(0.01, color='yellow', linewidth=1, linestyle=':', label='1% exceedance')
        ax8.set_xlabel("I/N (dB)", color='white')
        ax8.set_ylabel("Probability of Exceedance", color='white')
        ax8.set_title("CCDF — Exceedance Probability", color='white')
        ax8.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=8)
        ax8.tick_params(colors='white')
        ax8.grid(color='#333', alpha=0.4)
        for sp in ax8.spines.values():
            sp.set_color('#444')

        plt.tight_layout()
        st.pyplot(fig7)

        ex("The CCDF shows P(I/N > x). Reading off at x = I/N threshold gives the violation probability directly. The SM.2028 criterion requires this value < 5% — if a proponent's CCDF intersects the threshold above the 5% line, their scenario is incompatible.")

        # Sensitivity: vary N interferers
        st.subheader("Sensitivity: Violation Probability vs Number of Interferers")
        ex("Aggregate I/N grows roughly as 10·log10(N) at low densities and more slowly as the deployment radius fills. The inflection point identifies a maximum compatible density — propose this as a PFD density limit or coordination zone in your contribution text.")
        n_range = list(range(1, min(int(n_interferers) * 3, 100), max(1, int(n_interferers) // 5)))
        viol_probs = []
        with st.spinner("Running sensitivity sweep..."):
            for n in n_range:
                r = monte_carlo_aggregate(
                    n_interferers=n,
                    tx_power_dbm=mc_tx_power,
                    tx_gain_dbi=mc_tx_gain,
                    freq_mhz=mc_freq,
                    deployment_radius_km=mc_radius,
                    rx_noise_floor_dbm=mc_noise,
                    in_threshold_db=mc_thresh,
                    n_trials=500,
                    model=mc_model,
                    terrain_type=mc_terrain,
                    aircraft_alt_km=mc_alt,
                )
                viol_probs.append(r['violation_probability'] * 100)

        fig8, ax9 = plt.subplots(figsize=(10, 4))
        fig8.patch.set_facecolor("#0e1117")
        ax9.set_facecolor("#0e1117")
        ax9.plot(n_range, viol_probs, color='#ff8844', linewidth=2, marker='o', markersize=4)
        ax9.axhline(5, color='red', linestyle='--', label='5% threshold')
        ax9.axvline(n_interferers, color='white', linestyle=':', alpha=0.5, label=f'Selected N={n_interferers}')
        ax9.set_xlabel("Number of Interferers (N)", color='white')
        ax9.set_ylabel("Violation Probability (%)", color='white')
        ax9.set_title("Risk vs. Deployment Density", color='white')
        ax9.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=9)
        ax9.tick_params(colors='white')
        ax9.grid(color='#333', alpha=0.4)
        for sp in ax9.spines.values():
            sp.set_color('#444')
        plt.tight_layout()
        st.pyplot(fig8)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — CONTRIBUTION SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
elif selected_tab == "📋 Contribution Summary":
    st.title("📋 ITU-R Contribution Summary Generator")
    ex("A contribution carries the weight of your administration's regulatory position — vague language weakens it. Specific dB margins, named propagation models, and cited Recommendation/RR article numbers are what Working Party rapporteurs actually use to draft agreed text.")

    st.subheader("Scenario Setup")
    contrib_title = st.text_input("Analysis Title", value="Interference Assessment: New IMT Allocation vs. Radio Altimeter Band")
    submitter = st.text_input("Submitting Administration", value="United States of America")
    meeting = st.text_input("Target Meeting", value="ITU-R WP 5D Meeting, [Date]")
    wrc_agenda = st.text_input("WRC Agenda Item", value="AI [X.Y] — [description]")

    protected_band = st.selectbox("Protected FAA System", list(FAA_BANDS.keys()))
    b_c = FAA_BANDS[protected_band]

    st.subheader("Analysis Parameters (from your prior analysis)")
    col1, col2, col3 = st.columns(3)
    with col1:
        c_tx_power = st.number_input("Interferer EIRP (dBm)", value=58.0)
        c_freq = st.number_input("Interferer Frequency (MHz)", value=4300.0)
        c_sep = st.number_input("Frequency Separation (MHz)", value=200.0)
    with col2:
        c_dist = st.number_input("Min Separation Distance (km)", value=5.0)
        c_in = st.number_input("Computed I/N (dB)", value=-4.0)
        c_viol_prob = st.number_input("Violation Probability (%)", value=8.5)
    with col3:
        c_margin = b_c["in_threshold_db"] - c_in
        st.metric("Protection Margin", f"{c_margin:.1f} dB",
                  delta_color="normal" if c_margin >= 0 else "inverse")
        compatible = c_margin >= 0 and c_viol_prob < 5.0

    # Generate contribution text
    if st.button("📄 Generate Contribution Technical Summary", type="primary"):
        status = "COMPATIBLE" if compatible else "INCOMPATIBLE"
        recommendation = (
            "The analysis supports compatibility subject to the conditions specified below."
            if compatible else
            "The analysis demonstrates that the proposed new allocation/use presents an unacceptable "
            "risk of harmful interference to the protected aeronautical service. The United States "
            "opposes this proposal without additional protective measures."
        )

        contrib_text = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ITU-R WORKING PARTY [5D / 5B] — TECHNICAL CONTRIBUTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Document:      [Document Number TBD]
Submitted by:  {submitter}
Meeting:       {meeting}
Agenda Item:   {wrc_agenda}

TITLE: {contrib_title}

━━━━━━━━━━━━━━━━━━━━━━
1. EXECUTIVE SUMMARY
━━━━━━━━━━━━━━━━━━━━━━
{submitter} has conducted an interference analysis assessing the 
compatibility of proposed new spectrum use at {c_freq:.1f} MHz with 
the protected aeronautical service: {b_c['system']} operating in 
{b_c['f_low_mhz']}–{b_c['f_high_mhz']} MHz ({b_c['allocation']}).

PRELIMINARY FINDING: {status}
{recommendation}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. PROTECTION CRITERIA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The following protection criteria are applied, consistent with 
ITU-R Recommendation {b_c.get('rtca_doc','[applicable Rec.]')} 
and established ICAO/FAA standards:

  • Protected System:     {b_c['system']}
  • Operating Band:       {b_c['f_low_mhz']}–{b_c['f_high_mhz']} MHz
  • Allocation:           {b_c['allocation']}
  • Safety-of-Life:       {'YES — RR No. 4.10 applies' if b_c['safety_of_life'] else 'No'}
  • I/N Threshold:        {b_c['in_threshold_db']} dB
  • Receiver Noise Floor: {b_c['noise_floor_dbm']} dBm (estimated)
  • Frequency Separation: {c_sep:.1f} MHz from edge of protected band
  • Reference Standard:   {b_c['rtca_doc']}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. INTERFERENCE ANALYSIS METHODOLOGY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The analysis follows the methodology of ITU-R Recommendation 
SM.2028 (Monte Carlo simulation) and ITU-R M.1642 (IMT/ARNS 
compatibility). Propagation was assessed using:

  • ITU-R P.528  — Aeronautical mobile/radionavigation propagation
  • ITU-R P.452  — Terrestrial interference (point-to-area)
  • ITU-R P.676  — Gaseous attenuation (itur library implementation)

Scenario parameters:
  • Interferer EIRP:           {c_tx_power:.1f} dBm
  • Interferer Frequency:      {c_freq:.1f} MHz
  • Victim Receiver Location:  Airborne / ground-based aeronautical
  • Minimum Separation:        {c_dist:.1f} km

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. KEY RESULTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • Computed I/N (worst case):  {c_in:.1f} dB
  • I/N Threshold:              {b_c['in_threshold_db']} dB
  • Protection Margin:          {c_margin:.1f} dB  ({'VIOLATED' if c_margin < 0 else 'MET'})
  • Monte Carlo violation prob: {c_viol_prob:.1f}%  ({'EXCEEDS' if c_viol_prob > 5 else 'within'} 5% criterion)

{'⚠ The I/N threshold is violated under the assessed scenario.' if c_margin < 0 else '✓ The I/N threshold is nominally met under median conditions.'}
{'⚠ The 5% exceedance criterion is not met — harmful interference cannot be excluded.' if c_viol_prob > 5 else '✓ Exceedance probability is within the 5% criterion.'}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5. PROPOSED REGULATORY ACTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{'The United States proposes that the Working Party adopt a footnote or Resolution establishing the following protection measures to safeguard ' + b_c["system"] + ':' if not compatible else 'The United States notes that compatibility can be maintained provided the following conditions are observed:'}

  [a] Power flux density limits at the edge of the aeronautical 
      service area shall not exceed [XX] dB(W/m²)/MHz;
  [b] Minimum coordination distance of [XX] km from aeronautical 
      facilities shall be maintained;
  [c] Out-of-band emission limits into {b_c['f_low_mhz']}–{b_c['f_high_mhz']} MHz 
      shall not exceed [XX] dBc/MHz at {c_sep:.0f} MHz offset;
  [d] Further study is required under [specific WRC-27 agenda item].

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
6. REGULATORY BASIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • Radio Regulations No. 4.10 — Protection of safety-of-life services
  • Radio Regulations No. 5.444 — ARNS frequency coordination
  • RR Resolution 233 — Protection of RNSS systems
  • ITU-R M.1642 — Methodology for IMT/ARNS compatibility
  • ICAO Annex 10 — Aeronautical Telecommunications

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
7. COORDINATION NOTE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This analysis was developed in coordination with the Federal 
Aviation Administration (FAA) and submitted through the National 
Telecommunications and Information Administration (NTIA). 
ICAO has been informed and is developing complementary input 
through [ICAO Navigation Systems Panel / FMG].

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
8. CONCLUSION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{submitter} urges the Working Party to take the results of this 
analysis into account when developing the CPM Report method(s) 
for WRC-27 Agenda Item {wrc_agenda}.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[End of Document]
Generated by FAA RF Interference Analysis Tool
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        st.text_area("Generated Contribution Draft", contrib_text, height=600)
        st.download_button("⬇️ Download as .txt", contrib_text,
                           file_name="ITU_R_Contribution_Draft.txt", mime="text/plain")
        ex("Before submission through NTIA: verify all numerical results with SEAMCAT or ITU-R SoftTools for P.452/P.528; confirm I/N threshold citations against current RTCA DO standards; have FAA spectrum office review for classification and policy alignment.")

    # Regulatory citation quick reference
    st.subheader("Regulatory Citation Quick Reference")
    ex("Regulatory citations transform a technical finding into a legal position — 'our analysis shows threshold violated' is an opinion; 'computed I/N exceeds the criterion in M.1318 Annex 1 Step 1b, invoking RR No. 4.10' is a treaty-level regulatory argument.")
    reg_refs = pd.DataFrame([
        ["RR No. 4.10", "No harmful interference to safety-of-life services", "Strongest lever — invoked for ALL FAA safety systems"],
        ["RR No. 5.444", "ARNS protection at 960–1215 MHz", "Use for DME/TACAN/SSR/TCAS/ADS-B bands"],
        ["RR No. 5.328", "ARNS at 108–137 MHz; RNSS cannot claim protection from ARNS in 1164–1215 MHz", "VOR/ILS protection; also limits RNSS protection claims vs DME"],
        ["RR Resolution 233", "Protection of RNSS (GPS/GNSS)", "Use for all GPS/GNSS band defense"],
        ["RR Resolution 750", "IMT and safety services coexistence", "Relevant for all WP 5D IMT proposals"],
        ["ITU-R M.1318 Annex 1", "c = a − b methodology for aggregate non-RNSS interference to GNSS", "Cite as the formal calculation framework for GPS L1/L2/L5 protection analysis"],
        ["ITU-R M.1477 Annex 5", "Aeronautical safety margin ≥6 dB for GNSS; +10 dB for narrowband (≤700 Hz) interferers", "Cite to justify b = 6 dB in c = a − b; invoke narrowband rule for CW/tonal interferers"],
        ["ITU-R M.1904", "GLONASS spaceborne receiver — safety margin = 6 dB (Annex 1 Table 1 Note 3)", "Cite alongside M.1905 for GLONASS band contributions; reinforces 6 dB doctrine"],
        ["ITU-R M.1905", "Safety margin must be applied for RNSS safety-of-life interference analyses (Note 1: 6 dB aero)", "Broadest RNSS safety margin authority — applies to ALL RNSS systems, cite in every GNSS contribution"],
        ["ITU-R M.1642", "IMT→ARNS methodology", "Cite as methodology basis for non-GNSS aeronautical analysis"],
        ["ITU-R SM.2028", "Monte Carlo simulation methodology", "Cite to validate your simulation approach"],
        ["ITU-R P.528", "Aeronautical propagation model", "Model authority — cite when using P.528 curves"],
        ["ICAO Annex 10", "Aeronautical telecomm standards", "Aligns ITU-R work with ICAO civil aviation requirements"],
    ], columns=["Reference", "Subject", "When to Cite"])
    st.dataframe(reg_refs, use_container_width=True)
    ex("Citation stacking is a standard ITU-R practice: M.1318 provides the methodology, M.1477 provides the aeronautical safety margin value, M.1904/M.1905 establish the 6 dB doctrine across GNSS systems — together they preclude any argument that the margin is discretionary.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 7 — TUTORIAL
# ─────────────────────────────────────────────────────────────────────────────
elif selected_tab == "📚 Tutorial":
    st.title("📚 How to Use This Tool")
    ex("The five-step workflow maps directly to the ITU-R study cycle timeline: identify threats at CPM study launch (4–5 years before WRC), quantify during WP meetings, draft contributions 6 weeks before each meeting, engage at the meeting, implement after WRC.")

    st.markdown("---")
    st.header("🗺️ The Big Picture — Your Workflow")
    ex("The study cycle runs approximately 4 years from WRC agenda item adoption to the next WRC. Planting conservative protection criteria early in Step 1 Recommendations is far more effective than fighting a nearly-agreed CPM method in the final study cycle year.")

    st.markdown("""
```
1. IDENTIFY  →  A new ITU-R proposal may affect a protected FAA band
2. ASSESS    →  Run link budget + Monte Carlo to quantify the threat
3. COMPARE   →  Check results against ITU-R protection criteria
4. DRAFT     →  Build the US contribution with technical findings
5. ENGAGE    →  Submit through NTIA, coordinate with ICAO/allies
```
    """)

    st.markdown("---")
    st.header("📡 Module 1 — Protected Bands")
    ex("Always start here — know exactly what you're defending before touching any analysis tool.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("What to do")
        st.markdown("""
**Step 1.** Open the **Spectrum Overview** chart — scan it visually to see if the new proposal's frequency sits near or inside a protected zone.

**Step 2.** Select the threatened band from the **Band Details** dropdown. Note these four numbers — you will use them in every subsequent module:
- Lower / upper frequency bounds
- I/N Threshold (dB)
- Noise floor (dBm)
- RTCA reference document

**Step 3.** Confirm the band is marked **Safety-of-Life** — if it is, RR No. 4.10 applies and your policy position is automatically stronger.
        """)
    with col2:
        st.subheader("Key questions to answer")
        st.markdown("""
| Question | Where to find it |
|---|---|
| What allocation protects this band? | Allocation field |
| What I/N threshold must I defend? | I/N Threshold field |
| What RTCA standard governs the receiver? | RTCA Doc field |
| Is it safety-of-life? | Red 🔴 label |
| How wide is the band? | BW (MHz) column |
        """)

    st.markdown("---")
    st.header("🔗 Module 2 — Link Budget")
    ex("The link budget tells you how much interference actually reaches the victim receiver — this is the single-scenario worst-case calculation.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Step-by-step walkthrough")
        st.markdown("""
**Step 1 — Set up the interferer (left panel)**
- Enter the new system's transmit power in dBm
  - *Tip: 43 dBm = 20W typical LTE macro base station*
  - *Tip: 30 dBm = 1W small cell / low-power device*
- Enter antenna gain toward the victim receiver
  - *Use 0 dBi for worst case (isotropic toward victim)*
- Set cable loss (usually 1–3 dB)

**Step 2 — Choose propagation model**
- **FSPL** — Use first; gives you the worst-case bound (most interference)
- **P.452** — Use for ground-to-ground scenarios; pick terrain type
- **P.528** — Use when victim is airborne (aircraft, helicopter)
- *Rule: Always start with FSPL. If it passes, you're protected. If it fails, run P.528/P.452 for a more realistic picture.*

**Step 3 — Set up the victim receiver (right panel)**
- Click **"Auto-fill from FAA band"** and select your protected band
- This loads the correct noise floor and I/N threshold automatically

**Step 4 — Click Run Link Budget**
- Read the **Protection Margin** metric — this is your headline number
- Positive = protected; negative = threshold violated
        """)
    with col2:
        st.subheader("How to interpret results")
        st.markdown("""
| Protection Margin | Meaning | Action |
|---|---|---|
| > +10 dB | Well protected | Compatible; document and move on |
| 0 to +10 dB | Marginally protected | Flag for conservative assumptions; run Monte Carlo |
| < 0 dB | Threshold violated | Cite harmful interference; draft objection |

**Reading the waterfall chart:**
- Green bars = gains (help the signal)
- Red bars = losses (reduce interference reaching victim)
- The gap between Rx Power and the red dotted threshold line is your protection margin

**Reading the distance sweep:**
- The red dotted line shows the minimum path loss needed for protection
- Where FSPL curve crosses that line = minimum safe separation distance
- Cite this distance in your contribution as a coordination zone requirement

**Common mistakes to avoid:**
- Don't use FSPL as your final model for regulatory submissions — reviewers will push back
- Don't forget to account for OOBE (out-of-band emissions) — the interferer's frequency may be outside the protected band but its sidelobes may not be
- Always use 0 dBi receive antenna gain for worst-case analysis
        """)

    st.markdown("---")
    st.header("📊 Module 3 — Noise & I/N Analysis")
    ex("Use this module to verify protection criteria and build the receiver characterization section of your contribution.")

    st.markdown("""
**When to use this module:**
- When you need to calculate the noise floor for a specific receiver that isn't in the FAA band database
- When you want to show how I/N varies across a range of interference levels (for the heatmap)
- When reviewing another administration's claimed protection criteria — enter their numbers and verify

**Key formula to remember:**
```
Noise Floor (dBm) = -174 + 10·log₁₀(Bandwidth_Hz) + Noise_Figure_dB
```
- -174 dBm/Hz is thermal noise at 290K (room temperature)
- Bandwidth widens the noise floor upward (10 MHz BW = 10 dB higher floor than 1 MHz BW)
- Every dB of noise figure directly raises the floor

**I/N threshold selection guide:**
| System | Threshold | Rationale |
|---|---|---|
| GNSS / GPS | −10 dB | Satellite signal is extremely weak; any noise rise is catastrophic |
| ADS-B / Mode-S | −10 dB | Safety surveillance; false negatives unacceptable |
| Radio Altimeter | −6 dB | Safety-of-life; conservative protection required |
| VOR / ILS | −6 dB | Approach/landing guidance; high consequence |
| General ARNS | −6 dB | ITU-R M.1642 standard criterion |
    """)

    st.markdown("---")
    st.header("🌐 Module 4 — Propagation")
    ex("Use this to compare models side-by-side and select the most appropriate one before running your formal analysis.")

    st.markdown("""
**Model selection decision tree:**
```
Is the victim receiver airborne?
  YES → Use P.528
  NO  → Is the path over ocean or open terrain?
          YES → Use P.452 (open)
          NO  → Use P.452 (suburban or urban)
          
Always run FSPL first as a bounding calculation.
```

**The atmospheric attenuation chart (P.676 via itur):**
- Shows how much signal is absorbed by oxygen and water vapor over your chosen path length
- Above ~10 GHz, this becomes significant and actually helps protect receivers
- Below ~3 GHz, gaseous absorption is minimal — don't count on it for protection

**What the time percentage means in P.528:**
- 50% = median conditions (use for typical interference assessment)
- 1% = worst-case propagation (use for regulatory/protection studies — most conservative)
- ITU-R contributions for protection should use 1% or worst-case conditions
    """)

    st.markdown("---")
    st.header("🎲 Module 5 — Monte Carlo")
    ex("Monte Carlo is the ITU-R standard methodology for aggregate interference — it's what distinguishes a rigorous contribution from a back-of-envelope calculation.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("How to set it up")
        st.markdown("""
**Number of interferers (N):**
- This is the most influential assumption — challenge it in others' contributions
- For IMT base stations: 5–20 per km² is typical urban density
- For satellite downlinks: 1 (single beam) to thousands (LEO constellation)
- Always ask: is the proponent's assumed N realistic for actual deployment?

**Deployment radius:**
- How far from the victim receiver could interferers realistically be placed?
- For aeronautical: consider that an aircraft at altitude has line-of-sight to transmitters hundreds of km away
- For ground receivers: 20–50 km is a reasonable urban deployment zone

**Propagation model for Monte Carlo:**
- Use P.528 if victim is airborne
- Use P.452 for ground victims
- FSPL gives you the worst-case bound
        """)
    with col2:
        st.subheader("How to interpret results")
        st.markdown("""
**The four key numbers:**
| Metric | How to use it |
|---|---|
| Median I/N (50th pct) | Typical day; use for general compatibility claim |
| 95th percentile I/N | Near-worst case; use in contribution headline finding |
| 99th percentile I/N | Extreme case; use to argue for extra margin |
| Violation probability | Must be < 5% (ITU-R standard criterion) |

**Reading the CCDF chart:**
- X-axis = I/N level; Y-axis = probability of exceeding that level
- Draw a vertical line at your I/N threshold — read across to Y-axis
- That Y value is your violation probability
- If it's above 5% (0.05), the scenario is incompatible under ITU-R criteria

**The sensitivity sweep:**
- Shows how violation probability grows with deployment density
- Use this to argue for a maximum density limit in your contribution
- Example: "Compatibility is maintained only if deployment density does not exceed N units per km²"
        """)

    st.markdown("---")
    st.header("📋 Module 6 — Contribution Summary")
    ex("This module generates a structured ITU-R contribution draft — fill in your analysis numbers and it produces properly formatted regulatory text.")

    st.markdown("""
**Anatomy of an ITU-R contribution:**
| Section | Purpose | Tips |
|---|---|---|
| Executive Summary | One-paragraph finding | Lead with your conclusion — Compatible or Incompatible |
| Protection Criteria | What you're protecting | Cite I/N threshold, RTCA doc, RR article |
| Methodology | How you analyzed it | Cite P.528, SM.2028, M.1642 |
| Key Results | Your numbers | I/N value, margin, violation probability |
| Proposed Regulatory Action | What you want | Specific PFD limits, coordination distances, OOBE masks |
| Regulatory Basis | Legal grounding | RR 4.10, Resolution 233, etc. |
| Coordination Note | Who's behind you | NTIA, FAA, ICAO — weight in numbers |

**Tips for getting your contribution accepted:**
1. **Lead with safety** — open with the safety-of-life consequence of interference, not the math
2. **Be specific** — "interference may occur" is weak; "I/N threshold exceeded by 4.2 dB at 5 km separation" is strong
3. **Pre-coordinate with ICAO** — an ICAO liaison statement supporting your position changes the room dynamics entirely
4. **Propose text, not just concerns** — submissions that include proposed Radio Regulations language are far more influential than those that only raise problems
5. **Bracket aggressively early** — in early study cycles, propose conservative limits; there will be pressure to relax them later
    """)

    st.markdown("---")
    st.header("🔍 Module 7 — Contribution Analyzer (AI-Powered)")
    ex("The next module uses AI to read any WP 5D or 5B contribution you paste in and instantly gives you FAA-focused policy guidance.")
    st.markdown("""
**How to use it:**
1. Go to the **🤖 Contribution Analyzer** module in the sidebar
2. Paste the text or key sections of any ITU-R contribution
3. Fill in the metadata (document number, working party, submitting admin)
4. Click **Analyze** — the AI will assess:
   - Which FAA protected bands are at risk
   - What the submitting administration is trying to achieve
   - What the US/FAA policy position should be
   - What counter-arguments to raise
   - Which RR articles and Recommendations to cite in response
   - Whether to oppose, support, or propose amendments
    """)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 8 — AI CONTRIBUTION ANALYZER
# ─────────────────────────────────────────────────────────────────────────────
elif selected_tab == "🤖 Contribution Analyzer":
    st.title("🤖 AI Contribution Analyzer")
    ex("Paste any ITU-R WP 5D or WP 5B contribution and get instant FAA-focused policy guidance — powered by Claude AI.")

    # API key handling
    api_key = None
    try:
        api_key = st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        pass

    if not api_key:
        st.warning("⚠️ Anthropic API key not configured. See setup instructions below.")
        with st.expander("🔧 How to add your API key to Streamlit Cloud"):
            st.markdown("""
**One-time setup — takes 2 minutes:**

1. Go to [console.anthropic.com](https://console.anthropic.com) and create a free account
2. Click **API Keys** → **Create Key** → copy it
3. Go to your app at [share.streamlit.io](https://share.streamlit.io)
4. Click the **⋮ menu** next to your app → **Settings** → **Secrets**
5. Paste this exactly (replace with your real key):
```toml
ANTHROPIC_API_KEY = "sk-ant-..."
```
6. Click **Save** — the app restarts automatically

Once configured, the analyzer will work every time you visit the app.
            """)
        st.stop()

    # Metadata inputs
    st.subheader("Contribution Metadata")
    ex("Fill in what you know about the document — even partial info helps the AI contextualize the analysis.")

    col1, col2, col3 = st.columns(3)
    with col1:
        doc_number = st.text_input("Document Number", placeholder="e.g., 5D/123-E")
        working_party = st.selectbox("Working Party", ["WP 5D (IMT/Mobile)", "WP 5B (Maritime/Radiodetermination)", "WP 5A (Land Mobile)", "SG 5 (Study Group)", "Other"])
    with col2:
        submitting_admin = st.text_input("Submitting Administration(s)", placeholder="e.g., China, European Union")
        meeting_date = st.text_input("Meeting / Date", placeholder="e.g., WP 5D #44, Oct 2025")
    with col3:
        agenda_item = st.text_input("WRC Agenda Item", placeholder="e.g., AI 1.2 — IMT identification")
        doc_type = st.selectbox("Document Type", [
            "New proposal / spectrum sharing study",
            "Interference analysis",
            "Draft new Recommendation",
            "Amendment to existing Recommendation",
            "Liaison statement",
            "Information document",
        ])

    st.subheader("Contribution Text")
    ex("Paste the full contribution or the key technical and regulatory sections — the more context you provide, the more precise the guidance.")

    contrib_input = st.text_area(
        "Paste contribution text here:",
        height=300,
        placeholder="""Paste the ITU-R contribution text here. You can include:
- The executive summary or introduction
- Technical analysis sections
- Proposed regulatory text or amendments
- Conclusions and proposals

Even a partial paste (e.g., just the conclusions section) will produce useful guidance."""
    )

    # ── Interference Classification Reference Card ────────────────────────────
    with st.expander("📋 ITU-R Interference Classification Reference — Applied in Every Analysis"):
        ex("The AI uses this exact framework from your slide to characterize interference in every contribution it analyzes.")

        st.markdown("### Unwanted Emissions — Source Characterization")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.markdown("""
<div style='background:#1a2a3a;border-left:4px solid #4488ff;padding:10px 14px;border-radius:4px;margin-bottom:8px'>
<b style='color:#4488ff'>SPURIOUS EMISSIONS</b><br>
<span style='color:#cce;font-size:0.88em'>
Frequencies just outside the necessary bandwidth whose level <b>may be reduced without 
affecting the corresponding transmission of information.</b><br><br>
Includes: harmonics, parasitic emissions, intermodulation products, frequency conversion products.<br>
<b>Explicitly EXCLUDES out-of-band emissions.</b><br><br>
Regulatory handle: RR Appendix 3 spurious emission limits
</span>
</div>""", unsafe_allow_html=True)
        with col_r2:
            st.markdown("""
<div style='background:#1a2a3a;border-left:4px solid #ff8844;padding:10px 14px;border-radius:4px;margin-bottom:8px'>
<b style='color:#ff8844'>OUT-OF-BAND (OOB) EMISSIONS</b><br>
<span style='color:#cce;font-size:0.88em'>
Frequencies <b>immediately outside the necessary bandwidth</b> which result from 
<b>the modulation process itself</b>, but excluding spurious emissions.<br><br>
Key distinction: OOB = caused by modulation. Spurious = hardware non-idealities.<br><br>
Regulatory handle: Emission designator masks and guard band requirements
</span>
</div>""", unsafe_allow_html=True)

        st.markdown("### Interference Classification — RR Article 1.166")
        col_i1, col_i2, col_i3 = st.columns(3)
        with col_i1:
            st.markdown("""
<div style='background:#3a1a1a;border-left:4px solid #ff4444;padding:10px 14px;border-radius:4px'>
<b style='color:#ff6666'>🔴 HARMFUL INTERFERENCE</b><br>
<span style='color:#aaa;font-size:0.82em'>RR 1.169</span><br><br>
<span style='color:#ffaaaa;font-size:0.88em'>
Interference which <b>endangers the functioning</b> of a radionavigation service or 
other safety services, OR <b>seriously degrades, obstructs, or repeatedly interrupts</b> 
a radiocommunication service operating in accordance with the RR.<br><br>
<b>→ Triggers RR No. 4.10<br>→ US must oppose<br>→ Demand cessation</b>
</span>
</div>""", unsafe_allow_html=True)
        with col_i2:
            st.markdown("""
<div style='background:#1a3a1a;border-left:4px solid #44bb44;padding:10px 14px;border-radius:4px'>
<b style='color:#66dd66'>🟢 PERMISSIBLE INTERFERENCE</b><br>
<span style='color:#aaa;font-size:0.82em'>RR 1.167</span><br><br>
<span style='color:#aaffaa;font-size:0.88em'>
Observed or predicted interference which <b>complies with quantitative interference 
and sharing criteria</b> in the Radio Regulations or in ITU-R Recommendations or 
in special agreements.<br><br>
<b>→ Acceptable if within criteria<br>→ Used in frequency coordination<br>→ Monitor the criteria</b>
</span>
</div>""", unsafe_allow_html=True)
        with col_i3:
            st.markdown("""
<div style='background:#2a2a1a;border-left:4px solid #ffbb44;padding:10px 14px;border-radius:4px'>
<b style='color:#ffdd66'>🟡 ACCEPTED INTERFERENCE</b><br>
<span style='color:#aaa;font-size:0.82em'>RR 1.168</span><br><br>
<span style='color:#ffffaa;font-size:0.88em'>
Interference at a <b>higher level than permissible</b>, which has been <b>agreed upon 
between two or more administrations</b> without prejudice to other administrations.<br><br>
<b>→ Requires explicit bilateral agreement<br>→ Cannot bind other admins<br>→ Verify scope carefully</b>
</span>
</div>""", unsafe_allow_html=True)

        st.caption("Both permissible interference and accepted interference are used in the coordination of frequency assignments between administrations (RR 1.167/1.168).")

    st.markdown("---")

    # Optional context
    with st.expander("➕ Additional Context (optional but helpful)"):
        user_concern = st.text_area("Specific FAA concern you want addressed:",
            placeholder="e.g., We are concerned this proposal could affect radio altimeter operations during CAT III approaches near 5G deployments at major airports.",
            height=100)
        prior_us_position = st.text_area("Prior US position on this topic (if known):",
            placeholder="e.g., The US has previously opposed new primary allocations in the 4200-4400 MHz band and submitted 5D/456 opposing similar proposals.",
            height=100)

    analysis_depth = st.selectbox("Analysis Depth", [
        "Quick assessment (key risks + recommended US position)",
        "Standard analysis (full policy brief with citations)",
        "Deep dive (comprehensive brief + draft response contribution outline)",
    ])

    if st.button("🔍 Analyze Contribution", type="primary", disabled=not contrib_input.strip()):

        # Build the prompt
        depth_instruction = {
            "Quick assessment (key risks + recommended US position)":
                "Provide a concise 3-section analysis: (1) Key Risks to FAA systems, (2) Recommended US Position, (3) Top 3 regulatory citations to invoke. Be direct and brief.",
            "Standard analysis (full policy brief with citations)":
                "Provide a full policy brief with all sections as specified.",
            "Deep dive (comprehensive brief + draft response contribution outline)":
                "Provide a comprehensive brief with all sections, plus a detailed outline of a draft US response contribution including proposed regulatory text.",
        }[analysis_depth]

        faa_bands_summary = "\n".join([
            f"- {name}: {b['f_low_mhz']}–{b['f_high_mhz']} MHz ({b['allocation']}), "
            f"I/N threshold {b['in_threshold_db']} dB, "
            f"Aviation safety factor {b.get('aviation_safety_factor_db',0)} dB, "
            f"Effective threshold {b.get('effective_threshold_db', b['in_threshold_db'])} dB, "
            f"Safety Service (RR 1.59): {b.get('rr_1_59', True)}"
            for name, b in FAA_BANDS.items()
        ])

        system_protection_table = """
FAA SYSTEM PROTECTION LEVELS (from FAA Interference Protection Considerations slides):
| System | Protection Level | Notes |
|--------|-----------------|-------|
| ARSR (Air Route Surveillance Radar) | I/N = −6 dB | Long-range ATC radar |
| ASR (Airport Surveillance Radar) | I/N = −10 dB | Short-range terminal radar |
| RNSS Feeder Links | ΔT/T = 6% | Noise temperature rise metric |
| L1 SBAS Type 1 | I < −146.5 dBW/MHz; I/N ≈ −5 dB + 6 dB safety margin | Wideband RFI; acquisition threshold |
| L2 SBAS Ground Reference Rx | I < −147.5 dBW/MHz; I/N ≈ −6 dB | Wideband RFI |
| L-band AMS(R)S Service & Feeder Links | ΔT/T = 20% aggregate = 6% single-entry | Noise temperature metric |
| DME | epfd ≤ −121.5 dBW/m² in any 1 MHz band | Effective power flux density limit |
| MLS | pfd ≤ −124.5 dBW/m² in 150 kHz band | Power flux density limit |

AVIATION SAFETY FACTOR: Some aeronautical applications (e.g. precision approach and landing) are critical,
meriting an additional safety factor of not less than 6 dB on top of the base I/N threshold.

AERONAUTICAL SPECTRUM FUNCTIONS (per FAA slides):
1. Communications — AM(R)S & AMS(R)S: voice and data between aircraft and ground; safety requiring high integrity and rapid response; "(R)" = safety and regularity
2. Navigation — ARNS, RNSS, RNS (Radionavigation Service), RDSS (Radiodetermination-satellite), Aeronautical RNSS
3. Surveillance — SSR, ADS-B, TCAS for ATC separation

SAFETY SERVICE (RR 1.59): Any radiocommunication service used permanently or temporarily for the safeguarding of human life and property. ALL aeronautical bands listed above qualify.

FREQUENCY SHARING BASIS: Permissible interference / non-interference basis. Permissible interference results in acceptable minimal performance change and is typically a change in the noise floor or received signal-to-noise ratio."""

        system_prompt = f"""You are a senior RF spectrum policy advisor supporting the FAA and NTIA in ITU-R proceedings. 
Your role is to analyze ITU-R contributions from Working Party 5D (IMT/Mobile) and Working Party 5B (Maritime/Radiodetermination) 
and provide precise, actionable policy guidance to protect US aeronautical interests.

The following FAA frequency bands are protected and must be defended:
{faa_bands_summary}

{system_protection_table}

Key regulatory instruments available:
- RR No. 4.10: No harmful interference to safety-of-life services
- RR No. 1.59: Safety service — any radiocommunication service used for safeguarding human life and property
- RR No. 5.444: ARNS protection at 960-1215 MHz
- RR No. 5.328: ARNS at 108-137 MHz; RNSS cannot claim protection from ARNS in 1164-1215 MHz
- RR Resolution 233/236: RNSS/GNSS protection
- RR Resolution 750: IMT and safety services coexistence
- ITU-R M.1318: c = a − b methodology for GNSS aggregate interference
- ITU-R M.1477: 6 dB aeronautical safety margin; +10 dB for narrowband (≤700 Hz) interferers
- ITU-R M.1904 / M.1905: GLONASS and RNSS 6 dB safety margin doctrine
- ITU-R M.1642: Methodology for IMT/ARNS compatibility assessments
- ITU-R SM.2028: Monte Carlo simulation methodology
- ICAO Annex 10: Aeronautical telecommunications standards
- RTCA standards: DO-235B (GNSS), DO-260B (ADS-B), DO-155 (Radio Altimeter)

SOURCE-PATH-RECEIVER (SPR) ANALYSIS FRAMEWORK — MANDATORY:
Aviation interference analysis uses the SPR model with worst-case limits on ALL three elements:
- SOURCE: Max transmit power, worst-case antenna gain, worst-case signal characteristics (e.g. max EIRP, highest OOB emission level)
- PATH: Free-space attenuation especially above 1 GHz; distance separation >20 km; worst-case propagation (P.528 for airborne, P.452 for ground; time %=1%)
- VICTIM: Receiver susceptibility mask, worst-case antenna gain toward interferer, receiver noise power
AGGREGATE EFFECT: The aggregate effect of multiple interference sources must be considered and due allowance made (Monte Carlo per SM.2028).
PROTECTION CRITERIA: Max interference threshold limit is used as a protection criterion, taking into account all environmental conditions.
AVIATION SAFETY FACTOR: Precision approach and landing applications require an ADDITIONAL 6 dB safety factor on top of the base I/N threshold.
PERMISSIBLE INTERFERENCE: Results in acceptable minimal performance change — typically a change in the noise floor or received signal-to-noise ratio.

═══════════════════════════════════════════════════════════════════════════
MANDATORY INTERFERENCE CLASSIFICATION FRAMEWORK
Based on ITU Radio Regulations — apply to EVERY analysis without exception
═══════════════════════════════════════════════════════════════════════════

LAYER 1 — UNWANTED EMISSIONS (characterize the SOURCE of interfering energy):

SPURIOUS EMISSIONS
Definition: Emissions at frequency or frequencies just outside the necessary bandwidth,
the level of which MAY BE REDUCED without affecting the corresponding transmission of
information. Spurious emissions include: harmonic emissions, parasitic emissions,
intermodulation products, and frequency conversion products.
IMPORTANT: Spurious emissions EXPLICITLY EXCLUDE out-of-band emissions.
Regulatory handle: RR Appendix 3 spurious emission limits. Argue for tighter masks.

OUT-OF-BAND (OOB) EMISSIONS
Definition: Emissions at frequency or frequencies IMMEDIATELY outside the necessary
bandwidth, which result from the MODULATION PROCESS itself, but excluding spurious emissions.
Key distinction from spurious: OOB is an inherent consequence of modulation;
spurious is caused by hardware non-idealities (harmonics, intermod, parasitics).
Regulatory handle: Emission designator/mask requirements. Argue for tighter OOB masks
and larger guard bands between the interfering service and the protected band.

LAYER 2 — INTERFERENCE CLASSIFICATION (RR Article 1.166 — classify by IMPACT level):

HARMFUL INTERFERENCE (RR 1.169)
Definition: Interference which endangers the functioning of a radionavigation service
OR of other safety services OR seriously degrades, obstructs, or repeatedly interrupts
a radiocommunication service operating in accordance with the Radio Regulations.
Policy trigger: This is the threshold that invokes RR No. 4.10.
Action: The US MUST oppose and demand cessation or mitigation.
For ARNS/RNSS: ANY degradation that exceeds the I/N threshold (−6 dB or −10 dB)
constitutes harmful interference to a safety-of-life service (RR 1.59).

PERMISSIBLE INTERFERENCE (RR 1.167)
Definition: Observed or predicted interference which complies with quantitative
interference and sharing criteria contained in the Radio Regulations OR in ITU-R
Recommendations OR in special agreements as provided for in these Regulations.
Policy implication: Acceptable by definition if within agreed criteria.
Used in coordination of frequency assignments between administrations.
Caution: The US must ensure the agreed criteria are conservative enough — 
"permissible" is only as protective as the criteria it references.

ACCEPTED INTERFERENCE (RR 1.168)
Definition: Interference at a higher level than that defined as permissible interference
and which has been AGREED UPON between two or more administrations,
without prejudice to other administrations.
Policy implication: Requires explicit bilateral agreement. The US must ensure
acceptance by one administration does not bind other administrations (including FAA).
Both permissible and accepted interference are used in coordination of frequency
assignments between administrations.

LAYER 3 — TECHNICAL MECHANISMS (physical interference path):
- IN-BAND: Interfering energy falls within the allocated protected band spectrum
- OOB COUPLING: Interfering energy via modulation sidebands enters adjacent protected band
- RECEIVER BLOCKING/DESENSITIZATION: Strong out-of-band signal compresses the LNA,
  raising the effective noise floor — NO spectral overlap required
- INTERMODULATION: Non-linear mixing produces products at new in-band frequencies
- SPURIOUS RESPONSE: Receiver responds at image or IF frequencies due to poor selectivity

CLASSIFICATION DECISION TREE:
Step 1 → What is the emission type? (Spurious or OOB?)
Step 2 → What is the technical mechanism? (In-band, blocking, intermod, OOB coupling?)
Step 3 → What is the impact level? (Harmful 1.169, Permissible 1.167, or Accepted 1.168?)
Step 4 → Does it rise to Harmful under 1.169? → If yes: RR 4.10 applies → OPPOSE
Step 5 → Is it safety-of-life (RR 1.59)? → Lower threshold applies; 6 dB aviation safety factor for precision approach
Step 6 → Apply SPR framework: are worst-case source, path, AND victim parameters used?

For each analysis you MUST produce a formal interference characterization block
that answers all six steps explicitly before proceeding to policy recommendations.
═══════════════════════════════════════════════════════════════════════════

{depth_instruction}

Structure your response with clear headers. Use plain language that a policy official 
can act on immediately. Flag anything requiring urgent escalation to NTIA or ICAO."""

        user_message = f"""Please analyze the following ITU-R contribution and provide FAA-focused policy guidance.

DOCUMENT METADATA:
- Document Number: {doc_number or 'Not provided'}
- Working Party: {working_party}
- Submitting Administration(s): {submitting_admin or 'Not provided'}
- Meeting/Date: {meeting_date or 'Not provided'}
- WRC Agenda Item: {agenda_item or 'Not provided'}
- Document Type: {doc_type}

CONTRIBUTION TEXT:
{contrib_input}

{f"SPECIFIC FAA CONCERN: {user_concern}" if user_concern else ""}
{f"PRIOR US POSITION: {prior_us_position}" if prior_us_position else ""}

Please provide comprehensive policy guidance including:

1. INTERFERENCE CHARACTERIZATION (apply ITU-RR taxonomy + SPR framework)
   a) SPR Analysis: Source worst-case parameters, Path worst-case model, Victim worst-case receiver — are all three applied correctly?
   b) Emission type: Spurious emissions, out-of-band (OOB) emissions, or in-band operation? Cite which RR definition applies.
   c) Interference classification: Harmful (RR 1.169), permissible (RR 1.167), or accepted (RR 1.168)? Justify with quantitative threshold from the system protection table.
   d) Technical mechanism: In-band, adjacent band OOB coupling, receiver blocking/desensitization, intermodulation, or spurious response?
   e) Aviation safety factor: Does the 6 dB precision approach safety factor apply? Is it accounted for in the analysis?
   f) Is RR No. 4.10 triggered? Is this a safety service per RR 1.59? State explicitly.

2. THREAT ASSESSMENT — Which FAA protected bands are at risk and how

3. SUBMITTER'S OBJECTIVE — What the submitting administration is actually trying to achieve

4. TECHNICAL CONCERNS — Specific interference mechanisms and vulnerable aeronautical systems

5. RECOMMENDED US POSITION — Oppose / Support / Propose amendments (with rationale)

6. COUNTER-ARGUMENTS — Technical and regulatory arguments to raise in the Working Party

7. REGULATORY CITATIONS — Specific RR articles, Resolutions, and ITU-R Recommendations to invoke

8. COALITION STRATEGY — Which administrations/organizations to coordinate with

9. REQUIRED ANALYSIS — What technical studies the US should conduct or commission

10. URGENCY & TIMELINE — How quickly must the US respond and through what mechanism

11. DRAFT RESPONSE LANGUAGE — Key phrases/text for the US contribution or intervention,
    including precise interference characterization language using the ITU-RR taxonomy"""

        with st.spinner("Analyzing contribution... this takes 15–30 seconds for deep analysis."):
            try:
                client = anthropic.Anthropic(api_key=api_key)
                response = client.messages.create(
                    model="claude-opus-4-5",
                    max_tokens=4000,
                    messages=[{"role": "user", "content": user_message}],
                    system=system_prompt,
                )
                analysis_text = response.content[0].text

                st.success("✅ Analysis complete")
                st.markdown("---")

                # Interference characterization quick-badge at top
                st.subheader("⚡ Interference Classification (ITU-RR Taxonomy)")
                ex("The AI has characterized the interference using the formal ITU Radio Regulations definitions from RR Articles 1.166–1.169. This framing is what Working Party rapporteurs and experienced delegations use to assess the regulatory weight of a contribution.")

                ic1, ic2, ic3 = st.columns(3)
                with ic1:
                    st.markdown("""
<div style='background:#1a2a3a;border-left:4px solid #4488ff;padding:10px;border-radius:4px'>
<b style='color:#4488ff'>📡 Emission Type</b><br>
<span style='color:#aaddff;font-size:0.85em'>
Spurious (RR 1.145) — hardware non-linearity<br>
Out-of-Band / OOB — from modulation process<br>
In-band — co-frequency operation
</span></div>""", unsafe_allow_html=True)
                with ic2:
                    st.markdown("""
<div style='background:#1a3a1a;border-left:4px solid #44bb44;padding:10px;border-radius:4px'>
<b style='color:#44bb44'>⚖️ RR Classification</b><br>
<span style='color:#aaffaa;font-size:0.85em'>
Harmful (RR 1.169) → triggers RR 4.10<br>
Permissible (RR 1.167) → within criteria<br>
Accepted (RR 1.168) → bilateral agreement
</span></div>""", unsafe_allow_html=True)
                with ic3:
                    st.markdown("""
<div style='background:#3a1a1a;border-left:4px solid #ff4444;padding:10px;border-radius:4px'>
<b style='color:#ff4444'>🔧 Mechanism</b><br>
<span style='color:#ffaaaa;font-size:0.85em'>
In-band / Adjacent band coupling<br>
Receiver blocking / desensitization<br>
Intermodulation / spurious response
</span></div>""", unsafe_allow_html=True)
                st.caption("See full definitions in the 📖 Glossary module under 'ITU-R & Regulatory' category.")
                st.markdown("---")
                st.subheader("Policy Guidance")
                st.markdown(analysis_text)

                # Export
                st.markdown("---")
                export_text = f"""FAA RF INTERFERENCE TOOL — CONTRIBUTION ANALYSIS
Generated: {meeting_date or 'N/A'}
Document: {doc_number or 'N/A'} | WP: {working_party} | Admin: {submitting_admin or 'N/A'}
Agenda Item: {agenda_item or 'N/A'}
Analysis Depth: {analysis_depth}

{'='*60}
CONTRIBUTION TEXT (INPUT)
{'='*60}
{contrib_input}

{'='*60}
AI POLICY ANALYSIS
{'='*60}
{analysis_text}
"""
                st.download_button(
                    "⬇️ Download Full Analysis as .txt",
                    export_text,
                    file_name=f"policy_analysis_{doc_number.replace('/', '_') if doc_number else 'contribution'}.txt",
                    mime="text/plain"
                )

            except anthropic.AuthenticationError:
                st.error("❌ Invalid API key. Check your Streamlit secrets configuration.")
            except anthropic.RateLimitError:
                st.error("❌ API rate limit reached. Wait a moment and try again.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    elif not contrib_input.strip():
        st.info("👆 Paste a contribution above and click Analyze to get policy guidance.")

    # Example contributions to try
    with st.expander("📋 Example contributions to test with"):
        st.markdown("""
**Example 1 — IMT identification near radio altimeter band (WP 5D)**
> *"This contribution proposes to identify the frequency range 4800–4990 MHz for IMT under WRC-27 Agenda Item 1.2. 
Sharing studies indicate that IMT base stations operating with EIRP not exceeding 58 dBm and using downtilted antennas 
can coexist with existing services. The submitting administrations request that the Working Party develop a draft 
CPM Report method supporting IMT identification in this band."*

**Example 2 — VDES proposal near aeronautical band (WP 5B)**
> *"This document proposes expansion of the VHF Data Exchange System (VDES) to include additional channels 
in the 156–174 MHz band to support maritime e-navigation. The proposal includes shore-based transmitters 
with power levels up to 50W ERP. Compatibility with aeronautical VHF communications has not been fully assessed."*

**Example 3 — LEO satellite downlinks (WP 4A / SG 5)**
> *"The proposed non-GSO satellite constellation will operate downlinks in the 1525–1559 MHz band with 
aggregate EPFD levels approaching the limits specified in RR Appendix 7. Individual satellite EIRP is 
limited to 15 dBW per beam with a minimum elevation angle of 10 degrees."*
        """)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 9 — RF TRAINING
# ─────────────────────────────────────────────────────────────────────────────
elif selected_tab == "🎓 RF Training":
    st.title("🎓 RF Fundamentals Training")
    st.markdown("*Ground-up refresher designed for engineers returning to RF after time away — focused on what you need for ITU-R policy support.*")
    ex("Work through these lessons in order. Each builds on the last. Every concept has a worked example and a self-check.")

    lesson = st.selectbox("Select Lesson:", [
        "Lesson 1 — The Decibel: Your Most Important Tool",
        "Lesson 2 — Frequency, Wavelength & The EM Spectrum",
        "Lesson 3 — Transmit Power, EIRP & Antenna Gain",
        "Lesson 4 — Free Space Path Loss",
        "Lesson 5 — The Link Budget: Putting It All Together",
        "Lesson 6 — Noise, Sensitivity & the Noise Floor",
        "Lesson 7 — Interference: I/N, C/I & Protection Criteria",
        "Lesson 8 — Propagation Models: P.452, P.528, P.676",
        "Lesson 9 — Monte Carlo & Aggregate Interference",
        "Lesson 10 — From RF Math to ITU-R Policy",
    ])

    st.markdown("---")

    # ── LESSON 1 ──────────────────────────────────────────────────────────────
    if lesson.startswith("Lesson 1"):
        st.header("📐 Lesson 1 — The Decibel: Your Most Important Tool")
        ex("The dB scale is log base 10 of a power ratio multiplied by 10. Because the Friis equation is multiplicative in linear scale, it becomes additive in dB — this is why every RF link analysis is a simple arithmetic chain of dB values.")

        st.subheader("Why Decibels?")
        st.markdown("""
RF signals span an enormous range — a transmitter might produce 100 watts while a GPS receiver needs just 
0.000000000000001 watts (10⁻¹⁵ W). Writing those numbers linearly is unworkable. Decibels compress 
this range into manageable numbers by using logarithms.

**The core formula:**
""")
        st.latex(r"\text{Value (dB)} = 10 \cdot \log_{10}\left(\frac{P_2}{P_1}\right)")
        st.markdown("""
This compares two power levels. When you say a signal is "30 dB stronger," you mean it is **1000× more powerful**.

**The three numbers you must memorize:**

| dB | Power Ratio | Memory Hook |
|---|---|---|
| **+3 dB** | 2× (double) | "3 dB = double the power" |
| **+10 dB** | 10× | "10 dB = ten times the power" |
| **−3 dB** | 0.5× (half) | "−3 dB = half the power" |
| **−10 dB** | 0.1× | "−10 dB = one tenth the power" |
| **+30 dB** | 1000× | Combine: 10 × 10 × 10 |
| **−60 dB** | 0.000001× | Combine: −10 − 10 − 10 − 10 − 10 − 10 |

**In RF we use two reference points:**
- **dBm** = decibels relative to 1 milliwatt. Used for signal power.
- **dBi** = decibels relative to an isotropic antenna. Used for antenna gain.
- **dBW** = decibels relative to 1 watt. Used in satellite/high-power work (0 dBW = 30 dBm).
        """)

        st.subheader("Converting Between Watts and dBm")
        st.latex(r"P_{\text{dBm}} = 10 \cdot \log_{10}(P_{\text{mW}})")
        st.latex(r"P_{\text{mW}} = 10^{P_{\text{dBm}}/10}")

        st.markdown("**Common conversions to memorize:**")
        conv_data = {
            "Power (Watts)": ["0.001 W (1 mW)", "0.01 W (10 mW)", "0.1 W (100 mW)", "1 W", "10 W", "20 W", "100 W"],
            "Power (dBm)": ["0 dBm", "10 dBm", "20 dBm", "30 dBm", "40 dBm", "43 dBm", "50 dBm"],
            "Typical Use": ["Reference level", "Low-power IoT device", "Wi-Fi access point", "Small radio", "Mobile phone max", "LTE base station typical", "FM broadcast transmitter"],
        }
        st.table(pd.DataFrame(conv_data))

        st.subheader("🔢 Interactive dB Calculator")
        ex("Note that dBm is an absolute power level (referenced to 1 mW), while dB is a dimensionless ratio. You can add dB values to dBm, but you cannot add two dBm values — that would require linear addition of milliwatts, then convert back.")
        col1, col2 = st.columns(2)
        with col1:
            p_watts = st.number_input("Power (Watts)", value=1.0, min_value=0.000001, format="%.6f")
            p_dbm_calc = 10 * np.log10(p_watts * 1000)
            st.metric("In dBm", f"{p_dbm_calc:.2f} dBm")
            st.metric("In dBW", f"{p_dbm_calc - 30:.2f} dBW")
        with col2:
            p_dbm_in = st.number_input("Power (dBm)", value=30.0, step=1.0)
            p_watts_calc = 10 ** (p_dbm_in / 10) / 1000
            st.metric("In Watts", f"{p_watts_calc:.6f} W")
            st.metric("In milliwatts", f"{p_watts_calc*1000:.4f} mW")

        st.subheader("The Golden Rule of dB Arithmetic")
        st.markdown("""
In dB, **multiplication becomes addition and division becomes subtraction.**
This is why link budgets work — instead of multiplying and dividing many large/small numbers,
you just add and subtract dB values.

**Example:**
> A transmitter puts out 43 dBm. It goes through a 3 dB cable loss, then a 15 dBi antenna.
> What is the EIRP?
>
> EIRP = 43 − 3 + 15 = **55 dBm** ✓
>
> (In linear: 20W × 0.5 × 31.6 = 316W = 55 dBm. Same answer, harder math.)
        """)

        st.subheader("✅ Self-Check")
        with st.expander("Try these — click to reveal answers"):
            st.markdown("""
**Q1:** A signal drops by 20 dB. By what factor did the power decrease?
> **Answer:** 100× decrease (−10 dB = ÷10, twice = ÷100)

**Q2:** A GPS satellite transmits at 27 dBW. What is that in dBm? In Watts?
> **Answer:** 27 dBW = 57 dBm = 500 Watts

**Q3:** You have 43 dBm Tx power, 2 dB cable loss, 17 dBi antenna gain. What is the EIRP?
> **Answer:** 43 − 2 + 17 = **58 dBm** (631 Watts equivalent)

**Q4:** A receiver needs −90 dBm to work. The received signal is −105 dBm. By how many dB is it below threshold?
> **Answer:** −105 − (−90) = **−15 dB** below threshold (about 30× too weak)
            """)

    # ── LESSON 2 ──────────────────────────────────────────────────────────────
    elif lesson.startswith("Lesson 2"):
        st.header("📻 Lesson 2 — Frequency, Wavelength & The EM Spectrum")
        ex("The free-space path loss equation has a 20·log10(f) term — doubling frequency adds 6 dB of path loss at fixed distance. This frequency dependence is why UHF/L-band systems like GPS require extremely sensitive receivers to overcome orbital path losses exceeding 180 dB.")

        st.subheader("The Fundamental Relationship")
        st.latex(r"c = f \cdot \lambda \quad \Rightarrow \quad \lambda = \frac{c}{f}")
        st.markdown("""
Where:
- **c** = speed of light = 3 × 10⁸ m/s
- **f** = frequency in Hz
- **λ** = wavelength in meters

**Quick rule:** λ(meters) ≈ 300 / f(MHz)
        """)

        st.subheader("🔢 Frequency ↔ Wavelength Calculator")
        col1, col2 = st.columns(2)
        with col1:
            f_mhz_l2 = st.number_input("Frequency (MHz)", value=1090.0, step=10.0)
            lam = 300 / f_mhz_l2
            st.metric("Wavelength", f"{lam*100:.2f} cm")
            st.metric("Half-wavelength (antenna)", f"{lam/2*100:.2f} cm")
        with col2:
            lam_cm = st.number_input("Wavelength (cm)", value=27.5, step=1.0)
            f_from_lam = 300 / (lam_cm / 100)
            st.metric("Frequency", f"{f_from_lam:.1f} MHz")

        st.subheader("The Aeronautical Spectrum — Bands You Work With")
        ex("The key parameter is the ratio of wavelength to obstacle size: wavelength < obstacle → shadowing/reflection dominate; wavelength ≈ obstacle → diffraction and scattering matter. At C-band (λ ≈ 7 cm), rain drops scatter efficiently — hence airborne weather radar operates here.")
        band_data = pd.DataFrame([
            ["VHF", "30–300 MHz", "~1–10 m", "Line-of-sight + some diffraction", "VOR, ILS, VHF comms"],
            ["UHF", "300 MHz–3 GHz", "~10 cm–1 m", "Line-of-sight dominant", "DME, TACAN, ADS-B, GPS L1"],
            ["L-band", "1–2 GHz", "~15–30 cm", "Good penetration, long range", "GPS, GLONASS, DME, Mode-S"],
            ["C-band", "4–8 GHz", "~4–7 cm", "Line-of-sight, rain starts to matter", "Radio altimeter, en-route radar"],
            ["X-band", "8–12 GHz", "~2.5–4 cm", "Rain attenuation significant", "Weather radar, surface radar"],
        ], columns=["Band", "Frequency Range", "Wavelength", "Propagation Character", "FAA Systems"])
        st.table(band_data)

        st.subheader("Why Frequency Matters for Interference")
        st.markdown("""
**1. Path Loss scales with frequency** — higher frequency = more free-space path loss per km
(we'll calculate this in Lesson 4). This can work in your favor — a 5 GHz interferer loses
more power over distance than a 500 MHz one.

**2. Antenna size scales with wavelength** — a half-wave dipole at 1090 MHz (ADS-B) is
~13.7 cm. At 121.5 MHz (VHF emergency) it's ~1.23 m. This affects what fits on an aircraft.

**3. Atmospheric absorption** — above ~10 GHz, oxygen and water vapor absorb RF energy.
At 60 GHz, absorption is so high signals can barely travel a few km (used in 5G mmWave
for this reason — natural isolation).

**4. Rain attenuation** — above ~3 GHz, raindrops are comparable to the wavelength
and scatter energy. Airborne weather radar at 9 GHz exploits this to detect storms.
        """)

        st.subheader("✅ Self-Check")
        with st.expander("Click to reveal answers"):
            st.markdown("""
**Q1:** ADS-B operates at 1090 MHz. What is its wavelength?
> **Answer:** λ = 300/1090 = **0.275 m = 27.5 cm**

**Q2:** GPS L1 is at 1575.42 MHz. GPS L5 is at 1176.45 MHz. Which has a longer wavelength?
> **Answer:** L5 (lower frequency = longer wavelength). L5: 25.5 cm vs L1: 19.0 cm

**Q3:** Why does the radio altimeter band (4200–4400 MHz) require more careful protection from nearby 5G (3700–3980 MHz) than, say, VHF comms at 121 MHz?
> **Answer:** The 5G band is only ~220 MHz away from the radio altimeter band. At these frequencies, receiver front-ends can be overloaded by strong adjacent signals even if the 5G signal is not technically "in-band." At VHF, the fractional frequency separation would be enormous.
            """)

    # ── LESSON 3 ──────────────────────────────────────────────────────────────
    elif lesson.startswith("Lesson 3"):
        st.header("📡 Lesson 3 — Transmit Power, EIRP & Antenna Gain")
        ex("EIRP is the product of transmit power and antenna gain in the direction of the victim, less cable losses. It is invariant to whether gain is achieved by increasing power or focusing the antenna — both create identical interference at a distant receiver.")

        st.subheader("Transmit Power")
        st.markdown("""
Transmit power (Pt) is the RF power delivered to the antenna input. It does NOT account for the antenna's directionality.

Common values in ITU-R work:
| System | Typical Tx Power |
|---|---|
| LTE macro base station | 43–46 dBm (20–40W) |
| 5G NR base station | 46–53 dBm (40–200W) |
| Mobile handset | 23–30 dBm (200mW–1W) |
| VOR ground station | 50 dBm (100W) |
| GPS satellite | 27 dBW = 57 dBm (500W) |
| ADS-B transponder | 18–21 dBW = 48–51 dBm |
        """)

        st.subheader("Antenna Gain")
        st.markdown("""
An antenna does not amplify power — it **redirects** it. Gain (G) describes how much more power 
is concentrated in the main beam versus an isotropic (perfectly omnidirectional) antenna.

- **0 dBi** — isotropic; radiates equally in all directions (theoretical)
- **2 dBi** — simple dipole; slight directivity
- **15 dBi** — typical sector antenna on a cell tower
- **30+ dBi** — high-gain dish or phased array

**Key insight for interference analysis:** When assessing interference, use the antenna gain 
**in the direction of the victim receiver** — not the peak gain. For worst-case analysis, 
assume 0 dBi (isotropic) toward the victim unless you have a specific antenna pattern.
        """)

        st.subheader("EIRP — Effective Isotropic Radiated Power")
        st.latex(r"\text{EIRP (dBm)} = P_t \text{(dBm)} + G_t \text{(dBi)} - L_{cable} \text{(dB)}")
        st.markdown("""
EIRP represents the power that would need to be fed into an isotropic antenna to produce 
the same signal strength in the direction of maximum radiation.

**It is the standard metric used in:**
- ITU-R Radio Regulations (PFD and EIRP limits)
- Coordination agreements between administrations
- Interference analysis inputs
        """)

        st.subheader("🔢 Interactive EIRP Calculator")
        col1, col2, col3 = st.columns(3)
        with col1:
            pt = st.number_input("Tx Power (dBm)", value=43.0, step=1.0)
        with col2:
            gt = st.number_input("Antenna Gain (dBi)", value=15.0, step=1.0)
        with col3:
            lc = st.number_input("Cable Loss (dB)", value=2.0, step=0.5)
        eirp_val = pt + gt - lc
        st.metric("EIRP", f"{eirp_val:.1f} dBm = {10**(eirp_val/10)/1000:.1f} W equivalent")
        ex("In ITU-R Radio Regulations, EIRP limits are specified per carrier, per antenna beam, or as aggregate — always check which definition applies. A proponent quoting per-carrier EIRP for a MIMO system may be understating the effective aggregate EIRP toward your victim.")

        st.subheader("Power Flux Density (PFD)")
        st.markdown("""
PFD describes how much power arrives per unit area at a given distance. 
ITU-R uses PFD limits in the Radio Regulations to protect Earth-based receivers from satellites.
        """)
        st.latex(r"\text{PFD (dBW/m}^2) = \text{EIRP (dBW)} - 10\log_{10}(4\pi d^2)")

        d_km_l3 = st.slider("Distance (km)", 1, 500, 10)
        eirp_dbw = eirp_val - 30
        pfd_val = eirp_dbw - 10 * np.log10(4 * np.pi * (d_km_l3 * 1000) ** 2)
        st.metric("PFD at distance", f"{pfd_val:.1f} dBW/m²")
        ex("PFD (W/m²) is power per unit area at the victim, independent of the victim's antenna aperture. RR Appendix 7 PFD limits are specified by elevation angle — the lower the elevation, the tighter the limit, because low-elevation satellite paths have longer atmospheric path lengths.")

        st.subheader("✅ Self-Check")
        with st.expander("Click to reveal answers"):
            st.markdown("""
**Q1:** A 5G base station has Tx power 46 dBm, antenna gain 18 dBi, cable loss 1 dB. What is the EIRP?
> **Answer:** 46 + 18 − 1 = **63 dBm** (2000W equivalent — this is why 5G can interfere with things far away)

**Q2:** For worst-case interference analysis, should you use the base station's peak antenna gain or 0 dBi toward the victim?
> **Answer:** Use the gain **toward the victim** — which may be a side lobe. For a worst-case bound, use 0 dBi. In a realistic analysis, model the actual antenna pattern.

**Q3:** Why is EIRP used instead of just transmit power in regulatory limits?
> **Answer:** EIRP captures both the transmitter power AND the antenna focusing effect. Two transmitters with the same Tx power but different antennas can have very different interference impacts — EIRP accounts for this in a single number.
            """)

    # ── LESSON 4 ──────────────────────────────────────────────────────────────
    elif lesson.startswith("Lesson 4"):
        st.header("📉 Lesson 4 — Free Space Path Loss")
        ex("FSPL arises from the inverse square law: power density decreases as 1/4πr². The 20·log10(d) + 20·log10(f) form captures both the geometric spreading and the frequency dependence of effective antenna aperture (Ae = λ²G/4π).")

        st.subheader("The Formula")
        st.latex(r"FSPL \text{ (dB)} = 20\log_{10}(d_{km}) + 20\log_{10}(f_{MHz}) + 32.44")
        st.markdown("""
This is the **Friis free-space path loss** equation. It tells you how much signal is lost 
between a transmitter and receiver separated by distance d, at frequency f.

**Key observations:**
- **Distance doubles → path loss increases by 6 dB** (power drops to ¼)
- **Frequency doubles → path loss increases by 6 dB**
- Higher frequency signals lose more power over the same distance
- This is WHY 5G mmWave (26 GHz) has short range and GPS (1.5 GHz) needs very sensitive receivers
        """)

        st.subheader("🔢 Interactive FSPL Calculator")
        col1, col2 = st.columns(2)
        with col1:
            f_l4 = st.number_input("Frequency (MHz)", value=1090.0, step=10.0, key="l4f")
            d_l4 = st.number_input("Distance (km)", value=10.0, step=1.0, key="l4d")
        fspl_val = 20*np.log10(d_l4) + 20*np.log10(f_l4) + 32.44
        with col2:
            st.metric("Free Space Path Loss", f"{fspl_val:.1f} dB")
            st.metric("Signal reduced by factor of", f"{10**(fspl_val/10):.2e}")
        ex("FSPL is the dominant loss term for most aeronautical links — it typically ranges from 100 dB (VHF, short range) to 190 dB (GPS orbital distance). All other loss mechanisms (atmospheric, terrain, clutter) are additive corrections to this baseline.")

        st.subheader("FSPL vs Distance — Interactive Chart")
        freqs_to_plot = st.multiselect("Select frequencies to compare:",
            [121.5, 328.6, 1090, 1575, 4300, 9375],
            default=[1090, 1575, 4300],
            format_func=lambda x: f"{x} MHz")

        if freqs_to_plot:
            d_range = np.linspace(0.1, 100, 300)
            fig_l4, ax_l4 = plt.subplots(figsize=(10, 5))
            fig_l4.patch.set_facecolor("#0e1117")
            ax_l4.set_facecolor("#0e1117")
            colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(freqs_to_plot)))
            for freq, col in zip(freqs_to_plot, colors):
                fspl_curve = 20*np.log10(d_range) + 20*np.log10(freq) + 32.44
                ax_l4.plot(d_range, fspl_curve, color=col, linewidth=2, label=f"{freq} MHz")
            ax_l4.set_xlabel("Distance (km)", color='white')
            ax_l4.set_ylabel("Path Loss (dB)", color='white')
            ax_l4.set_title("Free Space Path Loss vs Distance", color='white')
            ax_l4.legend(facecolor='#1a1a2e', labelcolor='white')
            ax_l4.tick_params(colors='white')
            ax_l4.grid(color='#333', alpha=0.5)
            for sp in ax_l4.spines.values(): sp.set_color('#444')
            plt.tight_layout()
            st.pyplot(fig_l4)

        st.subheader("Why FSPL is Just the Starting Point")
        st.markdown("""
FSPL assumes a perfect vacuum with no obstructions. Real paths add loss from:

| Additional Loss Mechanism | Typical Extra Loss | When It Applies |
|---|---|---|
| Atmospheric gases (O₂, H₂O) | 0.01–10 dB/km | Above ~1 GHz, long paths |
| Rain attenuation | 0.01–10 dB/km | Above ~3 GHz |
| Building/terrain diffraction | 10–30 dB | Ground paths, urban |
| Vegetation/foliage | 5–20 dB | Low-altitude paths |
| Multipath fading | ±20 dB | Reflective environments |

**For interference analysis:** FSPL gives you the **optimistic** interference estimate 
(most interference at the victim). If even FSPL shows compatibility, you're safe.
If FSPL fails, use P.452 or P.528 for a more realistic (higher loss) calculation.
        """)

        st.subheader("✅ Self-Check")
        with st.expander("Click to reveal answers"):
            st.markdown("""
**Q1:** Calculate FSPL at 4300 MHz, 5 km.
> **Answer:** 20·log(5) + 20·log(4300) + 32.44 = 13.98 + 72.67 + 32.44 = **119.1 dB**

**Q2:** If you double the distance, how much does FSPL increase?
> **Answer:** 20·log(2) = **6 dB** increase

**Q3:** GPS signal arrives at ~−130 dBm at a receiver. GPS satellites transmit at ~57 dBm EIRP. 
What is the approximate path loss over the ~20,200 km orbital distance at 1575 MHz?
> **Answer:** FSPL = 20·log(20200) + 20·log(1575) + 32.44 ≈ 86.1 + 63.9 + 32.4 = **182.4 dB**. 
This is why GPS receivers are so sensitive — and why any interference is so damaging.
            """)

    # ── LESSON 5 ──────────────────────────────────────────────────────────────
    elif lesson.startswith("Lesson 5"):
        st.header("🔗 Lesson 5 — The Link Budget: Putting It All Together")
        ex("The Friis transmission equation: Pr(dBm) = Pt + Gt − Lcable − FSPL + Gr. For interference analysis, Pr is the interference power at the victim input — compare it to the noise floor (N = kTBF) using I/N = Pr − N to assess compatibility.")

        st.subheader("The Friis Transmission Equation")
        st.latex(r"P_r = P_t + G_t - L_{cable} - FSPL + G_r \quad \text{(all in dB)}")
        st.markdown("""
| Term | Symbol | Typical Range | Description |
|---|---|---|---|
| Received power | Pr | −60 to −130 dBm | What arrives at receiver input |
| Transmit power | Pt | 20 to 53 dBm | PA output power |
| Tx antenna gain | Gt | 0 to 30 dBi | Toward receiver |
| Cable/feeder loss | Lcable | 0.5 to 5 dB | Physical losses in hardware |
| Path loss | FSPL | 60 to 200 dB | Distance + frequency dependent |
| Rx antenna gain | Gr | 0 to 30 dBi | Toward transmitter |
        """)

        st.subheader("🔢 Build a Link Budget Step by Step")
        ex("Sensitivity analysis: vary one parameter at a time to identify which term most strongly drives the I/N result. If path loss dominates, argue for a more realistic propagation model. If EIRP dominates, argue for a PFD or EIRP limit in the regulatory text.")

        col1, col2 = st.columns(2)
        with col1:
            lb_pt = st.number_input("① Tx Power (dBm)", value=43.0, step=1.0, key="lb1")
            lb_gt = st.number_input("② Tx Antenna Gain (dBi)", value=15.0, step=1.0, key="lb2")
            lb_lc = st.number_input("③ Cable Loss (dB)", value=2.0, step=0.5, key="lb3")
            lb_f  = st.number_input("④ Frequency (MHz)", value=4300.0, step=100.0, key="lb4")
            lb_d  = st.number_input("⑤ Distance (km)", value=10.0, step=1.0, key="lb5")
            lb_gr = st.number_input("⑥ Rx Antenna Gain (dBi)", value=0.0, step=1.0, key="lb6")

        lb_eirp = lb_pt + lb_gt - lb_lc
        lb_fspl = 20*np.log10(lb_d) + 20*np.log10(lb_f) + 32.44
        lb_pr = lb_eirp - lb_fspl + lb_gr

        with col2:
            st.markdown("**Running Total:**")
            st.markdown(f"① Tx Power: **{lb_pt} dBm**")
            st.markdown(f"② + Tx Gain: **+{lb_gt} dBi** → {lb_pt+lb_gt:.1f} dBm")
            st.markdown(f"③ − Cable Loss: **−{lb_lc} dB** → {lb_eirp:.1f} dBm (EIRP)")
            st.markdown(f"④⑤ − Path Loss ({lb_f:.0f}MHz, {lb_d}km): **−{lb_fspl:.1f} dB** → {lb_eirp-lb_fspl:.1f} dBm")
            st.markdown(f"⑥ + Rx Gain: **+{lb_gr} dBi** → **{lb_pr:.1f} dBm** ← Received Power")
            st.metric("Received Power", f"{lb_pr:.1f} dBm")

        st.subheader("The Interference Link Budget")
        st.markdown("""
When you're assessing interference (not a wanted signal), the link budget is the same —
but instead of comparing received power to a minimum detectable signal,
you compare it to the **noise floor** using the I/N criterion:
        """)
        st.latex(r"\frac{I}{N} \text{ (dB)} = P_{interference} \text{ (dBm)} - \text{Noise Floor (dBm)}")
        st.markdown("""
**Protection is maintained when:** I/N < threshold (typically −6 dB or −10 dB)

**Protection margin** = threshold − I/N  
- Positive margin → protected  
- Negative margin → interference threshold exceeded → you have grounds to object
        """)

        lb_nf_bw = st.number_input("Rx Bandwidth (MHz)", value=100.0, step=10.0)
        lb_nf_nf = st.number_input("Rx Noise Figure (dB)", value=5.0, step=0.5)
        lb_thresh = st.number_input("I/N Threshold (dB)", value=-6.0, step=1.0)
        lb_noise = -174 + 10*np.log10(lb_nf_bw*1e6) + lb_nf_nf
        lb_in = lb_pr - lb_noise
        lb_margin = lb_thresh - lb_in

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Noise Floor", f"{lb_noise:.1f} dBm")
        col_b.metric("I/N", f"{lb_in:.1f} dB")
        col_c.metric("Protection Margin", f"{lb_margin:.1f} dB",
                     delta_color="normal" if lb_margin >= 0 else "inverse")

        if lb_margin < 0:
            warn(f"Threshold violated by {abs(lb_margin):.1f} dB — at {lb_d} km this interferer exceeds the protection criterion.")
        else:
            ok(f"Protected with {lb_margin:.1f} dB margin at {lb_d} km separation.")

        st.subheader("✅ Self-Check")
        with st.expander("Click to reveal answers"):
            st.markdown("""
**Q1:** A 5G base station (EIRP 58 dBm) is 5 km from a radio altimeter receiver (4300 MHz, noise floor −89 dBm, I/N threshold −6 dB). Is there an interference problem?
> FSPL = 20·log(5) + 20·log(4300) + 32.44 = 14.0 + 72.7 + 32.4 = **119.1 dB**
> Received power = 58 − 119.1 + 0 = **−61.1 dBm**
> I/N = −61.1 − (−89) = **+27.9 dB**
> Threshold = −6 dB. Margin = −6 − 27.9 = **−33.9 dB** ← Severely violated!
> This is exactly the 5G/rad-alt interference problem that grounded aircraft in 2022.

**Q2:** What separation distance would be needed for I/N ≤ −6 dB?
> Need: received power ≤ noise floor + threshold = −89 + (−6) = −95 dBm
> Need FSPL ≥ 58 − (−95) = 153 dB
> 153 = 20·log(d) + 72.7 + 32.4 → 20·log(d) = 47.9 → d = 10^(47.9/20) = **248 km!**
            """)

    # ── LESSON 6 ──────────────────────────────────────────────────────────────
    elif lesson.startswith("Lesson 6"):
        st.header("📊 Lesson 6 — Noise, Sensitivity & the Noise Floor")
        ex("The receiver noise floor N = kTB·NF is a physical lower bound set by thermodynamics. Interference raises the effective noise floor to N·(1 + I/N_linear), degrading SNR for the desired signal by the same amount — this is why I/N is the correct metric.")

        st.subheader("Thermal Noise — The Unavoidable Baseline")
        st.markdown("""
All electronic components generate noise due to random thermal motion of electrons.
This sets the absolute minimum noise any real receiver will experience.
        """)
        st.latex(r"N_{thermal} = k \cdot T \cdot B")
        st.latex(r"N_{thermal} \text{ (dBm)} = -174 + 10\log_{10}(B_{Hz}) \quad \text{at 290K}")
        st.markdown("""
Where:
- **k** = Boltzmann's constant = 1.38 × 10⁻²³ J/K
- **T** = temperature in Kelvin (290K = room temperature standard)
- **B** = bandwidth in Hz
- **−174 dBm/Hz** is the thermal noise spectral density at 290K — memorize this number
        """)

        st.subheader("🔢 Noise Floor Calculator")
        col1, col2 = st.columns(2)
        with col1:
            bw_l6 = st.number_input("Receiver Bandwidth (MHz)", value=100.0, min_value=0.001, step=10.0)
            nf_l6 = st.number_input("Noise Figure (dB)", value=5.0, step=0.5,
                help="Real receivers amplify the noise — NF captures how much")
            temp_l6 = st.number_input("Temperature (K)", value=290.0, step=10.0)
        kT_l6 = 10*np.log10(1.38e-23 * temp_l6) + 30
        nf_total = kT_l6 + 10*np.log10(bw_l6*1e6) + nf_l6
        with col2:
            st.metric("kT noise density", f"{kT_l6:.1f} dBm/Hz")
            st.metric("Noise floor (kTB)", f"{kT_l6 + 10*np.log10(bw_l6*1e6):.1f} dBm")
            st.metric("Noise floor (kTBNF)", f"{nf_total:.1f} dBm")
        ex("Noise floor = −174 + 10·log10(B) + NF. A GPS L1 receiver (2 MHz BW, 2 dB NF) has a noise floor of −174 + 63 + 2 = −109 dBm. A radio altimeter (200 MHz BW, 5 dB NF) has −174 + 83 + 5 = −86 dBm — the wider bandwidth raises the floor by 23 dB.")

        st.subheader("Noise Figure")
        st.markdown("""
No real receiver is perfect — its internal components (LNA, mixer, filters) add noise.
The **noise figure (NF)** measures how much worse the real receiver is compared to an ideal one.

| Receiver Type | Typical NF | Why |
|---|---|---|
| GPS/GNSS LNA | 1–2 dB | Extremely sensitive; expensive low-noise design |
| Radio Altimeter | 4–6 dB | Aviation grade; well-designed |
| ADS-B receiver | 4–8 dB | Varies by implementation |
| General avionic | 5–10 dB | Depends on age and design |
| Consumer Wi-Fi | 8–12 dB | Cost-optimized |

A 1 dB increase in noise figure directly raises the noise floor by 1 dB —
and reduces sensitivity by 1 dB. This is why avionics engineers fight hard for low NF.
        """)

        st.subheader("Sensitivity vs Selectivity")
        st.markdown("""
**Sensitivity** — minimum signal the receiver can detect above the noise floor.
(Usually defined as SNR = some threshold, e.g., 10 dB above noise floor)

**Selectivity** — how well the receiver rejects signals on adjacent or nearby frequencies.
Determined by the filter characteristics (bandwidth, roll-off, out-of-band rejection).

**Why this matters for interference:**
- A strong out-of-band signal can **overload** (block/desensitize) the receiver front-end 
  even if the filter should reject it — the amplifier saturates before the filter acts.
- This was the exact mechanism in the 5G/radio altimeter problem:
  5G at 3.8 GHz was outside the 4.2–4.4 GHz altimeter band, but the 5G signal was
  strong enough to drive the altimeter's LNA into compression, raising its noise floor
  and killing sensitivity across the whole band.
- **In your contributions:** distinguish between **in-band interference** (signal inside 
  the protected band) and **receiver blocking/desensitization** (out-of-band signal 
  overloading the front end). Both are harmful; blocking is often harder to defend against.
        """)

        st.subheader("✅ Self-Check")
        with st.expander("Click to reveal answers"):
            st.markdown("""
**Q1:** Calculate the noise floor for a radio altimeter with 200 MHz bandwidth and 5 dB noise figure.
> −174 + 10·log(200×10⁶) + 5 = −174 + 83.0 + 5 = **−86 dBm**

**Q2:** If you narrow the bandwidth from 200 MHz to 20 MHz, what happens to the noise floor?
> 10·log(20M/200M) = −10 dB. Noise floor drops to **−96 dBm**. 
> Narrower bandwidth = lower noise floor = better sensitivity. But you also reject more of the desired signal.

**Q3:** What is the difference between a receiver being "jammed" and being "desensitized"?
> **Jamming** = an in-band interferer that directly competes with the desired signal.
> **Desensitization/blocking** = a strong out-of-band signal overloads the front-end amplifier, 
> raising the effective noise floor and reducing sensitivity to the desired signal. 
> The interferer doesn't need to be in-band to cause this — it just needs to be strong enough.
            """)

    # ── LESSON 7 ──────────────────────────────────────────────────────────────
    elif lesson.startswith("Lesson 7"):
        st.header("⚡ Lesson 7 — Interference: I/N, C/I & Protection Criteria")
        ex("I/N = I(dBm) − N(dBm). The noise floor rise ΔN = 10·log10(1 + 10^(I/N/10)). For I/N = −6 dB: ΔN = 0.97 dB. For I/N = 0 dB: ΔN = 3 dB. The ITU-R thresholds (−6 dB, −10 dB) correspond to ΔN < 1 dB for safety-critical systems.")

        st.subheader("The Three Key Ratios")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**I/N — Interference to Noise**")
            st.latex(r"\frac{I}{N} \text{(dB)} = I_{\text{dBm}} - N_{\text{floor dBm}}")
            st.markdown("Used in: ITU-R protection criteria, most aeronautical compatibility studies")
        with col2:
            st.markdown("**C/I — Carrier to Interference**")
            st.latex(r"\frac{C}{I} \text{(dB)} = C_{\text{dBm}} - I_{\text{dBm}}")
            st.markdown("Used in: Communications system design, co-channel interference assessment")
        with col3:
            st.markdown("**C/N — Carrier to Noise**")
            st.latex(r"\frac{C}{N} \text{(dB)} = C_{\text{dBm}} - N_{\text{floor dBm}}")
            st.markdown("Used in: Link quality assessment, minimum SNR for operation")

        st.subheader("Why I/N and Not Just 'Is the Interference Below the Noise'?")
        st.markdown("""
The noise floor is not a hard wall — it's a statistical level. Adding an interferer 
**raises the effective noise floor** by:

| I/N | Noise floor increase | Effect |
|---|---|---|
| −10 dB | +0.41 dB | Barely noticeable — standard GNSS criterion |
| −6 dB | +0.97 dB | ~1 dB degradation — standard ARNS criterion |
| 0 dB | +3.0 dB | Significant degradation |
| +10 dB | +10.4 dB | Receiver largely unusable |

Setting I/N ≤ −6 dB means the interference raises the noise floor by less than 1 dB —
a small, tolerable degradation for safety-of-life systems.

Setting I/N ≤ −10 dB for GNSS is more conservative because GPS signals arrive at 
only ~−130 dBm — any noise floor increase materially degrades position accuracy.
        """)

        st.subheader("🔢 Interactive I/N Explorer")
        col1, col2 = st.columns(2)
        with col1:
            i_dbm = st.number_input("Interference Power (dBm)", value=-100.0, step=1.0)
            n_dbm = st.number_input("Noise Floor (dBm)", value=-89.0, step=1.0)
            thresh_l7 = st.number_input("I/N Threshold (dB)", value=-6.0, step=1.0)
        i_n_l7 = i_dbm - n_dbm
        margin_l7 = thresh_l7 - i_n_l7
        noise_rise = 10*np.log10(1 + 10**(i_n_l7/10))
        with col2:
            st.metric("I/N", f"{i_n_l7:.1f} dB")
            st.metric("Protection Margin", f"{margin_l7:.1f} dB",
                      delta_color="normal" if margin_l7 >= 0 else "inverse")
            st.metric("Effective Noise Floor Rise", f"+{noise_rise:.2f} dB")

        st.subheader("Protection Criteria by System")
        ex("RTCA Minimum Operational Performance Standards (MOPS) define receiver sensitivity and selectivity under interference. The I/N thresholds are derived from MOPS by determining the interference level that degrades receiver output (accuracy, availability, integrity) beyond acceptable limits.")
        prot_data = pd.DataFrame([
            ["GPS/GNSS L1/L5", "−10 dB", "DO-235B / DO-253", "Satellite signal extremely weak (~−130 dBm)"],
            ["ADS-B / Mode-S", "−10 dB", "DO-260B", "False target / missed detection consequences"],
            ["Radio Altimeter", "−6 dB", "DO-155 / ETSO-C87", "CAT III landing; 1 dB degradation max"],
            ["ILS Localizer/GS", "−6 dB", "DO-148", "Precision approach; safety-of-life"],
            ["VOR", "−6 dB", "DO-196", "En-route navigation"],
            ["DME / TACAN", "−6 dB", "DO-189", "Distance measuring, co-located with ILS"],
            ["General ARNS", "−6 dB", "ITU-R M.1642", "Default ITU-R criterion for aeronautical"],
        ], columns=["System", "I/N Threshold", "Reference", "Rationale"])
        st.table(prot_data)

        st.subheader("✅ Self-Check")
        with st.expander("Click to reveal answers"):
            st.markdown("""
**Q1:** An interference study shows I/N = −8 dB at a GNSS receiver. Is this acceptable?
> **No** — the GPS threshold is −10 dB. At −8 dB, the threshold is violated by 2 dB. 
> The US should oppose unless mitigation measures are applied.

**Q2:** The same study shows I/N = −8 dB at a radio altimeter. Is this acceptable?
> **Yes** — the radio altimeter threshold is −6 dB. At −8 dB, there is 2 dB of margin.
> Compatible, but flag it as marginal and watch for more aggressive scenarios.

**Q3:** A contribution claims "interference is 20 dB below the noise floor, so it's negligible." 
Is that a strong argument?
> **Yes, actually** — I/N = −20 dB is well within any protection criterion. However, you should 
> verify the assumptions: What noise floor did they use? What propagation model? What deployment 
> density? A −20 dB result with aggressive assumptions may be −6 dB under realistic ones.
            """)

    # ── LESSON 8 ──────────────────────────────────────────────────────────────
    elif lesson.startswith("Lesson 8"):
        st.header("🌐 Lesson 8 — Propagation Models: P.452, P.528, P.676")
        ex("Path loss model choice can swing computed I/N by 10–30 dB depending on terrain and scenario. This is the single parameter most worth challenging in a proponent's study — an unjustified urban clutter correction or inappropriate P.452 time percentage can flip the compatibility finding entirely.")

        st.subheader("Why FSPL Is Just the Starting Point")
        st.markdown("""
Free Space Path Loss assumes a perfect vacuum. Real paths include:
- **Atmospheric gases** — oxygen and water vapor absorb RF above ~1 GHz
- **Terrain** — hills cause diffraction (some loss), but can also provide shielding
- **Clutter** — buildings, trees add loss in urban/suburban areas
- **Multipath** — reflections can add constructively or destructively

Different ITU-R models capture these effects for different scenarios.
        """)

        st.subheader("ITU-R P.452 — Terrestrial Point-to-Point")
        st.markdown("""
**Use when:** Both transmitter and receiver are on or near the ground.

**What it models:**
- Line-of-sight propagation with atmospheric refraction
- Diffraction over terrain obstacles
- Tropospheric scatter (important for long paths)
- Ducting and anomalous propagation

**Key input:** Terrain profile between Tx and Rx — the full P.452 requires actual terrain 
elevation data along the path. Our simplified version uses empirical clutter corrections.

**Time percentage:** P.452 calculates loss exceeded for a given percentage of time.
- 50% = median (use for general studies)
- 1% = nearly worst-case propagation (use for protection studies — less path loss = more interference)
- Always use 1% for regulatory protection work

**Terrain/clutter corrections (simplified):**
| Environment | Additional Loss vs FSPL |
|---|---|
| Open (flat, no obstacles) | ~0 dB |
| Suburban | ~8 dB |
| Urban | ~18 dB |
| Dense urban | ~26 dB |
        """)

        st.subheader("ITU-R P.528 — Aeronautical Propagation")
        st.markdown("""
**Use when:** The victim receiver is airborne (aircraft, helicopter).

**What it models:**
- Slant-path geometry (ground-to-air or air-to-air)
- Atmospheric refraction at low elevation angles
- Tropospheric scatter
- Radio horizon effects

**Why it's different from P.452:**
Aircraft receivers have line-of-sight to transmitters over much longer distances.
An aircraft at 10,000 ft altitude can "see" a ground transmitter 130+ km away.
This dramatically increases the potential number of interferers — and is why 
airborne receivers are especially vulnerable to aggregate interference.

**Key parameters:**
- Aircraft altitude (higher = more line-of-sight, more interferers visible)
- Ground distance (horizontal separation)
- Time percentage (use 1% for protection studies)
        """)

        st.subheader("ITU-R P.676 — Atmospheric Gaseous Attenuation")
        st.markdown("""
**Use when:** Assessing attenuation on paths above ~1 GHz where atmospheric absorption matters.

**What it models:**
- Oxygen absorption (peaks at 60 GHz — used by 5G mmWave for natural isolation)
- Water vapor absorption (peaks at 22 GHz)
- Path length through the atmosphere

**Practical relevance for FAA work:**
- Below 3 GHz: gaseous absorption is minimal (<0.1 dB/km), not a useful protection mechanism
- Above 6 GHz: starts to matter for longer paths
- Above 10 GHz: significant; helps protect airborne weather radar (9–10 GHz)
- The itur library implements P.676 directly — you can run it live in the Propagation module

**Important:** Do NOT claim gaseous attenuation as a protection mechanism for ground-to-air 
paths below 6 GHz in ITU-R submissions — reviewers will challenge it as negligible.
        """)

        st.subheader("Model Selection Decision Tree")
        ex("The ITU-R rapporteur and experienced delegations will immediately flag model misapplication — e.g., using P.452 for an airborne receiver, using 50% time percentage for a protection study, or applying urban clutter corrections in open terrain near airports. These are grounds to request reanalysis.")
        st.markdown("""
```
Is the victim airborne?
├── YES → Use ITU-R P.528
│         (slant path, altitude matters)
└── NO → Is the path over open ocean or flat terrain?
         ├── YES → Use P.452 (open, time%=1)
         └── NO  → Use P.452 (suburban/urban, time%=1)

Always run FSPL first:
├── If FSPL shows compatibility → you're protected (FSPL is most optimistic)
└── If FSPL shows violation → run P.452/P.528 for realistic assessment
    ├── If realistic model still shows violation → cite harmful interference
    └── If realistic model shows compliance → document the margin
```
        """)

        st.subheader("✅ Self-Check")
        with st.expander("Click to reveal answers"):
            st.markdown("""
**Q1:** You're analyzing whether a new 5G base station could interfere with aircraft radio altimeters during approach. Which propagation model do you use?
> **P.528** — the victim is airborne, so slant-path aeronautical propagation applies.

**Q2:** A contribution uses FSPL and shows I/N = −8 dB. Should you be concerned?
> **Yes** — FSPL is the most optimistic model (least path loss = most interference at victim). 
> If FSPL only shows −8 dB margin, more realistic models (P.452/P.528) will show even less margin. 
> The scenario is close to the protection threshold and warrants further study.

**Q3:** A submitting administration uses P.452 with urban clutter correction (+18 dB) to show compatibility. What should you check?
> Check whether the deployment scenario actually justifies "urban" clutter. If the interferers 
> could be deployed in open or suburban areas near airports, the 18 dB clutter correction is 
> not valid and the real interference could be 10–18 dB higher than claimed.
            """)

    # ── LESSON 9 ──────────────────────────────────────────────────────────────
    elif lesson.startswith("Lesson 9"):
        st.header("🎲 Lesson 9 — Monte Carlo & Aggregate Interference")
        ex("Single-entry analysis captures the worst individual interferer but misses aggregate effects. For a deployment of N interferers, aggregate I/N ≈ single-entry I/N + 10·log10(N) at low N — 10 interferers add ~10 dB to the single-entry result, potentially flipping a compliant scenario into a violation.")

        st.subheader("Why Single-Scenario Analysis Isn't Enough")
        st.markdown("""
A link budget with one interferer at one distance answers: "What if the worst case happens?"
But in real deployments:
- Thousands of base stations exist, each at different distances
- Not all are transmitting at maximum power all the time
- The combined (aggregate) effect of many interferers can exceed the threshold 
  even if each individual one is below it

**Aggregate interference** is the sum of all interferers' contributions at the victim receiver.
This is what the ITU-R actually protects against, and it's what Monte Carlo quantifies.
        """)

        st.subheader("How Monte Carlo Works")
        st.markdown("""
**One trial:**
1. Randomly place N interferers within the deployment area
2. Calculate each interferer's received power at the victim (using path loss model)
3. Sum all N interference powers linearly (in milliwatts, not dB)
4. Convert back to dBm → compare to noise floor → compute I/N
5. Record whether I/N > threshold (a "violation")

**Many trials (typically 2,000–10,000):**
- Repeat steps 1–5 thousands of times
- Each trial uses different random positions
- Result: a statistical distribution of I/N values

**Key output:**
- **Violation probability** = fraction of trials where I/N > threshold
- **ITU-R criterion:** violation probability must be < 5% (or 1% for stringent cases)
        """)

        st.subheader("🔢 Mini Monte Carlo — Watch It Work")
        ex("Each trial places N interferers uniformly in area (positions drawn from P(r) ∝ r, i.e., r = √(U)·R_max for uniform areal density), computes path loss per interferer, sums interference powers linearly in mW, then converts aggregate to dBm for the I/N comparison.")

        col1, col2 = st.columns(2)
        with col1:
            mc_n = st.slider("Number of interferers (N)", 1, 50, 10)
            mc_eirp = st.number_input("Each interferer EIRP (dBm)", value=58.0, step=1.0)
            mc_f_l9 = st.number_input("Frequency (MHz)", value=4300.0, step=100.0)
            mc_r = st.number_input("Deployment radius (km)", value=20.0, step=1.0)
            mc_noise_l9 = st.number_input("Noise floor (dBm)", value=-89.0, step=1.0)
            mc_thresh_l9 = st.number_input("I/N threshold (dB)", value=-6.0, step=1.0)
            n_trials_l9 = st.select_slider("Trials", [500, 1000, 2000], value=1000)

        if st.button("▶ Run Mini Monte Carlo", key="mc_l9"):
            trials_in = []
            for _ in range(n_trials_l9):
                r = np.sqrt(np.random.uniform(0.1**2, mc_r**2, mc_n))
                pl = 20*np.log10(np.maximum(r, 0.01)) + 20*np.log10(mc_f_l9) + 32.44
                rx_pw = mc_eirp - pl
                agg_mw = np.sum(10**(rx_pw/10))
                agg_dbm = 10*np.log10(max(agg_mw, 1e-30))
                trials_in.append(agg_dbm - mc_noise_l9)

            trials_in = np.array(trials_in)
            vp = np.mean(trials_in > mc_thresh_l9) * 100

            with col2:
                st.metric("Median I/N", f"{np.percentile(trials_in,50):.1f} dB")
                st.metric("95th pct I/N", f"{np.percentile(trials_in,95):.1f} dB")
                st.metric("Violation Probability", f"{vp:.1f}%",
                          delta="High Risk" if vp > 5 else "OK",
                          delta_color="inverse" if vp > 5 else "normal")

            fig_l9, ax_l9 = plt.subplots(figsize=(8, 4))
            fig_l9.patch.set_facecolor("#0e1117")
            ax_l9.set_facecolor("#0e1117")
            ax_l9.hist(trials_in, bins=50, color='#4488ff', alpha=0.8, density=True)
            ax_l9.axvline(mc_thresh_l9, color='red', lw=2, linestyle='--', label=f'Threshold ({mc_thresh_l9} dB)')
            ax_l9.axvline(np.percentile(trials_in,95), color='orange', lw=1.5, linestyle=':', label='95th pct')
            ax_l9.set_xlabel("I/N (dB)", color='white')
            ax_l9.set_ylabel("Density", color='white')
            ax_l9.set_title("Monte Carlo I/N Distribution", color='white')
            ax_l9.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=8)
            ax_l9.tick_params(colors='white')
            for sp in ax_l9.spines.values(): sp.set_color('#444')
            plt.tight_layout()
            st.pyplot(fig_l9)

        st.subheader("How to Challenge a Monte Carlo Study")
        st.markdown("""
When reviewing another administration's Monte Carlo contribution, examine these assumptions:

| Assumption | What to look for | Your challenge |
|---|---|---|
| **N (number of interferers)** | Is the deployment density realistic? | Cite actual deployment data; argue for higher N |
| **Deployment area** | Did they exclude areas near airports? | Challenge exclusion zones as unrealistic |
| **Propagation model** | Did they use urban clutter inappropriately? | Argue for open/suburban for areas near airports |
| **Time percentage** | Did they use 50% instead of 1%? | Cite ITU-R SM.2028 — use 1% for protection studies |
| **Tx power distribution** | Did they assume all stations at max power? | If not, challenge as non-conservative |
| **Antenna pattern** | Did they model realistic azimuth patterns? | Check whether downtilt assumptions are valid near airports |
        """)

        st.subheader("✅ Self-Check")
        with st.expander("Click to reveal answers"):
            st.markdown("""
**Q1:** A Monte Carlo study shows violation probability of 3.2%. Is this compatible?
> **Yes** — below the ITU-R 5% criterion. However, check whether conservative assumptions were used. 
> If they used 50% time percentage, the real 1% result could exceed 5%.

**Q2:** Why do we sum interference powers linearly and then convert to dB — why not just add dBm values?
> **Because dB values represent logarithms — you can't add logarithms to get the sum.**
> 10 dBm + 10 dBm ≠ 20 dBm. In linear: 10mW + 10mW = 20mW = 13 dBm.
> Always convert to mW, sum linearly, then convert back to dBm.

**Q3:** What does the CCDF curve in the Monte Carlo module tell you?
> The CCDF (Complementary CDF) shows the probability that I/N *exceeds* a given level.
> Reading it at your I/N threshold directly gives you the violation probability — 
> the key number for the ITU-R compatibility criterion.
            """)

    # ── LESSON 10 ──────────────────────────────────────────────────────────────
    elif lesson.startswith("Lesson 10"):
        st.header("🏛️ Lesson 10 — From RF Math to ITU-R Policy")
        ex("The CPM Report method that gains traction at WRC is the one with the most administrations behind it, the most rigorous technical basis, and the clearest proposed regulatory text. Your job is to build all three simultaneously — analysis alone without proposed text is advisory, not decisive.")

        st.subheader("The Translation Problem")
        st.markdown("""
RF engineers speak in dB, dBm, I/N, path loss, and propagation models.
Policy officials speak in regulatory text, agenda items, resolutions, and geopolitical interests.

**Your job** is to be fluent in both. A technically perfect analysis that can't be 
communicated as policy language will not change the outcome of a Working Party session.
        """)

        st.subheader("Mapping RF Results to Policy Language")
        ex("The translation from dB margins to regulatory language is not mechanical — you must also characterize the consequence of interference (what flight operation fails, what safety margin is lost) to justify the regulatory weight of your proposed protection measure.")
        mapping = pd.DataFrame([
            ["I/N threshold violated", "Harmful interference cannot be excluded", "Grounds to oppose"],
            ["I/N margin < 3 dB", "Compatibility is marginal; further study required", "Request additional studies"],
            ["I/N margin > 10 dB", "Compatible under assessed conditions", "Support with conditions documented"],
            ["Violation probability > 5%", "The proposed use poses unacceptable interference risk", "Oppose or require mitigation"],
            ["Violation probability < 1%", "Compatible under conservative assumptions", "Support with caveats"],
            ["FSPL violates, P.528 complies", "Compatibility depends on deployment restrictions", "Propose coordination distance limits"],
            ["Aggregate > single-source", "Deployment density must be limited", "Propose density or PFD limits in RR footnote"],
            ["Receiver blocking identified", "Out-of-band OOBE limits are required", "Propose emission mask in RR or Recommendation"],
        ], columns=["RF Finding", "Policy Translation", "US Position"])
        st.table(mapping)

        st.subheader("The Regulatory Toolkit")
        st.markdown("""
Once you have a finding, you need to know what regulatory mechanism to request.
These are listed from most protective to least protective:

**1. Primary allocation protection (strongest)**
— Keep the protected band free of new allocations. Cite RR No. 4.10.

**2. Radio Regulations footnote**
— Add a footnote to the frequency table requiring coordination with aeronautical services. Negotiated at WRC.

**3. WRC Resolution**
— A formal decision requiring future studies or imposing conditions. Less binding than the RR table.

**4. ITU-R Recommendation**
— Technical guidance document with protection criteria. Not legally binding but heavily cited.

**5. Coordination zone / distance (weakest)**
— Informal agreement to coordinate around airports. Relies on national implementation.

**Rule:** Always ask for the strongest mechanism you can justify technically.
Start with footnote/Resolution language; let others negotiate you down. 
Never start by proposing weaker protection than you actually need.
        """)

        st.subheader("Writing the US Position — A Framework")
        st.markdown("""
When preparing for a Working Party meeting, your position paper should follow this logic:

```
1. IDENTIFY the threat
   "Document 5D/[X] proposes [Y] which could affect [Z FAA system] 
    operating in [band] under [allocation]."

2. QUANTIFY the risk  
   "Our analysis using ITU-R P.528 and SM.2028 shows I/N = [X] dB,
    exceeding the [−6/−10] dB threshold by [Y] dB under [scenario].
    Monte Carlo results indicate [Z]% violation probability."

3. STATE the consequence
   "This would degrade [system] performance, affecting [safety 
    consequence — e.g., CAT III approach capability, GPS accuracy]."

4. INVOKE the legal basis
   "Under RR No. 4.10, this constitutes harmful interference to a 
    safety-of-life service. Resolution 233 requires that new 
    allocations not degrade RNSS performance."

5. PROPOSE the solution
   "The United States proposes [specific text — PFD limit / 
    coordination distance / OOBE mask / footnote language]."

6. BUILD the coalition
   "We note that ICAO, [Admin A], and [Admin B] share these 
    concerns and have submitted aligned contributions."
```
        """)

        st.subheader("Red Flags in Others' Contributions")
        ex("Proponents have strong incentives to reach a compatibility finding — check every assumption that reduces computed interference: time percentage, clutter correction, deployment exclusion zones, antenna downtilt, frequency separation, and receiver bandwidths used for power density normalization.")
        st.markdown("""
| Red Flag | What It Means | Your Response |
|---|---|---|
| Uses 50% time percentage | Should be 1% for protection studies | Cite SM.2028; request reanalysis at 1% |
| Uses urban clutter near airports | Airports are open terrain | Challenge clutter classification |
| Excludes area within X km of airports | No regulatory basis for exclusion | Demand coordination distance be justified |
| Uses receiver characteristics worse than RTCA standard | Understates protection need | Cite RTCA DO-xxx; use correct receiver parameters |
| Only analyzes single interferer | Misses aggregate effect | Request Monte Carlo analysis per SM.2028 |
| Claims "no co-frequency operation" | Ignores OOBE/blocking | Raise receiver blocking/desensitization mechanism |
| Analysis only at median conditions | Should use worst-case | Request 99th percentile analysis |
        """)

        st.subheader("Your 30-Day Readiness Plan")
        ex("Follow this to get fully confident before your first Working Party meeting.")
        st.markdown("""
**Week 1 — Foundation**
- Complete Lessons 1–5 in this module
- Run the Link Budget module for the Radio Altimeter band (4200–4400 MHz) vs 5G at 4300 MHz
- Read the FAA/NTIA C-band technical report (publicly available) — it's a complete worked example

**Week 2 — Analysis**
- Complete Lessons 6–9
- Run Monte Carlo for Radio Altimeter, varying N from 5 to 50
- Read ITU-R M.1642 (free at itu.int) — 20 pages, the foundation of all IMT/ARNS studies

**Week 3 — Policy**
- Complete Lesson 10
- Use the Contribution Analyzer with the three example contributions
- Read two real US contributions from a past WP 5D meeting (ask NTIA for access)

**Week 4 — Integration**
- Draft a mock US contribution using the Contribution Summary module
- Present your analysis to a colleague and defend your assumptions
- Review the WRC-27 agenda items list and identify which ones affect FAA bands
        """)

        ok("You've completed the RF Training curriculum. You now have the fundamentals to run interference analyses, interpret results, and translate findings into ITU-R policy language.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 10 — MEETING NOTES
# ─────────────────────────────────────────────────────────────────────────────
elif selected_tab == "📓 Meeting Notes":
    st.title("📓 ITU-R Working Party Meeting Notes")
    ex("P1–P5 are the five formal plenary sessions that open, checkpoint, and close a WP meeting. Working Group sessions between plenaries produce the actual technical text — your most important notes are from WG sessions, not plenaries.")

    # ── Initialize session state stores ──────────────────────────────────────
    if "mn_meeting_info" not in st.session_state:
        st.session_state.mn_meeting_info = {
            "meeting_name": "", "location": "", "dates": "", "working_party": "WP 5D"
        }
    if "mn_sessions" not in st.session_state:
        st.session_state.mn_sessions = {}       # key: session_id → session dict
    if "mn_documents" not in st.session_state:
        st.session_state.mn_documents = {}      # key: doc_id → doc dict
    if "mn_ai_items" not in st.session_state:
        st.session_state.mn_ai_items = {}       # key: ai_id → agenda item dict
    if "mn_actions" not in st.session_state:
        st.session_state.mn_actions = []        # list of action items

    # ── Sub-page navigation ───────────────────────────────────────────────────
    sub = st.radio("Section:", [
        "🏁 Meeting Setup",
        "📝 Session Notes",
        "📄 Document Tracker",
        "🗂️ Agenda Item Tracker",
        "✅ Action Items",
        "📊 Meeting Dashboard",
        "📤 Export Full Record",
    ], horizontal=True)

    st.markdown("---")

    # ── MEETING SETUP ─────────────────────────────────────────────────────────
    if sub == "🏁 Meeting Setup":
        st.subheader("🏁 Meeting Setup")
        ex("The US delegation head is the spokesperson who must approve any US floor intervention — always check with them before speaking. The FAA lead is your technical authority for aeronautical positions. Know both before the meeting opens.")

        info = st.session_state.mn_meeting_info
        col1, col2 = st.columns(2)
        with col1:
            info["meeting_name"] = st.text_input("Meeting Name",
                value=info.get("meeting_name", ""),
                placeholder="e.g., ITU-R WP 5D Meeting #45")
            info["working_party"] = st.selectbox("Working Party",
                ["WP 5D", "WP 5B", "WP 5A", "SG 5", "Other"],
                index=["WP 5D","WP 5B","WP 5A","SG 5","Other"].index(info.get("working_party","WP 5D")))
            info["location"] = st.text_input("Location",
                value=info.get("location", ""),
                placeholder="e.g., ITU Headquarters, Geneva")
        with col2:
            info["dates"] = st.text_input("Dates",
                value=info.get("dates", ""),
                placeholder="e.g., 14–25 October 2025")
            info["us_head"] = st.text_input("US Delegation Head",
                value=info.get("us_head", ""),
                placeholder="Name / Organization")
            info["faa_lead"] = st.text_input("FAA Technical Lead",
                value=info.get("faa_lead", ""),
                placeholder="Your name")

        st.markdown("**WRC-27 Agenda Items Under Watch**")
        ex("WRC-27 agenda items are formally adopted at WRC-23. Each item is assigned to one or more Study Groups and Working Parties for technical study. The CPM Report summarizing results is due approximately 12 months before WRC-27.")
        ai_text = st.text_area("One per line (number — description):",
            value=info.get("ai_text", ""),
            placeholder="1.2 — IMT identification above 6 GHz\n1.4 — RNSS additional allocations\n9.1(b) — Resolution 236 review",
            height=120)
        info["ai_text"] = ai_text

        st.session_state.mn_meeting_info = info
        ok("Meeting info saved to session.")

    # ── SESSION NOTES ─────────────────────────────────────────────────────────
    elif sub == "📝 Session Notes":
        st.subheader("📝 Session Notes")
        ex("WP meetings typically last 2 weeks with 50–100+ documents. Working Groups handle specific agenda items and produce draft Recommendations, Reports, and CPM text. Track which WG is handling each of your agenda items — that is where the real work happens.")

        col1, col2, col3 = st.columns(3)
        with col1:
            session_label = st.selectbox("Session", [
                "P1 — Opening Plenary",
                "WG-1 (morning)", "WG-1 (afternoon)",
                "WG-2 (morning)", "WG-2 (afternoon)",
                "WG-3 (morning)", "WG-3 (afternoon)",
                "WG-4 (morning)", "WG-4 (afternoon)",
                "P2 — Mid-week Plenary",
                "WG-5 (morning)", "WG-5 (afternoon)",
                "WG-6 (morning)", "WG-6 (afternoon)",
                "WG-7 (morning)", "WG-7 (afternoon)",
                "P3 — Plenary",
                "WG-8 (morning)", "WG-8 (afternoon)",
                "P4 — Plenary",
                "P5 — Closing Plenary",
                "Coordination Meeting",
                "Bilateral",
            ])
        with col2:
            session_date = st.text_input("Date", placeholder="e.g., Mon 14 Oct")
        with col3:
            session_chair = st.text_input("Chair / Rapporteur", placeholder="Name")

        ai_context = st.text_input("Agenda Item(s) covered", placeholder="e.g., AI 1.2, AI 9.1(b)")

        st.markdown("**Session Notes**")
        ex("Note the document number of any text that was agreed, modified, or deferred — you will need to reference these exactly in your trip report and in any follow-up contributions. Agreed text is marked [AGREED]; contentious text is bracketed [ ] pending resolution.")

        note_text = st.text_area("Notes:", height=200,
            placeholder="""• Doc 5D/123 (China) introduced — proposes IMT identification at 4800-4990 MHz
• US (Smith) intervened citing FAA radio altimeter concerns — I/N analysis shows threshold exceeded
• France/Germany supported further study before any identification
• Rapporteur agreed to hold for additional contributions next meeting
• Japan proposed compromise text with coordination zone — US noted this is insufficient without PFD limits
• Decision: AIs 1.2 and 9.1(b) deferred to next meeting; new contributions due by [date]""")

        faa_outcome = st.selectbox("FAA Outcome This Session", [
            "✅ Favorable — FAA position advanced",
            "⚠️ Mixed — partial progress, concerns remain",
            "🔴 Unfavorable — opposing text gaining traction",
            "⏳ Deferred — no decision, carried forward",
            "ℹ️ Informational — no action required",
        ])

        key_decisions = st.text_area("Key Decisions / Agreed Text:", height=80,
            placeholder="Record any agreed language, decisions, or text that was approved...")

        follow_up = st.text_area("Follow-up Required:", height=80,
            placeholder="What does the US/FAA need to do before the next session?")

        if st.button("💾 Save Session Notes", type="primary"):
            sid = f"{session_label}_{session_date}".replace(" ","_").replace("/","_")
            st.session_state.mn_sessions[sid] = {
                "session": session_label,
                "date": session_date,
                "chair": session_chair,
                "ai_context": ai_context,
                "notes": note_text,
                "faa_outcome": faa_outcome,
                "key_decisions": key_decisions,
                "follow_up": follow_up,
            }
            ok(f"Session notes saved: {session_label} — {session_date}")

        # Show existing sessions
        if st.session_state.mn_sessions:
            st.markdown("---")
            st.subheader("Saved Sessions")
            for sid, s in st.session_state.mn_sessions.items():
                outcome_color = {
                    "✅ Favorable — FAA position advanced": "🟢",
                    "⚠️ Mixed — partial progress, concerns remain": "🟡",
                    "🔴 Unfavorable — opposing text gaining traction": "🔴",
                    "⏳ Deferred — no decision, carried forward": "🔵",
                    "ℹ️ Informational — no action required": "⚪",
                }.get(s["faa_outcome"], "⚪")
                with st.expander(f"{outcome_color} {s['session']} — {s['date']} | {s.get('ai_context','')}"):
                    st.markdown(f"**Chair:** {s.get('chair','—')}")
                    st.markdown(f"**FAA Outcome:** {s['faa_outcome']}")
                    st.markdown("**Notes:**")
                    st.text(s["notes"])
                    if s.get("key_decisions"):
                        st.markdown(f"**Key Decisions:** {s['key_decisions']}")
                    if s.get("follow_up"):
                        st.markdown(f"**Follow-up:** {s['follow_up']}")
                    if st.button(f"🗑️ Delete", key=f"del_sess_{sid}"):
                        del st.session_state.mn_sessions[sid]
                        st.rerun()

    # ── DOCUMENT TRACKER ──────────────────────────────────────────────────────
    elif sub == "📄 Document Tracker":
        st.subheader("📄 Document Tracker")
        ex("Documents are pre-posted to the ITU-R document system (TIES) before each meeting. Flag high-concern documents before the meeting starts so you arrive with prepared interventions — reacting in real time to a complex sharing study is not ideal.")

        with st.expander("➕ Add New Document", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                doc_num = st.text_input("Document Number", placeholder="e.g., 5D/234-E")
                doc_title = st.text_input("Title / Subject", placeholder="Brief description")
                doc_admin = st.text_input("Submitting Admin(s)", placeholder="e.g., China, Korea")
            with col2:
                doc_ai = st.text_input("WRC Agenda Item", placeholder="e.g., AI 1.2")
                doc_session = st.text_input("Introduced in Session", placeholder="e.g., WG-1 Mon AM")
                doc_type = st.selectbox("Document Type", [
                    "Sharing/compatibility study",
                    "Draft new Recommendation",
                    "Amendment to Recommendation",
                    "Liaison statement",
                    "Information document",
                    "Working document / DT",
                    "Chairman's report",
                ])
            with col3:
                doc_concern = st.selectbox("FAA Concern Level", [
                    "🔴 HIGH — Directly threatens FAA band",
                    "🟡 MEDIUM — Adjacent band / indirect risk",
                    "🟢 LOW — Monitor only",
                    "✅ FAVORABLE — Supports FAA position",
                    "⚪ NEUTRAL — No FAA impact",
                ])
                doc_us_action = st.selectbox("Required US Action", [
                    "Oppose — prepare rebuttal contribution",
                    "Comment — propose amendments",
                    "Support — align with US position",
                    "Monitor — no action this meeting",
                    "Refer to FAA for technical input",
                    "Coordinate with ICAO",
                ])

            doc_summary = st.text_area("Technical Summary / Notes:", height=100,
                placeholder="What does this document propose? What are the FAA implications? Key technical claims?")
            doc_faa_response = st.text_area("FAA Response / Intervention Taken:", height=80,
                placeholder="What did the US say? What text was proposed?")

            if st.button("💾 Save Document", type="primary", key="save_doc"):
                if doc_num:
                    did = doc_num.replace("/","_").replace("-","_").replace(" ","_")
                    st.session_state.mn_documents[did] = {
                        "doc_num": doc_num,
                        "title": doc_title,
                        "admin": doc_admin,
                        "ai": doc_ai,
                        "session": doc_session,
                        "doc_type": doc_type,
                        "concern": doc_concern,
                        "us_action": doc_us_action,
                        "summary": doc_summary,
                        "faa_response": doc_faa_response,
                    }
                    ok(f"Document {doc_num} saved.")
                else:
                    warn("Please enter a document number.")

        # Document table
        if st.session_state.mn_documents:
            st.subheader("Document Index")
            ex("Focus first on documents proposing new or amended Radio Regulations text (footnotes, allocations, Resolutions) — these have direct treaty-level impact. Informational documents and liaison statements are lower priority unless they establish technical precedent.")

            filter_concern = st.multiselect("Filter by concern level:",
                ["🔴 HIGH — Directly threatens FAA band",
                 "🟡 MEDIUM — Adjacent band / indirect risk",
                 "🟢 LOW — Monitor only",
                 "✅ FAVORABLE — Supports FAA position",
                 "⚪ NEUTRAL — No FAA impact"],
                default=["🔴 HIGH — Directly threatens FAA band",
                         "🟡 MEDIUM — Adjacent band / indirect risk"])

            rows = []
            for did, d in st.session_state.mn_documents.items():
                if not filter_concern or d["concern"] in filter_concern:
                    rows.append({
                        "Doc #": d["doc_num"],
                        "Admin": d["admin"],
                        "AI": d["ai"],
                        "Session": d["session"],
                        "Concern": d["concern"].split("—")[0].strip(),
                        "US Action": d["us_action"].split("—")[0].strip(),
                        "Title": d["title"],
                    })
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

            # Detail view
            st.subheader("Document Detail")
            doc_select = st.selectbox("Select document for full details:",
                [d["doc_num"] for d in st.session_state.mn_documents.values()])
            if doc_select:
                did_sel = doc_select.replace("/","_").replace("-","_").replace(" ","_")
                if did_sel in st.session_state.mn_documents:
                    d = st.session_state.mn_documents[did_sel]
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Submitting Admin:** {d['admin']}")
                        st.markdown(f"**Agenda Item:** {d['ai']}")
                        st.markdown(f"**Type:** {d['doc_type']}")
                        st.markdown(f"**FAA Concern:** {d['concern']}")
                    with col2:
                        st.markdown(f"**Session:** {d['session']}")
                        st.markdown(f"**US Action:** {d['us_action']}")
                    st.markdown(f"**Summary:** {d['summary']}")
                    if d.get("faa_response"):
                        st.markdown(f"**FAA Response:** {d['faa_response']}")
                    if st.button("🗑️ Delete this document", key=f"del_doc_{did_sel}"):
                        del st.session_state.mn_documents[did_sel]
                        st.rerun()

    # ── AGENDA ITEM TRACKER ───────────────────────────────────────────────────
    elif sub == "🗂️ Agenda Item Tracker":
        st.subheader("🗂️ Agenda Item Tracker")
        ex("The US position matrix is the internal coordination document that tracks where each agenda item stands relative to the US goal. Update it at the end of each day — it feeds directly into the delegation head's daily debrief and NTIA post-meeting report.")

        with st.expander("➕ Add / Update Agenda Item", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                ai_num = st.text_input("Agenda Item Number", placeholder="e.g., 1.2")
                ai_title = st.text_input("Title", placeholder="e.g., IMT identification above 6 GHz")
                ai_faa_bands = st.text_input("FAA Bands at Risk",
                    placeholder="e.g., Radio Altimeter 4200-4400 MHz, ARNS 5000-5150 MHz")
                ai_us_position = st.selectbox("Current US Position", [
                    "🔴 OPPOSE — unacceptable interference risk",
                    "🟡 CONDITIONAL — support with protective measures",
                    "🟢 SUPPORT — compatible with FAA systems",
                    "🔵 STUDYING — analysis underway",
                    "⚪ TBD — position not yet established",
                ])
            with col2:
                ai_status = st.selectbox("Meeting Status", [
                    "📥 Not yet discussed",
                    "🔄 Under discussion — active drafting",
                    "⚠️ Contentious — major disagreement",
                    "🤝 Converging — near agreement",
                    "✅ Agreed — text finalized this meeting",
                    "⏳ Deferred — carried to next meeting",
                ])
                ai_rapporteur = st.text_input("Rapporteur", placeholder="Name / Admin")
                ai_next_steps = st.text_area("Next Steps / FAA Required Actions:", height=80,
                    placeholder="What does the US/FAA need to do before next meeting?")

            ai_current_text = st.text_area("Current Draft Text (paste key proposed RR language):",
                height=100,
                placeholder="Paste the current state of any proposed RR footnote, Resolution, or Recommendation text...")
            ai_faa_concerns = st.text_area("Technical FAA Concerns:", height=80,
                placeholder="What specific interference mechanism or regulatory gap is the FAA concerned about?")
            ai_allies = st.text_input("Aligned Administrations / Organizations",
                placeholder="e.g., EU, Canada, Australia, ICAO")

            if st.button("💾 Save Agenda Item", type="primary", key="save_ai"):
                if ai_num:
                    aid = ai_num.replace(".","_").replace(" ","_")
                    st.session_state.mn_ai_items[aid] = {
                        "num": ai_num, "title": ai_title,
                        "faa_bands": ai_faa_bands,
                        "us_position": ai_us_position,
                        "status": ai_status,
                        "rapporteur": ai_rapporteur,
                        "next_steps": ai_next_steps,
                        "current_text": ai_current_text,
                        "faa_concerns": ai_faa_concerns,
                        "allies": ai_allies,
                    }
                    ok(f"Agenda Item {ai_num} saved.")
                else:
                    warn("Please enter an agenda item number.")

        # Position matrix
        if st.session_state.mn_ai_items:
            st.subheader("US Position Matrix")
            ex("End-of-day updates are critical: overnight, other delegations coordinate bilaterally and positions shift. If you arrive the next morning without an updated picture, you may be caught off-guard by a compromise proposal that moved without you.")
            rows = []
            for aid, a in st.session_state.mn_ai_items.items():
                rows.append({
                    "AI #": a["num"],
                    "Title": a["title"],
                    "FAA Bands at Risk": a["faa_bands"],
                    "US Position": a["us_position"].split("—")[0].strip(),
                    "Status": a["status"].split("—")[0].strip(),
                    "Allies": a["allies"],
                    "Rapporteur": a["rapporteur"],
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            # Detail expanders
            for aid, a in st.session_state.mn_ai_items.items():
                pos_icon = a["us_position"].split(" ")[0]
                with st.expander(f"{pos_icon} AI {a['num']} — {a['title']}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**FAA Bands:** {a['faa_bands']}")
                        st.markdown(f"**US Position:** {a['us_position']}")
                        st.markdown(f"**Status:** {a['status']}")
                        st.markdown(f"**Rapporteur:** {a['rapporteur']}")
                        st.markdown(f"**Allied Admins:** {a['allies']}")
                    with col2:
                        if a.get("faa_concerns"):
                            st.markdown(f"**FAA Concerns:** {a['faa_concerns']}")
                        if a.get("next_steps"):
                            st.markdown(f"**Next Steps:** {a['next_steps']}")
                    if a.get("current_text"):
                        st.markdown("**Current Draft Text:**")
                        st.code(a["current_text"], language=None)
                    if st.button("🗑️ Delete", key=f"del_ai_{aid}"):
                        del st.session_state.mn_ai_items[aid]
                        st.rerun()

    # ── ACTION ITEMS ──────────────────────────────────────────────────────────
    elif sub == "✅ Action Items":
        st.subheader("✅ Action Items")
        ex("Contribution deadlines are typically 4–6 weeks before each WP meeting. Missing a deadline means your analysis is submitted as 'late' — it may not be formally considered. Action items from the meeting floor (rapporteur requests) often have even shorter turnaround windows.")

        with st.expander("➕ Add Action Item", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                act_desc = st.text_area("Action Description:", height=80,
                    placeholder="e.g., Prepare interference analysis for AI 1.2 at 4800-4990 MHz using P.528, submit as US contribution")
            with col2:
                act_owner = st.text_input("Owner", placeholder="Name / Organization")
                act_due = st.text_input("Due Date", placeholder="e.g., 3 days before next meeting")
                act_ai = st.text_input("Related Agenda Item", placeholder="e.g., AI 1.2")
            with col3:
                act_priority = st.selectbox("Priority", [
                    "🔴 URGENT — before next session",
                    "🟡 HIGH — before end of meeting",
                    "🟢 NORMAL — before next meeting",
                    "🔵 LOW — informational follow-up",
                ])
                act_status = st.selectbox("Status", [
                    "⬜ Not started",
                    "🔄 In progress",
                    "✅ Complete",
                    "⛔ Blocked",
                ])

            if st.button("💾 Add Action Item", type="primary", key="save_act"):
                if act_desc:
                    st.session_state.mn_actions.append({
                        "desc": act_desc,
                        "owner": act_owner,
                        "due": act_due,
                        "ai": act_ai,
                        "priority": act_priority,
                        "status": act_status,
                    })
                    ok("Action item added.")

        if st.session_state.mn_actions:
            st.subheader("Open Action Items")
            for i, act in enumerate(st.session_state.mn_actions):
                pri_icon = act["priority"].split(" ")[0]
                stat_icon = act["status"].split(" ")[0]
                with st.expander(f"{pri_icon} {stat_icon} AI {act.get('ai','—')} — {act['desc'][:60]}..."):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Owner:** {act['owner']}")
                        st.markdown(f"**Due:** {act['due']}")
                        st.markdown(f"**Agenda Item:** {act['ai']}")
                    with col2:
                        st.markdown(f"**Priority:** {act['priority']}")
                        new_status = st.selectbox("Update Status:",
                            ["⬜ Not started","🔄 In progress","✅ Complete","⛔ Blocked"],
                            index=["⬜ Not started","🔄 In progress","✅ Complete","⛔ Blocked"].index(act["status"]),
                            key=f"act_stat_{i}")
                        if new_status != act["status"]:
                            st.session_state.mn_actions[i]["status"] = new_status
                    st.markdown(f"**Description:** {act['desc']}")
                    if st.button("🗑️ Remove", key=f"del_act_{i}"):
                        st.session_state.mn_actions.pop(i)
                        st.rerun()

    # ── DASHBOARD ─────────────────────────────────────────────────────────────
    elif sub == "📊 Meeting Dashboard":
        st.subheader("📊 Meeting Dashboard")
        info = st.session_state.mn_meeting_info
        if info.get("meeting_name"):
            st.markdown(f"### {info.get('meeting_name','Meeting')} — {info.get('working_party','')} | {info.get('location','')} | {info.get('dates','')}")
        ex("The dashboard gives you a rapid red/yellow/green picture of FAA equities across the meeting. A shift from 'Converging' to 'Contentious' on a high-concern agenda item is an escalation trigger — notify NTIA and consider requesting a bilateral with the opposing administration.")

        # Summary metrics
        docs = st.session_state.mn_documents
        ais = st.session_state.mn_ai_items
        sessions = st.session_state.mn_sessions
        actions = st.session_state.mn_actions

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Sessions Logged", len(sessions))
        col2.metric("Documents Tracked", len(docs))
        col3.metric("Agenda Items", len(ais))
        col4.metric("Action Items", len(actions))
        col5.metric("Open Actions",
            sum(1 for a in actions if "Complete" not in a["status"]))

        # Document concern breakdown
        if docs:
            st.markdown("---")
            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("Documents by Concern Level")
                concern_counts = {}
                for d in docs.values():
                    c = d["concern"].split("—")[0].strip()
                    concern_counts[c] = concern_counts.get(c, 0) + 1
                fig_dash, ax_dash = plt.subplots(figsize=(5, 4))
                fig_dash.patch.set_facecolor("#0e1117")
                ax_dash.set_facecolor("#0e1117")
                labels = list(concern_counts.keys())
                sizes = list(concern_counts.values())
                colors_pie = ['#ff4444','#ffaa00','#44bb44','#4488ff','#aaaaaa'][:len(labels)]
                wedges, texts, autotexts = ax_dash.pie(sizes, labels=labels,
                    autopct='%1.0f%%', colors=colors_pie, textprops={'color':'white','fontsize':8})
                plt.tight_layout()
                st.pyplot(fig_dash)

            with col_b:
                st.subheader("US Position Summary")
                if ais:
                    pos_counts = {}
                    for a in ais.values():
                        p = a["us_position"].split("—")[0].strip()
                        pos_counts[p] = pos_counts.get(p, 0) + 1
                    fig_pos, ax_pos = plt.subplots(figsize=(5, 4))
                    fig_pos.patch.set_facecolor("#0e1117")
                    ax_pos.set_facecolor("#0e1117")
                    bars = ax_pos.barh(list(pos_counts.keys()),
                        list(pos_counts.values()),
                        color=['#ff4444','#ffaa00','#44bb44','#4488ff','#aaaaaa'][:len(pos_counts)])
                    ax_pos.tick_params(colors='white', labelsize=8)
                    ax_pos.set_xlabel("Count", color='white')
                    for sp in ax_pos.spines.values(): sp.set_color('#444')
                    plt.tight_layout()
                    st.pyplot(fig_pos)
                else:
                    st.info("No agenda items logged yet.")

        # Session outcomes timeline
        if sessions:
            st.markdown("---")
            st.subheader("Session Outcomes")
            for sid, s in sessions.items():
                outcome = s["faa_outcome"]
                icon = "🟢" if "Favorable" in outcome else "🔴" if "Unfavorable" in outcome else "🟡" if "Mixed" in outcome else "🔵"
                st.markdown(f"{icon} **{s['session']}** ({s['date']}) — {s.get('ai_context','—')} — *{outcome.split('—')[-1].strip()}*")

        # High-priority actions
        if actions:
            st.markdown("---")
            st.subheader("🔴 Open High-Priority Actions")
            urgent = [a for a in actions if "URGENT" in a["priority"] and "Complete" not in a["status"]]
            if urgent:
                for a in urgent:
                    st.markdown(f"- **{a['owner']}** | AI {a.get('ai','—')} | Due: {a['due']} — {a['desc'][:80]}...")
            else:
                ok("No urgent open actions.")

    # ── EXPORT ────────────────────────────────────────────────────────────────
    elif sub == "📤 Export Full Record":
        st.subheader("📤 Export Full Meeting Record")
        ex("The trip report is a formal deliverable to NTIA and FAA within 5–10 business days of the meeting. It must document: all agreed text affecting FAA interests, US interventions made, outstanding action items, and recommended US positions for the next meeting cycle.")

        info = st.session_state.mn_meeting_info
        docs = st.session_state.mn_documents
        ais = st.session_state.mn_ai_items
        sessions = st.session_state.mn_sessions
        actions = st.session_state.mn_actions

        if st.button("📄 Generate Full Export", type="primary"):
            lines = []
            lines.append("=" * 70)
            lines.append("ITU-R WORKING PARTY MEETING RECORD")
            lines.append("FAA RF Interference Analysis Tool — Meeting Notes Export")
            lines.append("=" * 70)
            lines.append(f"Meeting:      {info.get('meeting_name','')}")
            lines.append(f"Working Party:{info.get('working_party','')}")
            lines.append(f"Location:     {info.get('location','')}")
            lines.append(f"Dates:        {info.get('dates','')}")
            lines.append(f"US Del. Head: {info.get('us_head','')}")
            lines.append(f"FAA Lead:     {info.get('faa_lead','')}")
            lines.append("")

            # Position Matrix
            lines.append("=" * 70)
            lines.append("US POSITION MATRIX — AGENDA ITEMS")
            lines.append("=" * 70)
            if ais:
                for a in ais.values():
                    lines.append(f"\nAI {a['num']} — {a['title']}")
                    lines.append(f"  FAA Bands at Risk: {a['faa_bands']}")
                    lines.append(f"  US Position:       {a['us_position']}")
                    lines.append(f"  Meeting Status:    {a['status']}")
                    lines.append(f"  Rapporteur:        {a['rapporteur']}")
                    lines.append(f"  Allied Admins:     {a['allies']}")
                    if a.get("faa_concerns"):
                        lines.append(f"  FAA Concerns:      {a['faa_concerns']}")
                    if a.get("next_steps"):
                        lines.append(f"  Next Steps:        {a['next_steps']}")
                    if a.get("current_text"):
                        lines.append(f"  Draft Text:\n    {a['current_text']}")
            else:
                lines.append("  No agenda items logged.")

            # Document Index
            lines.append("")
            lines.append("=" * 70)
            lines.append("DOCUMENT INDEX — FAA FLAGGED DOCUMENTS")
            lines.append("=" * 70)
            if docs:
                high = [d for d in docs.values() if "HIGH" in d["concern"]]
                med  = [d for d in docs.values() if "MEDIUM" in d["concern"]]
                for group, label in [(high,"HIGH CONCERN"),(med,"MEDIUM CONCERN")]:
                    if group:
                        lines.append(f"\n--- {label} ---")
                        for d in group:
                            lines.append(f"\n  Doc {d['doc_num']} | {d['admin']} | AI {d['ai']} | {d['session']}")
                            lines.append(f"  Title:   {d['title']}")
                            lines.append(f"  Type:    {d['doc_type']}")
                            lines.append(f"  Action:  {d['us_action']}")
                            if d.get("summary"):
                                lines.append(f"  Summary: {d['summary']}")
                            if d.get("faa_response"):
                                lines.append(f"  US Response: {d['faa_response']}")
            else:
                lines.append("  No documents logged.")

            # Session Notes
            lines.append("")
            lines.append("=" * 70)
            lines.append("SESSION NOTES")
            lines.append("=" * 70)
            if sessions:
                for s in sessions.values():
                    lines.append(f"\n{s['session']} — {s['date']}")
                    lines.append(f"  Chair: {s.get('chair','—')} | AI(s): {s.get('ai_context','—')}")
                    lines.append(f"  FAA Outcome: {s['faa_outcome']}")
                    lines.append(f"  Notes:\n    {s['notes']}")
                    if s.get("key_decisions"):
                        lines.append(f"  Key Decisions: {s['key_decisions']}")
                    if s.get("follow_up"):
                        lines.append(f"  Follow-up: {s['follow_up']}")
            else:
                lines.append("  No sessions logged.")

            # Action Items
            lines.append("")
            lines.append("=" * 70)
            lines.append("ACTION ITEMS")
            lines.append("=" * 70)
            if actions:
                open_acts = [a for a in actions if "Complete" not in a["status"]]
                done_acts = [a for a in actions if "Complete" in a["status"]]
                if open_acts:
                    lines.append("\n--- OPEN ---")
                    for a in open_acts:
                        lines.append(f"\n  [{a['status']}] {a['priority']}")
                        lines.append(f"  Owner: {a['owner']} | Due: {a['due']} | AI: {a.get('ai','—')}")
                        lines.append(f"  {a['desc']}")
                if done_acts:
                    lines.append("\n--- COMPLETED ---")
                    for a in done_acts:
                        lines.append(f"  ✅ {a['owner']} | AI: {a.get('ai','—')} — {a['desc'][:60]}...")
            else:
                lines.append("  No action items logged.")

            lines.append("")
            lines.append("=" * 70)
            lines.append("END OF MEETING RECORD")
            lines.append("Generated by FAA RF Interference Analysis Tool")
            lines.append("=" * 70)

            export_str = "\n".join(lines)
            st.text_area("Preview:", export_str, height=400)
            fname = f"meeting_record_{info.get('working_party','WP').replace(' ','_')}.txt"
            st.download_button("⬇️ Download Full Meeting Record (.txt)",
                export_str, file_name=fname, mime="text/plain")
            ok("Export ready. This document is suitable for your FAA/NTIA trip report.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 11 — CODE ANALYZER
# ─────────────────────────────────────────────────────────────────────────────
elif selected_tab == "🔬 Code Analyzer":
    st.title("🔬 Contribution Code Analyzer")
    st.markdown("*Paste MATLAB or Python code from any ITU-R contribution and get a line-by-line FAA interference critique across WP 5D, 5B, 7C, and 7D.*")
    ex("Code embedded in ITU-R contributions contains the assumptions that drive compatibility findings — propagation model choices, deployment densities, receiver parameters, time percentages, and aggregation methods are all buried here and almost never questioned in plenary sessions.")

    # API key
    api_key = None
    try:
        api_key = st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        pass

    if not api_key:
        st.warning("⚠️ Anthropic API key not configured — see the Contribution Analyzer module for setup instructions.")
        st.stop()

    st.markdown("---")

    # ── Metadata ─────────────────────────────────────────────────────────────
    st.subheader("📋 Code Context")
    ex("Context determines which WP-specific critique framework is applied — a 5D IMT sharing study is assessed against ARNS/RNSS criteria; a 7C radar study is assessed against radio altimeter and SSR criteria.")

    col1, col2, col3 = st.columns(3)
    with col1:
        code_wp = st.multiselect("Working Party (select all that apply):", [
            "WP 5D — IMT / Mobile",
            "WP 5B — Maritime / Radiodetermination",
            "WP 7C — Radiolocation / Radar",
            "WP 7D — Radio Astronomy / Passive",
            "SG 5 — Terrestrial Services",
            "Other",
        ], default=["WP 5D — IMT / Mobile"])
        code_doc = st.text_input("Document Number", placeholder="e.g., 5D/567-E")
        code_admin = st.text_input("Submitting Administration", placeholder="e.g., China")

    with col2:
        code_lang = st.selectbox("Code Language", [
            "MATLAB",
            "Python",
            "MATLAB + Python (mixed)",
            "R",
            "Unknown / Pseudocode",
        ])
        code_ai = st.text_input("WRC Agenda Item", placeholder="e.g., AI 1.2")
        code_scenario = st.selectbox("Analysis Type in Code", [
            "Sharing / compatibility study",
            "Aggregate interference (Monte Carlo)",
            "Single-entry interference",
            "Propagation / path loss calculation",
            "Link budget",
            "PFD / EPFD calculation",
            "Receiver sensitivity / selectivity model",
            "Terrain / geographic analysis",
            "Unknown",
        ])

    with col3:
        faa_victim = st.multiselect("FAA Victim System(s) to Assess Against:", [
            "GPS L1 (1559–1610 MHz)",
            "GPS L5 / ARNS (1164–1215 MHz)",
            "Radio Altimeter (4200–4400 MHz)",
            "ADS-B / Mode-S (1085–1095 MHz)",
            "DME / TACAN (960–1215 MHz)",
            "ILS / VOR (108–335 MHz)",
            "En-Route Radar (2700–2900 MHz)",
            "ARNS 5 GHz (5000–5150 MHz)",
            "Airborne Weather Radar (9000–9500 MHz)",
            "WAAS / SBAS (1559–1610 MHz)",
            "All FAA protected bands",
        ], default=["Radio Altimeter (4200–4400 MHz)"])
        critique_depth = st.selectbox("Critique Depth", [
            "Quick scan — flag critical assumption errors",
            "Standard — full line-by-line critique with corrections",
            "Deep — full critique + corrected code + contribution rebuttal text",
        ])

    # ── Code input ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📥 Code Input")
    ex("Paste the complete code block. If it spans multiple files, paste all sections concatenated — the AI will identify file/function boundaries. Comments and variable names are critical context.")

    code_tabs = st.tabs(["📝 Paste Code", "📎 Multiple Blocks"])

    with code_tabs[0]:
        code_input = st.text_area(
            "Paste MATLAB / Python code here:",
            height=350,
            placeholder="""% Example MATLAB snippet from a WP 5D sharing study
% ITU-R WP 5D Document 5D/XXX — Sharing study IMT vs ARNS

f_MHz = 4300;           % Interferer center frequency
Pt_dBm = 43;            % Transmit power
Gt_dBi = 15;            % Antenna gain
distances = 1:1:50;     % km

% Free space path loss
FSPL = 20*log10(distances) + 20*log10(f_MHz) + 32.44;

% Received interference
Pr = Pt_dBm + Gt_dBi - FSPL;

% Victim noise floor (assumed)
N_floor = -89;          % dBm

% I/N check
IN_ratio = Pr - N_floor;
threshold = -6;         % dB

compatible = all(IN_ratio < threshold);
fprintf('Compatible: %d\\n', compatible);""",
            key="code_main"
        )

    with code_tabs[1]:
        st.markdown("Paste additional code blocks (e.g., separate functions, helper scripts):")
        code_block2 = st.text_area("Block 2 (optional):", height=150, key="code_b2")
        code_block3 = st.text_area("Block 3 (optional):", height=150, key="code_b3")
        code_input_full = code_input
        if code_block2.strip():
            code_input_full += "\n\n% --- BLOCK 2 ---\n" + code_block2
        if code_block3.strip():
            code_input_full += "\n\n% --- BLOCK 3 ---\n" + code_block3

    # combine blocks — fallback if only tab 0 used
    if not code_block2.strip() and not code_block3.strip():
        code_input_full = code_input

    # ── Optional context ──────────────────────────────────────────────────────
    with st.expander("➕ Additional Context (recommended)"):
        code_summary = st.text_area(
            "What does this code claim to show? (from the contribution text):",
            height=80,
            placeholder="e.g., 'This study demonstrates that IMT base stations operating at 4800 MHz are compatible with radio altimeters at separation distances > 3 km using P.452 urban clutter model.'"
        )
        code_conclusion = st.text_area(
            "What conclusion does the contribution draw from this code?:",
            height=80,
            placeholder="e.g., 'The analysis shows compatibility at 99% of locations, supporting IMT identification at 4800–4990 MHz.'"
        )
        code_prior_concern = st.text_area(
            "Specific FAA technical concern to focus on (optional):",
            height=60,
            placeholder="e.g., Focus on whether receiver blocking / desensitization is modeled, not just in-band interference."
        )

    # ── Run analysis ──────────────────────────────────────────────────────────
    st.markdown("---")
    run_col1, run_col2 = st.columns([2, 1])
    with run_col1:
        run_btn = st.button("🔬 Analyze Code", type="primary",
                           disabled=not code_input.strip())
    with run_col2:
        st.caption("Analysis takes 20–45 seconds for deep critique")

    if run_btn and code_input.strip():

        # Build WP-specific context
        wp_context = {
            "WP 5D — IMT / Mobile": """WP 5D governs IMT (4G/5G/6G) spectrum. Key FAA concerns:
- New IMT allocations adjacent to radio altimeter (4200-4400 MHz), ARNS 5 GHz (5000-5150 MHz), GPS L5 (1164-1215 MHz)
- Sharing studies must use P.528 for airborne victims, not P.452 terrestrial
- Monte Carlo must follow SM.2028; time percentage must be 1% for protection studies
- Out-of-band emissions and receiver blocking must be analyzed, not just in-band sharing
- Protection criteria: I/N ≤ −6 dB (ARNS), I/N ≤ −10 dB (GNSS), per M.1477/M.1642""",

            "WP 5B — Maritime / Radiodetermination": """WP 5B governs maritime mobile, VDES, and radiodetermination (radar).
Key FAA concerns:
- VDES/AIS expansion in VHF maritime band (156-174 MHz) near aeronautical VHF (118-136 MHz)
- Shore-based radar systems near airborne weather radar (9000-9500 MHz) and SSR (1030/1090 MHz)
- Radiodetermination allocations near DME/TACAN (960-1215 MHz)
- Protection criteria: I/N ≤ −6 dB for ARNS radar systems per ITU-R M.1849""",

            "WP 7C — Radiolocation / Radar": """WP 7C governs radiolocation and radar systems.
Key FAA concerns:
- New radiolocation allocations overlapping or adjacent to: radio altimeter (4200-4400 MHz),
  airborne weather radar (9000-9500 MHz), ATC en-route radar (2700-2900 MHz), SSR (1030/1090 MHz)
- Aggregate interference from ground-based radars into airborne radar receivers
- Receiver blocking from high-power ground radars into sensitive airborne LNAs
- Radar pulse characteristics (PRF, peak power, duty cycle) must be accounted for — not just average power""",

            "WP 7D — Radio Astronomy / Passive": """WP 7D governs radio astronomy and passive services.
Key FAA concerns:
- Passive service allocations adjacent to active aeronautical bands can create regulatory pressure
  to constrain FAA system emissions
- Radio astronomy protection zones near major airports can affect DME/SSR operating parameters
- Coordination with passive services in L-band (1400-1427 MHz) near GPS (1559-1610 MHz) and
  GNSS L2 (1215-1300 MHz) — passive allocations set precedent for strict emission limits
  that indirectly constrain aeronautical systems in adjacent bands""",
        }

        active_wp_context = "\n\n".join([
            wp_context.get(wp, "") for wp in code_wp if wp in wp_context
        ])

        faa_bands_str = "\n".join([
            f"- {name}: {b['f_low_mhz']}–{b['f_high_mhz']} MHz | I/N threshold: {b['in_threshold_db']} dB | Noise floor: {b['noise_floor_dbm']} dBm"
            for name, b in FAA_BANDS.items()
        ])

        depth_map = {
            "Quick scan — flag critical assumption errors":
                "Provide a QUICK SCAN: identify the 3–5 most critical assumption errors or omissions that would invalidate the compatibility finding from an FAA perspective. Be direct and specific. No need to rewrite code.",
            "Standard — full line-by-line critique with corrections":
                "Provide a FULL LINE-BY-LINE CRITIQUE: annotate each significant code block, identify all assumption errors, propose corrected parameter values with regulatory citations, and summarize the overall validity of the analysis.",
            "Deep — full critique + corrected code + contribution rebuttal text":
                "Provide a DEEP ANALYSIS: full line-by-line critique, a corrected version of the code with FAA-appropriate parameters, AND draft rebuttal text suitable for a US contribution opposing or amending the findings.",
        }

        system_prompt = f"""You are a senior RF systems engineer and ITU-R spectrum policy expert supporting the FAA and NTIA.
Your specialty is auditing MATLAB and Python code embedded in ITU-R Working Party contributions to identify
technical errors, unjustified assumptions, and methodology flaws that artificially produce favorable
compatibility findings for new spectrum users at the expense of protected aeronautical services.

You have deep expertise in:
- ITU-R propagation models: P.452, P.528, P.619, P.676, P.618
- Monte Carlo methodology per ITU-R SM.2028
- GNSS protection methodology: M.1318, M.1477, M.1904, M.1905
- IMT/ARNS compatibility methodology: M.1642
- Aeronautical receiver characteristics: RTCA DO-235B, DO-260B, DO-155, DO-189
- Radio Regulations: RR No. 4.10, RR 5.444, RR 5.328, Resolution 233, Resolution 750

FAA PROTECTED BANDS (your reference for victim system parameters):
{faa_bands_str}

WORKING PARTY CONTEXT:
{active_wp_context if active_wp_context else "General ITU-R aeronautical spectrum protection context."}

FAA VICTIM SYSTEMS UNDER ASSESSMENT: {", ".join(faa_victim)}

CRITIQUE PHILOSOPHY:
Every line of code in a sharing study encodes an assumption. Your job is to find where those assumptions
deviate from conservative, technically defensible values in ways that favor the proponent.

Common flaws to check:
1. PROPAGATION MODEL: Is P.528 used for airborne victims? Is time % = 1% for protection studies?
   Is urban clutter applied near airports (unjustified)? Is gaseous attenuation claimed below 3 GHz?
2. RECEIVER PARAMETERS: Is the noise floor consistent with RTCA MOPS? Is bandwidth correct?
   Is NF realistic? Is the I/N threshold correct for the system type?
3. EIRP / POWER: Is maximum authorized EIRP used, or median/typical? Are OOBE accounted for?
4. AGGREGATION: If N interferers, are they summed linearly in watts? Is N realistic?
   Is deployment density consistent with actual licensed densities?
5. GEOMETRY: For airborne victims, is the slant distance used (not horizontal)?
   Is aircraft altitude conservative (low altitude = shorter slant path = more interference)?
6. GNSS SPECIFIC: Is c = a − b applied per M.1318? Is the narrowband rule (≤700 Hz) checked?
   Is power normalized to dB(W/Hz) correctly using the right noise bandwidth?
7. RADAR SPECIFIC: For pulsed systems, is peak power vs average power handled correctly?
   Is the pulse duty cycle accounted for in the power density calculation?
8. MONTE CARLO SPECIFIC: Is the spatial distribution uniform in area (not radius)?
   Is the violation probability criterion 5%? Is 99th percentile cited?

{depth_map[critique_depth]}

Format your response with clear section headers. Use technical notation precisely.
Where you cite a correction, give the specific regulatory reference (Recommendation, RR article).
Be direct — this is an adversarial technical review, not a collaborative editorial."""

        user_msg = f"""Please analyze the following {code_lang} code from ITU-R contribution {code_doc or '[unknown doc]'} 
submitted by {code_admin or '[unknown admin]'} to {', '.join(code_wp)}.

ANALYSIS TYPE: {code_scenario}
WRC AGENDA ITEM: {code_ai or 'Not specified'}
FAA VICTIM SYSTEMS: {', '.join(faa_victim)}

{"CONTRIBUTION CLAIM: " + code_summary if code_summary else ""}
{"CONTRIBUTION CONCLUSION: " + code_conclusion if code_conclusion else ""}
{"SPECIFIC FAA CONCERN: " + code_prior_concern if code_prior_concern else ""}

CODE TO ANALYZE:
```{code_lang.lower().split()[0]}
{code_input_full}
```

Please provide your critique structured as follows:

## 1. CODE SUMMARY
What this code actually does (independently of what the contribution claims).

## 2. CRITICAL ASSUMPTION AUDIT
For each significant assumption: parameter name → value used → FAA-appropriate value → impact on I/N result → regulatory citation for the correct value.

## 3. PROPAGATION MODEL ASSESSMENT  
Is the correct ITU-R model used? Is the time percentage appropriate? Are terrain/clutter corrections justified?

## 4. RECEIVER CHARACTERIZATION ERRORS
Noise floor, bandwidth, NF, I/N threshold — are these consistent with RTCA MOPS and ITU-R Recommendations?

## 5. AGGREGATION / MONTE CARLO ERRORS (if applicable)
Spatial distribution, N value, linear summation, violation criterion.

## 6. GNSS-SPECIFIC ISSUES (if GNSS bands involved)
M.1318 c=a−b framework compliance, narrowband rule, power density normalization.

## 7. RADAR/PULSED SYSTEM ISSUES (if applicable)
Peak vs average power, duty cycle, pulse characteristics.

## 8. OVERALL VALIDITY ASSESSMENT
Is the compatibility conclusion defensible? What is the likely true I/N when correct parameters are applied?

## 9. RECOMMENDED US RESPONSE
What should the US delegation say in the Working Party session? What contribution text should be prepared?

{"## 10. CORRECTED CODE" if "Deep" in critique_depth else ""}
{"Provide corrected code with FAA-appropriate parameters annotated with regulatory justifications." if "Deep" in critique_depth else ""}"""

        with st.spinner(f"Running {critique_depth.split('—')[0].strip()} code analysis... 20–45 seconds"):
            try:
                client = anthropic.Anthropic(api_key=api_key)
                response = client.messages.create(
                    model="claude-opus-4-5",
                    max_tokens=6000,
                    messages=[{"role": "user", "content": user_msg}],
                    system=system_prompt,
                )
                critique_text = response.content[0].text

                st.success("✅ Code analysis complete")
                st.markdown("---")

                # Severity banner
                critical_keywords = ["CRITICAL", "violated", "invalid", "incorrect", "wrong",
                                     "understates", "overstates", "unjustified", "missing"]
                n_critical = sum(critique_text.upper().count(kw.upper()) for kw in critical_keywords)
                if n_critical > 15:
                    warn(f"HIGH CONCERN — analysis identified multiple significant methodology errors. Recommend preparing a US rebuttal contribution.")
                elif n_critical > 5:
                    st.warning("⚠️ MODERATE CONCERN — some assumption errors identified. Review carefully before accepting findings.")
                else:
                    ok("LOW CONCERN — code appears methodologically reasonable under initial review. Verify receiver parameters independently.")

                st.markdown("---")
                st.subheader("🔬 Technical Critique")
                st.markdown(critique_text)

                # Export
                st.markdown("---")
                export = f"""FAA RF INTERFERENCE TOOL — CODE ANALYSIS REPORT
Document: {code_doc or 'N/A'} | Admin: {code_admin or 'N/A'}
Working Party: {', '.join(code_wp)}
Agenda Item: {code_ai or 'N/A'}
Language: {code_lang} | Analysis Type: {code_scenario}
FAA Victim Systems: {', '.join(faa_victim)}
Critique Depth: {critique_depth}

{'='*60}
ORIGINAL CODE
{'='*60}
{code_input_full}

{'='*60}
FAA TECHNICAL CRITIQUE
{'='*60}
{critique_text}
"""
                col_dl1, col_dl2 = st.columns(2)
                with col_dl1:
                    st.download_button(
                        "⬇️ Download Full Critique (.txt)",
                        export,
                        file_name=f"code_critique_{(code_doc or 'doc').replace('/','_')}.txt",
                        mime="text/plain"
                    )
                with col_dl2:
                    # Prepare a quick intervention draft
                    intervention = f"""US DELEGATION INTERVENTION — TECHNICAL FLOOR STATEMENT
Document: {code_doc or '[doc number]'} | {', '.join(code_wp)} | AI {code_ai or '[AI]'}

The United States has reviewed the technical analysis in document {code_doc or '[doc number]'} 
and has identified the following technical concerns with the methodology:

[Insert top 3 findings from critique above]

The United States requests that the Working Party:
1. Require reanalysis using [corrected propagation model / receiver parameters / deployment assumptions]
2. Apply the protection criteria established in [M.1642 / M.1318 / M.1477 / SM.2028] 
3. Defer any regulatory conclusion pending submission of a corrected analysis

The United States will submit a formal contribution addressing these technical issues 
before the next meeting.

[US Delegation — {', '.join(code_wp)} | FAA/NTIA]
"""
                    st.download_button(
                        "⬇️ Download Floor Intervention Draft (.txt)",
                        intervention,
                        file_name=f"floor_intervention_{(code_doc or 'doc').replace('/','_')}.txt",
                        mime="text/plain"
                    )

            except anthropic.AuthenticationError:
                st.error("❌ Invalid API key.")
            except anthropic.RateLimitError:
                st.error("❌ Rate limit — wait a moment and retry.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    elif not code_input.strip():
        st.info("👆 Paste code above and click Analyze.")

    # ── Example code snippets ─────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("📋 Example code snippets to test with"):
        st.markdown("**Example 1 — WP 5D: IMT sharing study with several intentional flaws**")
        st.code("""% WP 5D Sharing Study — IMT at 4800 MHz vs Radio Altimeter
% Assumes: terrestrial P.452 urban, 50% time, low N, good NF

f_MHz = 4800;
Pt_dBm = 43;
Gt_dBi = 15;
cable_dB = 2;
EIRP = Pt_dBm + Gt_dBi - cable_dB;   % = 56 dBm

N_interferers = 3;
deploy_radius_km = 10;

% P.452 path loss with urban clutter
distances = sqrt(rand(N_interferers,1)) * deploy_radius_km;
FSPL = 20*log10(distances) + 20*log10(f_MHz) + 32.44;
urban_clutter_dB = 18;                 % urban correction
PL = FSPL + urban_clutter_dB;

% Received power per interferer
Pr_each = EIRP - PL;                   % dBm

% Aggregate (sum in dB - INCORRECT!)
Pr_agg = 10*log10(sum(10.^(Pr_each/10)));

% Victim receiver (radio altimeter)
BW_MHz = 500;                          % MHz - overstated
NF_dB = 10;                            % dB - pessimistic for victim
N_floor = -174 + 10*log10(BW_MHz*1e6) + NF_dB;

IN_dB = Pr_agg - N_floor;
threshold_dB = -6;
fprintf('I/N = %.1f dB, threshold = %d dB\\n', IN_dB, threshold_dB);
fprintf('Compatible: %d\\n', IN_dB < threshold_dB);
""", language="matlab")

        st.markdown("**Example 2 — WP 7C: Radiolocation study near SSR with pulsed power error**")
        st.code("""# WP 7C radiolocation study - interference to SSR at 1090 MHz
import numpy as np

f_MHz = 1090
# Pulsed radar parameters
peak_power_dBm = 70      # 10 kW peak
pulse_width_us = 1.0     # microseconds
prf_hz = 1000            # pulse repetition frequency

# INCORRECT: using average power without accounting for duty cycle properly
duty_cycle = pulse_width_us * 1e-6 * prf_hz
avg_power_dBm = peak_power_dBm + 10*np.log10(duty_cycle)

Gt_dBi = 20
EIRP_avg = avg_power_dBm + Gt_dBi    # Using average - wrong for pulse desensitization

dist_km = 10
FSPL = 20*np.log10(dist_km) + 20*np.log10(f_MHz) + 32.44

Pr = EIRP_avg - FSPL    # dBm at SSR receiver

# SSR noise floor - using wrong bandwidth
BW_MHz = 10             # SSR reply bandwidth ~1-2 MHz, not 10
NF_dB = 8
N_floor = -174 + 10*np.log10(BW_MHz*1e6) + NF_dB

IN_dB = Pr - N_floor
print(f"I/N = {IN_dB:.1f} dB")
print(f"Compatible: {IN_dB < -10}")
""", language="python")

        st.markdown("**Example 3 — WP 5D: GNSS L5 study missing M.1318 framework**")
        st.code("""% GPS L5 interference study - missing c = a-b framework
% WP 5D document - IMT at 1200 MHz vs GPS L5

f_MHz = 1200;
EIRP_dBm = 46;
dist_km = 5;

FSPL = 20*log10(dist_km) + 20*log10(f_MHz) + 32.44;
Pr_dBm = EIRP_dBm - FSPL;

% Using generic noise floor instead of M.1318 framework
BW_MHz = 20;
NF_dB = 2;
N_floor = -174 + 10*log10(BW_MHz*1e6) + NF_dB;

% Missing: conversion to dB(W/Hz), comparison to M.1318 'a' parameter
% Missing: application of 6 dB aeronautical safety margin (M.1477)
% Missing: narrowband rule check (M.1477 Annex 5 - 700 Hz threshold)
% Using simple I/N instead of c = a - b methodology

IN_dB = Pr_dBm - N_floor;
fprintf('I/N = %.1f dB vs threshold -10 dB\\n', IN_dB);
fprintf('Result: %s\\n', IN_dB < -10 ? 'Compatible' : 'Incompatible');
""", language="matlab")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 12 — GLOSSARY
# ─────────────────────────────────────────────────────────────────────────────
elif selected_tab == "📖 Glossary":
    st.title("📖 Glossary — Terms, Acronyms & Definitions")
    st.markdown("*Every acronym and technical term used in this tool and in ITU-R aeronautical spectrum work, in plain language.*")
    ex("Use Ctrl+F (or Cmd+F on Mac) to search this page, or use the category filter below to jump to a section.")

    # Search box
    search = st.text_input("🔍 Search glossary:", placeholder="Type any term or acronym...").strip().lower()

    # Category filter
    categories = [
        "All",
        "🔤 Acronyms — Systems & Organizations",
        "📡 RF & Signal Theory",
        "🌐 ITU-R & Regulatory",
        "✈️ Aeronautical Systems",
        "📐 Measurements & Units",
        "💻 Analysis Methods",
    ]
    cat_filter = st.selectbox("Filter by category:", categories)

    st.markdown("---")

    # ── Glossary data ─────────────────────────────────────────────────────────
    GLOSSARY = [

        # ── ACRONYMS — Systems & Organizations ───────────────────────────────
        ("ADS-B", "Automatic Dependent Surveillance — Broadcast",
         "A surveillance technology where aircraft determine their own position using GPS and periodically broadcast it. Air Traffic Control and other aircraft receive these broadcasts to track traffic. Operates at 1090 MHz. Mandated in US airspace since 2020.",
         "🔤 Acronyms — Systems & Organizations"),

        ("AGL", "Above Ground Level",
         "Height measured from the ground directly below, not from sea level. Antenna heights in interference analyses are typically specified in meters Above Ground Level (AGL).",
         "🔤 Acronyms — Systems & Organizations"),

        ("ARNS", "Aeronautical Radionavigation Service",
         "An ITU Radio Regulations allocation category covering radio systems used for aircraft navigation. ARNS is a primary allocation in most aeronautical frequency bands and carries safety-of-life status under Radio Regulations No. 4.10.",
         "🔤 Acronyms — Systems & Organizations"),

        ("ANS", "Aeronautical Navigation Service",
         "An ITU Radio Regulations service category covering navigation aids for aviation, including both ground-based and satellite-based systems.",
         "🔤 Acronyms — Systems & Organizations"),

        ("ATC", "Air Traffic Control",
         "The ground-based service that directs aircraft in controlled airspace. ATC uses radar (SSR, PSR), voice radio (VHF), and data links (ADS-B, CPDLC) to manage aircraft separation and flow.",
         "🔤 Acronyms — Systems & Organizations"),

        ("CAT I / II / III", "Instrument Landing System Categories I, II, III",
         "Categories of precision instrument approach defined by visibility and decision height. CAT I: visibility ≥ 550m. CAT II: visibility ≥ 300m. CAT III: visibility < 300m down to zero. Higher categories require more precise ILS signals and more stringent interference protection.",
         "🔤 Acronyms — Systems & Organizations"),

        ("CEPT", "European Conference of Postal and Telecommunications Administrations",
         "The European regional body that coordinates spectrum policy and produces shared European positions for ITU-R meetings. CEPT produces SEAMCAT (Spectrum Engineering Advanced Monte Carlo Analysis Tool) used in interference studies.",
         "🔤 Acronyms — Systems & Organizations"),

        ("CPM", "Conference Preparatory Meeting",
         "An ITU-R meeting held approximately 12 months before each World Radiocommunication Conference (WRC) that produces the CPM Report — a summary of all technical study results and proposed methods for each WRC agenda item.",
         "🔤 Acronyms — Systems & Organizations"),

        ("DME", "Distance Measuring Equipment",
         "A radio navigation system that measures the slant-range distance between an aircraft and a ground transponder. Aircraft interrogate the ground station at 1025–1150 MHz; the station responds at 962–1213 MHz. Used alongside VOR for position fixing and with ILS for approach guidance.",
         "🔤 Acronyms — Systems & Organizations"),

        ("EASA", "European Union Aviation Safety Agency",
         "The European counterpart to the FAA. EASA certifies aircraft and avionics in Europe and participates in ICAO and EUROCAE standards work.",
         "🔤 Acronyms — Systems & Organizations"),

        ("EGPWS", "Enhanced Ground Proximity Warning System",
         "An avionics system that alerts pilots when an aircraft is in dangerous proximity to terrain. Uses GPS position and a terrain database. Relies on accurate radio altimeter readings for low-altitude warnings.",
         "🔤 Acronyms — Systems & Organizations"),

        ("EPFD", "Equivalent Power Flux Density",
         "A measure of aggregate interference from non-geostationary satellite constellations at a point on the Earth's surface. EPFD limits are specified in ITU Radio Regulations Appendix 7 to protect terrestrial and geostationary services.",
         "🔤 Acronyms — Systems & Organizations"),

        ("ESA", "European Space Agency",
         "The intergovernmental organization responsible for Europe's space program, including the Galileo Global Navigation Satellite System (GNSS). ESA participates in ITU-R coordination meetings affecting GNSS spectrum.",
         "🔤 Acronyms — Systems & Organizations"),

        ("FAA", "Federal Aviation Administration",
         "The United States government agency responsible for regulating civil aviation, managing US airspace, and certifying aircraft and avionics. The FAA is the primary stakeholder defending aeronautical spectrum in US ITU-R proceedings through NTIA.",
         "🔤 Acronyms — Systems & Organizations"),

        ("FMG", "Frequency Management Group",
         "An ICAO working group that manages the frequency assignments for aeronautical services globally and prepares ICAO positions for ITU-R World Radiocommunication Conferences.",
         "🔤 Acronyms — Systems & Organizations"),

        ("FSMP", "Frequency Spectrum Management Panel",
         "An ICAO body responsible for managing the frequency spectrum used by international civil aviation and coordinating with ITU-R on spectrum issues affecting aviation.",
         "🔤 Acronyms — Systems & Organizations"),

        ("Galileo", "Galileo Global Navigation Satellite System",
         "The European Union's Global Navigation Satellite System (GNSS), operated by the European Space Agency (ESA) and European Union Agency for the Space Programme (EUSPA). Provides civil and safety-of-life positioning services on L1 (1575.42 MHz) and E5 (1164–1215 MHz) bands.",
         "🔤 Acronyms — Systems & Organizations"),

        ("GLONASS", "Global Navigation Satellite System (Russian)",
         "Russia's Global Navigation Satellite System, operated by the Russian Aerospace Defence Forces. Provides positioning on L1 (1602 MHz region) and L2 (1246 MHz region). Protected under ITU-R M.1904.",
         "🔤 Acronyms — Systems & Organizations"),

        ("GMDSS", "Global Maritime Distress and Safety System",
         "An internationally agreed-upon set of safety procedures, types of equipment, and communication protocols used to increase safety and make it easier to rescue distressed ships, boats, and aircraft. Governed by ITU-R Working Party 5B (WP 5B).",
         "🔤 Acronyms — Systems & Organizations"),

        ("GNSS", "Global Navigation Satellite System",
         "The generic term for satellite-based positioning systems including GPS (United States), GLONASS (Russia), Galileo (European Union), and BeiDou (China). GNSS signals arrive at Earth at approximately −130 dBm — roughly 20 dB above the receiver noise floor — making them extremely vulnerable to interference.",
         "🔤 Acronyms — Systems & Organizations"),

        ("GPS", "Global Positioning System",
         "The United States satellite-based navigation system operated by the US Space Force. Provides positioning, navigation, and timing (PNT) services on L1 (1575.42 MHz), L2 (1227.60 MHz), and L5 (1176.45 MHz). GPS L1 and L5 are used for aviation safety-of-life applications.",
         "🔤 Acronyms — Systems & Organizations"),

        ("GPWS", "Ground Proximity Warning System",
         "An avionics system that warns pilots when an aircraft is in immediate danger of controlled flight into terrain (CFIT). The enhanced version is called EGPWS (Enhanced Ground Proximity Warning System) and uses GPS and terrain databases.",
         "🔤 Acronyms — Systems & Organizations"),

        ("IALA", "International Association of Marine Aids to Navigation and Lighthouse Authorities",
         "The international organization that coordinates maritime navigation aids worldwide. IALA participates in ITU-R Working Party 5B (WP 5B) on maritime spectrum issues.",
         "🔤 Acronyms — Systems & Organizations"),

        ("IATA", "International Air Transport Association",
         "The trade association representing airlines worldwide. IATA participates in ITU-R as a Sector Member to protect airline operational spectrum interests.",
         "🔤 Acronyms — Systems & Organizations"),

        ("ICAO", "International Civil Aviation Organization",
         "A United Nations specialized agency that codifies the principles and techniques of international air navigation and promotes the planning of international air transport. ICAO has Sector Member status at ITU-R and submits influential contributions defending aeronautical spectrum allocations.",
         "🔤 Acronyms — Systems & Organizations"),

        ("ILS", "Instrument Landing System",
         "A precision radio navigation system that provides aircraft with horizontal (Localizer, 108–117.975 MHz) and vertical (Glide Slope, 328.6–335.4 MHz) guidance for landing in low visibility. ILS enables Category I, II, and III approaches. It is a safety-of-life system protected under Radio Regulations No. 4.10.",
         "🔤 Acronyms — Systems & Organizations"),

        ("IMO", "International Maritime Organization",
         "A United Nations agency responsible for regulating shipping. IMO coordinates with ITU-R Working Party 5B (WP 5B) on maritime radio spectrum, including GMDSS and VDES.",
         "🔤 Acronyms — Systems & Organizations"),

        ("IMT", "International Mobile Telecommunications",
         "The ITU-R family of standards for mobile broadband, including 4G (Long Term Evolution / LTE) and 5G (New Radio / NR). IMT spectrum studies are conducted in ITU-R Working Party 5D (WP 5D). New IMT allocations near aeronautical bands are a primary FAA concern.",
         "🔤 Acronyms — Systems & Organizations"),

        ("JAXA", "Japan Aerospace Exploration Agency",
         "Japan's national aerospace and space agency. JAXA operates the Multi-functional Satellite Augmentation System (MSAS), Japan's Satellite-Based Augmentation System (SBAS) for GPS. Participates in ITU-R coordination affecting GNSS spectrum.",
         "🔤 Acronyms — Systems & Organizations"),

        ("LNA", "Low Noise Amplifier",
         "The first active component in a receiver chain, designed to amplify weak signals while adding minimal noise. The LNA sets the receiver's noise figure. A strong out-of-band interferer can drive the LNA into compression (saturation), causing it to generate intermodulation products and lose sensitivity — this is called receiver desensitization or blocking.",
         "🔤 Acronyms — Systems & Organizations"),

        ("MOPS", "Minimum Operational Performance Standards",
         "RTCA documents that specify the minimum performance requirements for avionics systems, including receiver sensitivity, selectivity, and interference immunity. MOPS are the primary source for protection criteria cited in ITU-R contributions. Examples: DO-235B (GNSS), DO-260B (ADS-B), DO-155 (Radio Altimeter).",
         "🔤 Acronyms — Systems & Organizations"),

        ("MSAS", "Multi-functional Satellite Augmentation System",
         "Japan's Satellite-Based Augmentation System (SBAS) for GPS, operated by JAXA. Transmits GPS correction signals on L1 (1575.42 MHz) from geostationary satellites to improve positioning accuracy for aviation.",
         "🔤 Acronyms — Systems & Organizations"),

        ("NASA", "National Aeronautics and Space Administration",
         "The United States government agency responsible for civilian space program and aeronautics research. NASA operates GNSS reference stations and participates in ITU-R coordination affecting GNSS spectrum.",
         "🔤 Acronyms — Systems & Organizations"),

        ("NTIA", "National Telecommunications and Information Administration",
         "The US government agency, part of the Department of Commerce, responsible for managing the federal government's use of radio spectrum. NTIA coordinates all US government positions for ITU-R meetings and submits official US contributions. FAA spectrum concerns are channeled through NTIA.",
         "🔤 Acronyms — Systems & Organizations"),

        ("OOBE", "Out-of-Band Emissions",
         "Radio frequency energy emitted by a transmitter outside its assigned channel or band. OOBE can cause in-band interference to a victim receiver even when the interferer is nominally operating in a different frequency band. OOBE limits are specified in ITU-R Radio Regulations and system-specific Recommendations.",
         "🔤 Acronyms — Systems & Organizations"),

        ("PSR", "Primary Surveillance Radar",
         "A radar system that detects aircraft by reflecting radio waves off the aircraft skin — no cooperation from the aircraft is required. PSR provides range and bearing but no altitude. Used by Air Traffic Control alongside Secondary Surveillance Radar (SSR).",
         "🔤 Acronyms — Systems & Organizations"),

        ("RNSS", "Radionavigation Satellite Service",
         "An ITU Radio Regulations allocation category covering satellite-based navigation systems (GPS, GLONASS, Galileo, BeiDou). RNSS allocations in the L-band are protected under Resolution 233 and the GNSS protection methodology in ITU-R M.1318.",
         "🔤 Acronyms — Systems & Organizations"),

        ("RTCA", "Radio Technical Commission for Aeronautics",
         "A US standards organization that develops aviation standards, including Minimum Operational Performance Standards (MOPS) for avionics. RTCA documents (called DO-xxx) define the receiver sensitivity, selectivity, and interference immunity requirements cited in ITU-R contributions to defend aeronautical frequencies.",
         "🔤 Acronyms — Systems & Organizations"),

        ("SBAS", "Satellite-Based Augmentation System",
         "A system that improves GPS accuracy and integrity for aviation by broadcasting correction signals from geostationary satellites. Examples include WAAS (Wide Area Augmentation System, United States), EGNOS (Europe), MSAS (Japan), and GAGAN (India). SBAS signals occupy the GPS L1 band (1575.42 MHz).",
         "🔤 Acronyms — Systems & Organizations"),

        ("SSR", "Secondary Surveillance Radar",
         "An Air Traffic Control radar system that interrogates aircraft transponders (Mode A, C, S) to obtain identification and altitude. SSR operates at 1030 MHz (interrogation) and 1090 MHz (reply). ADS-B (Automatic Dependent Surveillance — Broadcast) uses the 1090 MHz SSR reply frequency.",
         "🔤 Acronyms — Systems & Organizations"),

        ("TACAN", "Tactical Air Navigation",
         "A military radio navigation system providing both bearing and distance information, operating in the 960–1215 MHz band. Civilian DME (Distance Measuring Equipment) and TACAN are co-located at VORTAC stations, providing both VOR bearing and distance to civilian and military aircraft.",
         "🔤 Acronyms — Systems & Organizations"),

        ("TAWS", "Terrain Awareness and Warning System",
         "An avionics system that provides pilots with alerts about terrain, obstacles, and excessive descent rates. TAWS uses GPS position, a terrain database, and radio altimeter data to generate alerts. Radio altimeter accuracy is critical to TAWS performance at low altitude.",
         "🔤 Acronyms — Systems & Organizations"),

        ("TCAS", "Traffic Collision Avoidance System",
         "An airborne system that monitors the airspace around an aircraft for other transponder-equipped aircraft and provides Resolution Advisories (RA) to avoid collisions. TCAS operates at 1030 MHz (interrogation) and 1090 MHz (reply), sharing the SSR/ADS-B frequencies.",
         "🔤 Acronyms — Systems & Organizations"),

        ("USTR", "United States Trade Representative",
         "The US government office that coordinates international trade and telecommunications negotiations. The USTR, together with the State Department and NTIA, represents the US at World Radiocommunication Conferences (WRC).",
         "🔤 Acronyms — Systems & Organizations"),

        ("VDES", "VHF Data Exchange System",
         "A maritime communication system operating in the Very High Frequency (VHF) maritime band (156–174 MHz), intended to succeed the Automatic Identification System (AIS). VDES adds data exchange capabilities for e-navigation. Governed by ITU-R Working Party 5B (WP 5B).",
         "🔤 Acronyms — Systems & Organizations"),

        ("VHF", "Very High Frequency",
         "The radio frequency range from 30 to 300 MHz, with wavelengths of 1 to 10 meters. Aeronautical VHF communications (118–136 MHz) and VOR/ILS navigation (108–117.975 MHz) operate in this band. VHF propagation is primarily line-of-sight.",
         "🔤 Acronyms — Systems & Organizations"),

        ("VOR", "VHF Omnidirectional Range",
         "A Very High Frequency (VHF) radio navigation beacon operating at 108–117.975 MHz that transmits bearing information in all directions simultaneously. Aircraft use VOR to determine their magnetic bearing (called a radial) from the station. VOR stations form the backbone of the US and international airway system. Co-located with DME at VORTAC stations.",
         "🔤 Acronyms — Systems & Organizations"),

        ("VORTAC", "VOR and TACAN Co-located Station",
         "A navigation facility combining a VHF Omnidirectional Range (VOR) beacon with a Tactical Air Navigation (TACAN) transponder. Provides civilian aircraft with VOR bearing and DME distance, and military aircraft with TACAN bearing and distance.",
         "🔤 Acronyms — Systems & Organizations"),

        ("WAAS", "Wide Area Augmentation System",
         "The United States Satellite-Based Augmentation System (SBAS), operated by the FAA. WAAS broadcasts GPS correction and integrity signals from geostationary satellites on L1 (1575.42 MHz), enabling GPS approaches to as low as 200 feet Decision Height (Category I equivalent). WAAS reference stations are a specific FAA concern in L-band spectrum discussions.",
         "🔤 Acronyms — Systems & Organizations"),

        ("WP 5D", "Working Party 5D — IMT Systems",
         "The ITU-R Working Party responsible for International Mobile Telecommunications (IMT), including 4G (LTE), 5G (NR), and IMT-2030 (6G). WP 5D conducts sharing studies for new IMT spectrum that may threaten aeronautical bands including the Radio Altimeter band (4200–4400 MHz), ARNS at 5 GHz, and GNSS L-band.",
         "🔤 Acronyms — Systems & Organizations"),

        ("WP 5B", "Working Party 5B — Maritime Mobile and Radiodetermination",
         "The ITU-R Working Party responsible for maritime mobile communications (GMDSS, VDES, AIS) and radiodetermination (radar) services. WP 5B proposals near aeronautical VHF and radar bands require FAA monitoring.",
         "🔤 Acronyms — Systems & Organizations"),

        ("WP 7C", "Working Party 7C — Radiolocation",
         "The ITU-R Working Party responsible for radiolocation services including ground-based radar systems. WP 7C proposals can affect the Radio Altimeter band (4200–4400 MHz), Airborne Weather Radar band (9000–9500 MHz), and ATC radar bands (2700–2900 MHz).",
         "🔤 Acronyms — Systems & Organizations"),

        ("WP 7D", "Working Party 7D — Radio Astronomy",
         "The ITU-R Working Party responsible for radio astronomy and passive sensing services. WP 7D protection zones and emission limit proposals adjacent to aeronautical bands can indirectly constrain FAA system parameters.",
         "🔤 Acronyms — Systems & Organizations"),

        ("WRC", "World Radiocommunication Conference",
         "An international treaty conference held every 3–4 years under the International Telecommunication Union (ITU) that reviews and revises the Radio Regulations — the international treaty governing use of the radio frequency spectrum. WRC decisions are legally binding on all 193 ITU member states. WRC-27 is the next conference, scheduled for 2027.",
         "🔤 Acronyms — Systems & Organizations"),

        # ── RF & Signal Theory ────────────────────────────────────────────────
        ("Blocking / Desensitization", "Receiver Front-End Overload",
         "A condition where a strong out-of-band signal drives the receiver's Low Noise Amplifier (LNA) into compression (saturation), raising the effective noise floor and reducing sensitivity to the desired signal. The interferer does not need to be in-band to cause this — it only needs to be strong enough at the antenna input. This was the core mechanism in the 5G / Radio Altimeter interference problem of 2019–2022.",
         "📡 RF & Signal Theory"),

        ("C/I", "Carrier-to-Interference Ratio",
         "The ratio of the desired signal power to the interference power, expressed in decibels (dB). C/I = C(dBm) − I(dBm). Used in communications system design to assess whether the desired signal can be detected in the presence of interference. A higher C/I indicates less interference impact.",
         "📡 RF & Signal Theory"),

        ("C/N", "Carrier-to-Noise Ratio",
         "The ratio of the desired signal power to the thermal noise floor power, expressed in decibels (dB). C/N = C(dBm) − N(dBm). Determines whether a receiver can successfully decode a signal. GPS L1 operates at C/N of approximately 40–45 dB-Hz under normal conditions.",
         "📡 RF & Signal Theory"),

        ("dB", "Decibel",
         "A logarithmic unit expressing the ratio of two power levels: dB = 10 × log10(P2/P1). Because RF signal levels span many orders of magnitude (from kilowatts to femtowatts), the decibel scale makes calculations manageable. In decibels, multiplication of powers becomes addition, and division becomes subtraction.",
         "📡 RF & Signal Theory"),

        ("dBm", "Decibels Relative to One Milliwatt",
         "An absolute power level expressed in decibels referenced to 1 milliwatt. Formula: P(dBm) = 10 × log10(P(mW)). Common reference points: 0 dBm = 1 mW; 30 dBm = 1 W; 43 dBm = 20 W; −130 dBm = GPS signal at Earth's surface.",
         "📡 RF & Signal Theory"),

        ("dBi", "Decibels Relative to an Isotropic Antenna",
         "A unit of antenna gain relative to a theoretical isotropic antenna that radiates equally in all directions. 0 dBi = isotropic; 2 dBi = simple dipole; 15 dBi = typical cellular sector antenna; 30+ dBi = high-gain dish or phased array.",
         "📡 RF & Signal Theory"),

        ("dBW", "Decibels Relative to One Watt",
         "An absolute power level in decibels referenced to 1 watt. Relationship: 0 dBW = 30 dBm; 27 dBW = 57 dBm = 500 W (typical GPS satellite transmit power).",
         "📡 RF & Signal Theory"),

        ("EIRP", "Effective Isotropic Radiated Power",
         "The product of transmitter output power and antenna gain in a given direction, less cable losses. EIRP = Pt(dBm) + Gt(dBi) − Lcable(dB). EIRP represents the power that would need to be fed to an isotropic antenna to produce the same signal strength in the direction of interest. It is the standard metric for interference analysis and is used in ITU Radio Regulations power flux density limits.",
         "📡 RF & Signal Theory"),

        ("FSPL", "Free Space Path Loss",
         "The signal power loss between a transmitting and receiving antenna in free space (vacuum, no obstructions). Formula: FSPL(dB) = 20·log10(d_km) + 20·log10(f_MHz) + 32.44. FSPL is the most optimistic (least loss) propagation model and represents the worst-case interference scenario — if compatibility holds under FSPL, any realistic deployment will also be compatible.",
         "📡 RF & Signal Theory"),

        ("I/N", "Interference-to-Noise Ratio",
         "The ratio of interference power to thermal noise floor power at a receiver input, expressed in decibels (dB). I/N = I(dBm) − N(dBm). The ITU-R standard compatibility metric. At I/N = −6 dB, the noise floor rises by 0.97 dB. At I/N = −10 dB (GNSS standard), the rise is 0.41 dB. Exceeding the I/N threshold is the technical basis for citing harmful interference under Radio Regulations No. 4.10.",
         "📡 RF & Signal Theory"),

        ("Noise Figure (NF)", "Receiver Noise Figure",
         "A measure of how much noise a receiver adds to the signal beyond the fundamental thermal noise limit, expressed in decibels (dB). A 0 dB noise figure is theoretically perfect. Real avionics receivers typically have noise figures of 2–8 dB. Every 1 dB increase in noise figure raises the noise floor by 1 dB and reduces sensitivity by 1 dB.",
         "📡 RF & Signal Theory"),

        ("Noise Floor", "Receiver Thermal Noise Floor",
         "The minimum signal power a receiver can detect, set by thermal noise. Formula: N(dBm) = −174 + 10·log10(B_Hz) + NF(dB), where −174 dBm/Hz is thermal noise at 290 Kelvin, B is bandwidth in Hertz, and NF is the receiver noise figure in dB. Interference must remain below the I/N threshold relative to this level.",
         "📡 RF & Signal Theory"),

        ("PFD", "Power Flux Density",
         "The radio frequency power arriving per unit area at a given location, expressed in dB(W/m²) or dB(W/m²/MHz). PFD limits are specified in the ITU Radio Regulations to protect Earth-based receivers from satellite transmissions. RR Appendix 7 specifies PFD limits for non-geostationary satellite systems.",
         "📡 RF & Signal Theory"),

        ("PRF", "Pulse Repetition Frequency",
         "The number of radar pulses transmitted per second, expressed in Hertz (Hz) or pulses per second (pps). Used in pulsed radar systems such as ATC radar and airborne weather radar. The average power of a pulsed transmitter = Peak Power × Pulse Width × PRF (the duty cycle).",
         "📡 RF & Signal Theory"),

        ("SNR", "Signal-to-Noise Ratio",
         "The ratio of the desired signal power to the noise floor power, expressed in decibels (dB). SNR = S(dBm) − N(dBm). Interference raises the effective noise floor, thereby reducing SNR and degrading receiver performance. GPS requires SNR above approximately 25 dB for reliable tracking.",
         "📡 RF & Signal Theory"),

        # ── ITU-R & Regulatory ────────────────────────────────────────────────
        ("AI", "Agenda Item (WRC)",
         "A specific topic adopted by a World Radiocommunication Conference (WRC) for study and potential regulatory action at the following WRC. Agenda items are numbered (e.g., AI 1.2) and assigned to ITU-R Study Groups and Working Parties for technical study over the 3–4 year inter-conference period.",
         "🌐 ITU-R & Regulatory"),

        ("CPM Report", "Conference Preparatory Meeting Report",
         "The document produced by the ITU-R Conference Preparatory Meeting (CPM) summarizing all technical studies and presenting one or more 'methods' for each WRC Agenda Item. The CPM Report is the primary input to the WRC negotiation process.",
         "🌐 ITU-R & Regulatory"),

        ("Harmful Interference", "Harmful Interference — RR Article 1.169",
         "Interference which endangers the functioning of a radionavigation service or other safety service, OR seriously degrades, obstructs, or repeatedly interrupts a radiocommunication service operating in accordance with the Radio Regulations. This is the legally defined threshold for regulatory action under RR No. 4.10. When interference rises to this level, affected administrations have the right to require cessation of the interfering transmission. For FAA aeronautical systems, demonstrating that a new proposal causes harmful interference under RR 1.169 is the strongest possible policy lever.",
         "🌐 ITU-R & Regulatory"),

        ("Permissible Interference", "Permissible Interference — RR Article 1.167",
         "Observed or predicted interference which complies with quantitative interference and sharing criteria contained in the Radio Regulations or in ITU-R Recommendations or in special agreements as provided for in these Regulations. Permissible interference is below the harmful threshold — it is legally acceptable under the treaty framework. Both permissible interference and accepted interference are used in the coordination of frequency assignments between administrations.",
         "🌐 ITU-R & Regulatory"),

        ("Accepted Interference", "Accepted Interference — RR Article 1.168",
         "Interference at a higher level than that defined as permissible interference, which has been agreed upon between two or more administrations without prejudice to other administrations. Accepted interference is used in bilateral frequency coordination agreements where administrations choose to tolerate higher interference levels than the standard criteria in exchange for operational flexibility. It cannot be imposed on a third administration — only agreed bilaterally.",
         "🌐 ITU-R & Regulatory"),

        ("Interference (RR 1.166)", "Interference — RR Article 1.166",
         "The effect of unwanted energy due to one or a combination of emissions, radiations, or inductions upon reception in a radiocommunication system, manifested by any performance degradation, misinterpretation, or loss of information which could be extracted in the absence of such unwanted energy. The three regulatory categories — harmful (1.169), permissible (1.167), and accepted (1.168) — all fall under this parent definition.",
         "🌐 ITU-R & Regulatory"),

        ("Spurious Emissions", "Spurious Emissions (ITU-R definition)",
         "Emissions at frequencies outside the necessary bandwidth whose level can be reduced without affecting the corresponding transmission of information. Spurious emissions include harmonic emissions, parasitic emissions, intermodulation products, and frequency conversion products, but explicitly EXCLUDE out-of-band (OOB) emissions. Spurious emissions arise from hardware non-idealities (non-linearities, oscillator leakage) rather than the modulation process itself. They are regulated by ITU-R Recommendation SM.329 and are subject to limits in the Radio Regulations Appendix 3.",
         "🌐 ITU-R & Regulatory"),

        ("Out-of-Band (OOB) Emissions", "Out-of-Band Emissions (ITU-R definition)",
         "Emissions at frequencies immediately outside the necessary bandwidth which result from the modulation process, but excluding spurious emissions. OOB emissions are an unavoidable consequence of the modulation waveform — they arise from the spectral sidebands of the modulated signal. Unlike spurious emissions, OOB emissions cannot be eliminated without affecting the information content of the transmission. They can be reduced by filtering (at the cost of signal distortion) or by reducing modulation bandwidth. OOB emissions are the primary mechanism for interference into adjacent aeronautical bands — for example, 5G New Radio OOB emissions near the Radio Altimeter band (4200–4400 MHz).",
         "🌐 ITU-R & Regulatory"),

        ("Unwanted Emissions", "Unwanted Emissions",
         "The combination of spurious emissions and out-of-band (OOB) emissions. This is the umbrella category under which both emission types fall. In interference analysis for ITU-R contributions, you must characterize which type of unwanted emission is causing the problem, as they are subject to different regulatory limits and different mitigation approaches: OOB can be reduced by transmitter filtering or guard bands; spurious can be reduced by improving transmitter linearity or adding cavity filters.",
         "🌐 ITU-R & Regulatory"),

        ("In-Band Interference", "In-Band Interference (technical mechanism)",
         "Interference where the interfering signal's carrier or main lobe falls within the allocated frequency band of the victim service. In-band interference is the most direct form — the interferer and victim are co-frequency or co-channel. For FAA systems, in-band interference means the new service is proposed for allocation within a band already assigned to ARNS or RNSS. This is the most straightforward case for citing harmful interference under RR No. 4.10.",
         "🌐 ITU-R & Regulatory"),

        ("Adjacent Band Interference", "Adjacent Band / Near-Band Interference (technical mechanism)",
         "Interference where the interfering system operates in a different band but its out-of-band (OOB) or spurious emissions, or the victim receiver's finite selectivity, allow coupling of interfering energy into the victim receiver. The 5G C-band (3.7–3.98 GHz) / Radio Altimeter (4.2–4.4 GHz) case is a classic adjacent-band interference scenario: the 5G signal was outside the altimeter band, but OOB emissions and receiver front-end selectivity limitations allowed destructive coupling.",
         "🌐 ITU-R & Regulatory"),

        ("Intermodulation", "Intermodulation Interference (technical mechanism)",
         "Interference produced when two or more signals are applied to a non-linear device (amplifier, mixer, or antenna), producing new signals at frequencies that are sums and differences of the input frequencies and their harmonics. The most troublesome are third-order intermodulation products (2f1−f2 and 2f2−f1) which can fall in-band even when the original signals are outside the protected band. Intermodulation is particularly important at sites with multiple co-located transmitters.",
         "🌐 ITU-R & Regulatory"),

        ("Receiver Blocking", "Receiver Blocking / Desensitization (technical mechanism)",
         "A form of adjacent-band or out-of-band interference where a strong signal overloads the victim receiver's Low Noise Amplifier (LNA) or front-end stages, compressing their gain and raising the effective noise floor. Unlike intermodulation, blocking does not require the interferer to be at a specific frequency relationship to the victim — it only requires the interferer to be sufficiently powerful. Blocking is characterized by the receiver's 1 dB compression point (P1dB). The Radio Altimeter / 5G interference problem was primarily a blocking mechanism, not an in-band interference problem.",
         "🌐 ITU-R & Regulatory"),

        ("ITU", "International Telecommunication Union",
         "A United Nations specialized agency for information and communication technologies, headquartered in Geneva, Switzerland. The ITU-R (Radiocommunication Sector) manages the global radio frequency spectrum and satellite orbits through international treaty — the Radio Regulations.",
         "🌐 ITU-R & Regulatory"),

        ("ITU-R", "International Telecommunication Union — Radiocommunication Sector",
         "The ITU sector responsible for managing the radio frequency spectrum and satellite orbits. ITU-R produces international standards (Recommendations, Reports) and maintains the Radio Regulations through the World Radiocommunication Conference (WRC) process.",
         "🌐 ITU-R & Regulatory"),

        ("M.1318", "ITU-R Recommendation M.1318",
         "ITU-R Recommendation M.1318 — 'Model for the evaluation of continuous interference levels to radionavigation-satellite service receivers.' Defines the c = a − b calculation framework for assessing aggregate non-RNSS interference to GNSS receivers in the 1164–1215, 1215–1300, 1559–1610, and 5010–5030 MHz bands.",
         "🌐 ITU-R & Regulatory"),

        ("M.1477", "ITU-R Recommendation M.1477",
         "ITU-R Recommendation M.1477 — Annex 5 establishes the aeronautical safety margin for GNSS: at least 6 dB protection margin for safety-of-life applications, plus an additional 10 dB margin when the interfering signal bandwidth is 700 Hz or less (the narrowband rule).",
         "🌐 ITU-R & Regulatory"),

        ("M.1642", "ITU-R Recommendation M.1642",
         "ITU-R Recommendation M.1642 — 'Methodology for assessing the interference from IMT-2000 to aeronautical radionavigation systems.' The standard methodology for compatibility studies between IMT systems and ARNS, including radio altimeters and other aeronautical navigation aids.",
         "🌐 ITU-R & Regulatory"),

        ("M.1904", "ITU-R Recommendation M.1904",
         "ITU-R Recommendation M.1904 — Establishes protection criteria for GLONASS spaceborne receivers. Annex 1 Table 1 Note 3 specifies a safety margin of 6 dB. Part of the citation stack defending the 6 dB aeronautical safety margin doctrine alongside M.1477 and M.1905.",
         "🌐 ITU-R & Regulatory"),

        ("M.1905", "ITU-R Recommendation M.1905",
         "ITU-R Recommendation M.1905 — Recommends that a safety margin be applied for protection of the safety aspects and applications of RNSS when performing interference analyses. Note 1 specifies an aeronautical safety margin of 6 dB. Applies to ALL RNSS systems, making it the broadest authority for the 6 dB doctrine.",
         "🌐 ITU-R & Regulatory"),

        ("P.452", "ITU-R Recommendation P.452",
         "ITU-R Recommendation P.452 — 'Prediction procedure for the evaluation of interference between stations on the surface of the Earth at frequencies above about 0.1 GHz.' The standard terrestrial propagation model for point-to-area interference prediction. Used for ground-to-ground interference scenarios.",
         "🌐 ITU-R & Regulatory"),

        ("P.528", "ITU-R Recommendation P.528",
         "ITU-R Recommendation P.528 — 'Propagation curves for aeronautical mobile and radionavigation services using the VHF, UHF and SHF bands.' The standard propagation model for aeronautical scenarios. Required when the victim receiver is airborne. Accounts for slant-path geometry, atmospheric refraction, and radio horizon effects.",
         "🌐 ITU-R & Regulatory"),

        ("P.676", "ITU-R Recommendation P.676",
         "ITU-R Recommendation P.676 — 'Attenuation by atmospheric gases and related effects.' Specifies gaseous attenuation from oxygen and water vapor. Implemented in this tool via the itur Python library. Relevant above approximately 3 GHz; negligible for L-band and VHF/UHF paths.",
         "🌐 ITU-R & Regulatory"),

        ("Radio Regulations (RR)", "ITU Radio Regulations",
         "The international treaty governing use of the radio frequency spectrum and satellite orbits, binding on all 193 ITU member states. Key articles for aeronautical spectrum defense: RR No. 4.10 (no harmful interference to safety-of-life services), RR No. 5.444 (ARNS protection at 960–1215 MHz), RR No. 5.328 (ARNS at 108–137 MHz).",
         "🌐 ITU-R & Regulatory"),

        ("RR No. 4.10", "Radio Regulations Article No. 4.10",
         "The Radio Regulations provision stating that all stations must be established and operated so as not to cause harmful interference to safety-of-life services. This is the strongest regulatory instrument for protecting aeronautical frequencies — it applies regardless of allocation status and cannot be waived.",
         "🌐 ITU-R & Regulatory"),

        ("Resolution 233", "WRC Resolution 233",
         "A World Radiocommunication Conference Resolution requiring that new spectrum allocations and uses not degrade the performance of Radionavigation Satellite Service (RNSS) systems including GPS, GLONASS, and Galileo.",
         "🌐 ITU-R & Regulatory"),

        ("SM.2028", "ITU-R Recommendation SM.2028",
         "ITU-R Recommendation SM.2028 — 'Monte Carlo simulation methodology for use in sharing and compatibility studies between different radio services or systems.' The authoritative methodology for aggregate interference Monte Carlo analysis. Specifies that time percentage should be 1% for protection studies and that violation probability must be less than 5%.",
         "🌐 ITU-R & Regulatory"),

        ("SG 5", "Study Group 5 — Terrestrial Services",
         "An ITU-R Study Group responsible for terrestrial services including mobile, fixed, amateur, and broadcasting. Working Parties 5A, 5B, and 5D operate under SG 5. SG 5 plenary sessions approve and coordinate the work of these Working Parties.",
         "🌐 ITU-R & Regulatory"),

        ("TIES", "ITU Electronic Document System",
         "The ITU-R electronic document management system where all Working Party contributions, chairman's reports, and meeting documents are posted. Access to TIES requires ITU membership or delegation credentials.",
         "🌐 ITU-R & Regulatory"),

        # ── Aeronautical Systems ──────────────────────────────────────────────
        ("AIS", "Automatic Identification System",
         "A maritime tracking system that broadcasts vessel identity, position, speed, and heading on VHF frequencies (161.975 MHz and 162.025 MHz). AIS is being succeeded by VDES (VHF Data Exchange System) for expanded maritime data communications. Governed by ITU-R Working Party 5B (WP 5B).",
         "✈️ Aeronautical Systems"),

        ("ARSR", "Air Route Surveillance Radar",
         "Long-range Air Traffic Control radar used to track aircraft along airways. Operates in the 2700–2900 MHz band. The FAA operates the ARSR-4 system. Also refers to the L-band Air Route Surveillance Radar variant of WAAS reference station receivers.",
         "✈️ Aeronautical Systems"),

        ("ASR", "Airport Surveillance Radar",
         "Short-range Air Traffic Control radar used to track aircraft in the terminal area around an airport. Typically operates in the 2700–2900 MHz (S-band) or 9000–9500 MHz (X-band) range.",
         "✈️ Aeronautical Systems"),

        ("Radio Altimeter", "Radio Altimeter (Rad Alt)",
         "An aircraft instrument that measures the exact height above ground by transmitting radio pulses downward and measuring the time for the echo to return. Operates in 4200–4400 MHz. Critical for Category III instrument landings (near-zero visibility), Terrain Awareness and Warning System (TAWS), Enhanced Ground Proximity Warning System (EGPWS), and helicopter operations. A safety-of-life system protected under Radio Regulations No. 4.10.",
         "✈️ Aeronautical Systems"),

        # ── Measurements & Units ─────────────────────────────────────────────
        ("GHz", "Gigahertz",
         "A unit of frequency equal to one billion (10⁹) cycles per second, or 1000 Megahertz (MHz). Used for microwave and millimeter-wave frequencies. Example: Radio Altimeter at 4.2–4.4 GHz; 5G NR at 3.5 GHz or 26 GHz.",
         "📐 Measurements & Units"),

        ("Hz", "Hertz",
         "The base unit of frequency, equal to one cycle per second. Radio frequencies are typically expressed in kilohertz (kHz = 10³ Hz), Megahertz (MHz = 10⁶ Hz), or Gigahertz (GHz = 10⁹ Hz).",
         "📐 Measurements & Units"),

        ("kHz", "Kilohertz",
         "A unit of frequency equal to one thousand (10³) cycles per second. Used for low-frequency and medium-frequency radio systems. The 700 Hz narrowband threshold in ITU-R M.1477 Annex 5 is specified in Hertz — approximately 0.0007 MHz.",
         "📐 Measurements & Units"),

        ("kT", "Boltzmann's Constant × Temperature",
         "The product of Boltzmann's constant (k = 1.38 × 10⁻²³ Joules per Kelvin) and temperature in Kelvin (T). At 290 Kelvin (room temperature), kT = −174 dBm/Hz — the fundamental thermal noise spectral density. This is the starting point for all receiver noise floor calculations.",
         "📐 Measurements & Units"),

        ("MHz", "Megahertz",
         "A unit of frequency equal to one million (10⁶) cycles per second. The standard unit for specifying radio frequencies in the VHF, UHF, and microwave ranges. All FAA protected bands are specified in MHz in this tool.",
         "📐 Measurements & Units"),

        ("W", "Watt",
         "The SI unit of power. In radio engineering, transmitter power is often expressed in Watts or converted to decibels relative to one milliwatt (dBm): P(dBm) = 10 × log10(P(mW)). Common reference: 1 W = 30 dBm; 20 W = 43 dBm (typical LTE base station).",
         "📐 Measurements & Units"),

        # ── Analysis Methods ─────────────────────────────────────────────────
        ("CCDF", "Complementary Cumulative Distribution Function",
         "A statistical function showing the probability that a variable exceeds a given value. In Monte Carlo interference analysis, the CCDF of I/N is plotted to show the probability that aggregate interference exceeds the protection threshold. Reading the CCDF at the I/N threshold directly gives the violation probability — which must be less than 5% per ITU-R SM.2028.",
         "💻 Analysis Methods"),

        ("Monte Carlo", "Monte Carlo Simulation",
         "A statistical simulation method that uses random sampling to model uncertain or variable inputs. In ITU-R interference analysis (SM.2028), Monte Carlo simulates thousands of random interferer deployments, computing aggregate I/N for each trial, to produce a statistical distribution of interference outcomes and a violation probability.",
         "💻 Analysis Methods"),

        ("SEAMCAT", "Spectrum Engineering Advanced Monte Carlo Analysis Tool",
         "Free software developed by CEPT (European Conference of Postal and Telecommunications Administrations) for Monte Carlo aggregate interference analysis. Widely used in ITU-R Working Party contributions. Implements the methodology of ITU-R SM.2028.",
         "💻 Analysis Methods"),
    ]

    # ── Filter ────────────────────────────────────────────────────────────────
    filtered = []
    for term, full, definition, cat in GLOSSARY:
        if cat_filter != "All" and cat != cat_filter:
            continue
        if search and search not in term.lower() and search not in full.lower() and search not in definition.lower():
            continue
        filtered.append((term, full, definition, cat))

    st.markdown(f"**Showing {len(filtered)} of {len(GLOSSARY)} entries**")
    st.markdown("---")

    if not filtered:
        st.info("No entries match your search. Try a broader term.")
    else:
        # Group by category
        current_cat = None
        for term, full, definition, cat in filtered:
            if cat != current_cat:
                current_cat = cat
                st.subheader(cat)
            with st.expander(f"**{term}** — {full}"):
                st.markdown(definition)

    st.markdown("---")
    st.caption(f"FAA RF Interference Analysis Tool Glossary — {len(GLOSSARY)} terms across 6 categories.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 14 — MICROWAVE LINK BUDGET CALCULATOR
# ─────────────────────────────────────────────────────────────────────────────
elif selected_tab == "📻 Microwave Link Budget":
    st.title("📻 Microwave Link Budget Calculator")
    st.markdown("*Enter your site parameters and the tool instantly calculates EIRP, Free Space Path Loss, atmospheric absorption, and Received Signal Level (RSL) in both directions.*")
    ex("RSL = EIRP − Free Space Path Loss − Absorption + Rx Antenna Gain − Rx Losses. Enter any combination of sites — this calculator is not tied to any specific report format.")

    # ── Session state for saved links ─────────────────────────────────────────
    if "mw_saved_links" not in st.session_state:
        st.session_state.mw_saved_links = {}

    # ── Load saved link ───────────────────────────────────────────────────────
    saved_names = list(st.session_state.mw_saved_links.keys())
    load_choice = "(new link)"
    if saved_names:
        load_choice = st.selectbox("📂 Load saved link:", ["(new link)"] + saved_names)
    else:
        st.caption("No saved links yet — fill in parameters below and save.")

    # Defaults — load from saved if chosen
    D = {
        "site_a": "", "site_b": "",
        "location_a": "", "location_b": "",
        "lat_a": "", "lon_a": "", "lat_b": "", "lon_b": "",
        "elev_a": 0.0, "elev_b": 0.0,
        "struct_a": 0.0, "struct_b": 0.0,
        "path_override": False, "path_km_manual": 10.0,
        "band_ghz": 7.0, "freq_a": 7902.5, "freq_b": 8262.5,
        "tx_pwr_a": 30.0, "tx_pwr_b": 30.0,
        "ant_gain_a": 40.0, "ant_gain_b": 40.0,
        "ant_model_a": "", "ant_model_b": "",
        "other_loss_a": 2.0, "other_loss_b": 2.0,
        "branch_a": 0.0, "branch_b": 0.0,
        "radio_model": "", "bandwidth_mhz": 30.0,
        "modulation": "256QAM", "bitrate_mbps": 100.0,
        "spec_atten": 0.0102, "link_id": "",
    }
    if load_choice != "(new link)" and load_choice in st.session_state.mw_saved_links:
        for k, v in st.session_state.mw_saved_links[load_choice].items():
            D[k] = v

    # ── Coordinate helpers ────────────────────────────────────────────────────
    def parse_latlon(s):
        if not s or not s.strip(): return None
        s = s.strip().upper()
        hemi = None
        for suffix, sign in [(' N','N','N'),(' S','S','S'),(' E','E','E'),(' W','W','W')]:
            pass
        for ch, sign in [('N',1),('S',-1),('E',1),('W',-1)]:
            if s.endswith(' '+ch) or s.endswith(ch):
                hemi = sign; s = s.rstrip(ch).strip(); break
        s = s.replace('-',' ').replace('°',' ').replace("'",' ').replace('"',' ')
        parts = s.split()
        try:
            if len(parts)==1:   val = float(parts[0])
            elif len(parts)==2: val = float(parts[0]) + float(parts[1])/60
            else:               val = float(parts[0]) + float(parts[1])/60 + float(parts[2])/3600
            return -val if hemi==-1 else val
        except: return None

    def haversine_km(la, lo, lb, lb2):
        R = 6371.0
        p1,p2 = math.radians(la), math.radians(lb)
        dp,dl = math.radians(lb-la), math.radians(lb2-lo)
        a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
        return R*2*math.atan2(math.sqrt(a), math.sqrt(1-a))

    def dms(dd, is_lat):
        h = ("N" if dd>=0 else "S") if is_lat else ("E" if dd>=0 else "W")
        dd=abs(dd); d=int(dd); m=int((dd-d)*60); s=((dd-d)*60-m)*60
        return f"{d}°{m}'{s:.1f}\"{h}"

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1 — SITE IDENTITY & LOCATION
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("📍 Section 1 — Site Identity & Location")
    ex("Enter any site names and optional coordinates. Coordinates auto-calculate path length via the Haversine great-circle formula. All fields are free-form — not tied to any specific report.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Site A**")
        site_a     = st.text_input("Site A — Name / Identifier", value=D["site_a"], placeholder="e.g. BEDFORD or TX Site")
        location_a = st.text_input("Site A — Location Description", value=D["location_a"], placeholder="e.g. Bedford, Virginia, USA")
        lat_a_s    = st.text_input("Latitude A (optional)", value=D["lat_a"], placeholder="e.g. 37-31-21.9 N  or  37.5261")
        lon_a_s    = st.text_input("Longitude A (optional)", value=D["lon_a"], placeholder="e.g. 79-30-36.5 W  or  -79.5101")
        elev_a     = st.number_input("Ground Elevation A (m)", value=float(D["elev_a"]), step=1.0, help="Meters above mean sea level")
        struct_a   = st.number_input("Structure Height A (m AGL)", value=float(D["struct_a"]), step=0.5, help="Tower height above ground")
    with c2:
        st.markdown("**Site B**")
        site_b     = st.text_input("Site B — Name / Identifier", value=D["site_b"], placeholder="e.g. ROANOKE or RX Site")
        location_b = st.text_input("Site B — Location Description", value=D["location_b"], placeholder="e.g. Roanoke, Virginia, USA")
        lat_b_s    = st.text_input("Latitude B (optional)", value=D["lat_b"], placeholder="e.g. 37-18-32.7 N  or  37.3091")
        lon_b_s    = st.text_input("Longitude B (optional)", value=D["lon_b"], placeholder="e.g. 80-09-36.5 W  or  -80.1601")
        elev_b     = st.number_input("Ground Elevation B (m)", value=float(D["elev_b"]), step=1.0)
        struct_b   = st.number_input("Structure Height B (m AGL)", value=float(D["struct_b"]), step=0.5)

    # Parse coords
    la = parse_latlon(lat_a_s); lo = parse_latlon(lon_a_s)
    lb = parse_latlon(lat_b_s); lb2= parse_latlon(lon_b_s)
    coords_ok = all(v is not None for v in [la, lo, lb, lb2])
    auto_km = round(haversine_km(la, lo, lb, lb2), 3) if coords_ok else None

    if any([lat_a_s, lon_a_s, lat_b_s, lon_b_s]):
        st.markdown("**Coordinate Status**")
        cs1, cs2, cs3 = st.columns(3)
        with cs1:
            if la is not None and lo is not None:
                ok(f"Site A: {dms(la,True)}, {dms(lo,False)}")
            else:
                warn("Site A coordinates not parseable")
        with cs2:
            if lb is not None and lb2 is not None:
                ok(f"Site B: {dms(lb,True)}, {dms(lb2,False)}")
            else:
                warn("Site B coordinates not parseable")
        with cs3:
            if auto_km:
                ok(f"Path (Haversine): **{auto_km:.3f} km**")
            else:
                st.info("Enter both sites for auto path length")

    # Path length entry
    path_override = st.checkbox("Enter path length manually (override coordinate calculation)",
                                value=D["path_override"])
    if path_override or not auto_km:
        path_km = st.number_input("Path Length (km)",
                                  value=float(D["path_km_manual"]) if D["path_km_manual"] else (auto_km or 10.0),
                                  min_value=0.001, step=0.001, format="%.3f")
        if auto_km and abs(path_km - auto_km) > 0.1:
            warn(f"Manual path ({path_km:.3f} km) differs from coordinate calculation ({auto_km:.3f} km) by {abs(path_km-auto_km):.3f} km")
    else:
        path_km = auto_km
        st.metric("Path Length", f"{path_km:.3f} km", help="Haversine great-circle distance from coordinates")

    # Map
    if coords_ok:
        st.markdown("**🗺️ Link Map**")
        st.map(pd.DataFrame({'lat':[la,lb],'lon':[lo,lb2]}), zoom=7, use_container_width=True)
        st.caption(f"📍 {site_a or 'Site A'}: {la:.5f}°, {lo:.5f}°   |   📍 {site_b or 'Site B'}: {lb:.5f}°, {lb2:.5f}°   |   Path: {path_km:.3f} km")

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2 — RF PARAMETERS
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("📡 Section 2 — RF & Antenna Parameters")
    ex("Enter parameters for each end independently — supports asymmetric links where sites have different antennas, power levels, or losses.")

    c3, c4 = st.columns(2)
    with c3:
        st.markdown(f"**{site_a or 'Site A'} — Transmitter**")
        ant_model_a  = st.text_input("Antenna Model A", value=D["ant_model_a"], placeholder="e.g. Andrew HP6-71W")
        ant_gain_a   = st.number_input("Antenna Gain A (dBi)", value=float(D["ant_gain_a"]), step=0.1,
                                        help="Peak gain toward Site B. Use 0 dBi for isotropic worst-case.")
        tx_pwr_a     = st.number_input("Tx Power A (dBm)", value=float(D["tx_pwr_a"]), step=0.5,
                                        help="Radio output power in dBm. 30 dBm = 1 W, 43 dBm = 20 W")
        other_loss_a = st.number_input("Feeder / Connector Loss A (dB)", value=float(D["other_loss_a"]), step=0.1,
                                        help="Cable, connectors, jumpers between radio and antenna")
        branch_a     = st.number_input("Branching / Hybrid Loss A (dB)", value=float(D["branch_a"]), step=0.1,
                                        help="Combiner or branching losses. Zero for direct antenna connection.")
    with c4:
        st.markdown(f"**{site_b or 'Site B'} — Transmitter**")
        ant_model_b  = st.text_input("Antenna Model B", value=D["ant_model_b"], placeholder="e.g. Andrew HP10-71W")
        ant_gain_b   = st.number_input("Antenna Gain B (dBi)", value=float(D["ant_gain_b"]), step=0.1,
                                        help="Peak gain toward Site A")
        tx_pwr_b     = st.number_input("Tx Power B (dBm)", value=float(D["tx_pwr_b"]), step=0.5)
        other_loss_b = st.number_input("Feeder / Connector Loss B (dB)", value=float(D["other_loss_b"]), step=0.1)
        branch_b     = st.number_input("Branching / Hybrid Loss B (dB)", value=float(D["branch_b"]), step=0.1)

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3 — PATH & SYSTEM PARAMETERS
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("🌐 Section 3 — Path & System Parameters")
    ex("Band and path length drive the dominant Free Space Path Loss term. Specific attenuation is from ITU-R P.676 — 0.010 dB/km is a good starting value for 7 GHz under standard conditions.")

    c5, c6, c7 = st.columns(3)
    with c5:
        band_ghz   = st.number_input("Band (GHz)", value=float(D["band_ghz"]), step=0.5, min_value=0.1,
                                      help="Nominal band center frequency used for FSPL calculation")
        freq_a     = st.number_input("Tx Frequency A→B (MHz)", value=float(D["freq_a"]), step=0.5)
        freq_b     = st.number_input("Tx Frequency B→A (MHz)", value=float(D["freq_b"]), step=0.5)
    with c6:
        spec_atten = st.number_input("P.676 Attenuation (dB/km)", value=float(D["spec_atten"]),
                                      step=0.0001, format="%.4f",
                                      help="ITU-R P.676 specific gaseous attenuation. Typical values: 7 GHz ≈ 0.010, 15 GHz ≈ 0.015, 23 GHz ≈ 0.18, 60 GHz ≈ 15")
        radio_model= st.text_input("Radio Model", value=D["radio_model"], placeholder="e.g. Aviat ODU600v2")
        link_id    = st.text_input("Link ID / Reference", value=D["link_id"], placeholder="e.g. MW-001 or any identifier")
    with c7:
        bandwidth_mhz = st.number_input("Channel Bandwidth (MHz)", value=float(D["bandwidth_mhz"]), step=1.0)
        modulation    = st.text_input("Modulation", value=D["modulation"], placeholder="e.g. 256QAM, QPSK, 4096QAM")
        bitrate_mbps  = st.number_input("Bit Rate (Mb/s)", value=float(D["bitrate_mbps"]), step=1.0)

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # CALCULATIONS — run when path_km > 0
    # ══════════════════════════════════════════════════════════════════════════
    if path_km and path_km > 0 and band_ghz > 0:

        # Core link budget calculations
        eirp_a   = round(tx_pwr_a + ant_gain_a - other_loss_a - branch_a, 2)
        eirp_b   = round(tx_pwr_b + ant_gain_b - other_loss_b - branch_b, 2)
        fspl     = round(20*np.log10(path_km) + 20*np.log10(band_ghz*1000) + 32.44, 2)
        abs_db   = round(spec_atten * path_km, 3)
        tpl      = round(fspl + abs_db, 2)
        rsl_ab   = round(eirp_a - tpl + ant_gain_b - other_loss_b - branch_b, 2)
        rsl_ba   = round(eirp_b - tpl + ant_gain_a - other_loss_a - branch_a, 2)
        balanced = abs(rsl_ab - rsl_ba) < 0.05
        sA = site_a or "Site A"
        sB = site_b or "Site B"

        # ── TOP METRICS ───────────────────────────────────────────────────────
        st.subheader("⚡ Results")
        ex("All values recalculate instantly when any input changes. EIRP is the effective radiated power; RSL is the power arriving at the receiver input.")

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("EIRP — "+sA, f"{eirp_a:.2f} dBm",
            help=f"Pt({tx_pwr_a}) + Gt({ant_gain_a}) − Feeder({other_loss_a}) − Branching({branch_a}) = {eirp_a}")
        m2.metric("EIRP — "+sB, f"{eirp_b:.2f} dBm",
            help=f"Pt({tx_pwr_b}) + Gt({ant_gain_b}) − Feeder({other_loss_b}) − Branching({branch_b}) = {eirp_b}")
        m3.metric("Free Space Path Loss", f"{fspl:.2f} dB",
            help=f"20·log({path_km}) + 20·log({band_ghz*1000:.0f}) + 32.44")
        m4.metric("Absorption (P.676)", f"{abs_db:.3f} dB",
            help=f"{spec_atten} dB/km × {path_km} km")
        m5.metric(f"RSL at {sB}", f"{rsl_ab:.2f} dBm",
            delta_color="normal" if rsl_ab > -80 else "inverse",
            help=f"Power arriving at {sB} receiver input")
        m6.metric(f"RSL at {sA}", f"{rsl_ba:.2f} dBm",
            delta_color="normal" if rsl_ba > -80 else "inverse",
            help=f"Power arriving at {sA} receiver input (reverse path)")

        # Balance / capacity callouts
        if balanced:
            ok(f"Link is balanced — RSL is equal in both directions ({rsl_ab:.2f} dBm). Both sites use equal effective EIRP.")
        else:
            diff = abs(rsl_ab - rsl_ba)
            if diff < 1.0:
                warn(f"Link is slightly unbalanced — {diff:.2f} dB difference between forward and reverse RSL. Usually acceptable for ACM systems.")
            else:
                warn(f"Link imbalance of {diff:.2f} dB — check antenna gains and losses at each end. Significant imbalance reduces overall link availability.")

        # Capacity sanity
        mod_bits = {"4096QAM":12,"2048QAM":11,"1024QAM":10,"512QAM":9,"256QAM":8,
                    "128QAM":7,"64QAM":6,"32QAM":5,"16QAM":4,"QPSK":2,"BPSK":1}
        mod_key = modulation.strip().upper().split()[0] if modulation else ""
        bits = mod_bits.get(mod_key)
        if bits and bandwidth_mhz > 0:
            theo_max = bandwidth_mhz * bits
            if bitrate_mbps > theo_max:
                warn(f"Bit rate {bitrate_mbps:.0f} Mb/s exceeds theoretical {mod_key} maximum of {theo_max:.0f} Mb/s over {bandwidth_mhz:.0f} MHz — check inputs.")

        # ── STEP BY STEP WATERFALL ─────────────────────────────────────────────
        st.markdown("---")
        st.subheader("📊 Step-by-Step Link Budget")
        ex("Running total tracks cumulative signal power from transmitter output through to the receiver. Blue highlighted rows are EIRP and RSL — the two engineering milestones.")

        dir_tab_a, dir_tab_b = st.tabs([
            f"➡️  {sA} → {sB}",
            f"⬅️  {sB} → {sA}",
        ])

        def make_waterfall(tx_pwr, tx_gain, feeder, branch, fspl_v, abs_v,
                           rx_gain, rx_feeder, rx_branch, rsl_v, tx_name, rx_name):
            running = tx_pwr
            rows = []
            steps = [
                ("1", f"Transmit Power — {tx_name}",       f"+{tx_pwr:.2f}",     tx_pwr,       f"{running:.2f} dBm",   "Radio PA output"),
                ("2", "＋ Tx Antenna Gain",                  f"+{tx_gain:.2f}",    tx_gain,      None,                   "Peak gain toward far end (dBi)"),
                ("3", "－ Feeder / Connector Loss (Tx)",     f"−{feeder:.2f}",     -feeder,      None,                   "Cable, connectors, jumpers"),
                ("4", "－ Branching / Hybrid Loss (Tx)",     f"−{branch:.2f}",     -branch,      None,                   "Combiner losses (0 if direct)"),
                ("★", f"EIRP",                               f"{tx_pwr+tx_gain-feeder-branch:.2f} dBm", None, f"{tx_pwr+tx_gain-feeder-branch:.2f} dBm", "Effective Isotropic Radiated Power"),
                ("5", "－ Free Space Path Loss (FSPL)",      f"−{fspl_v:.2f}",     -fspl_v,      None,                   f"20·log({path_km})+20·log({band_ghz*1000:.0f})+32.44"),
                ("6", "－ Atmospheric Absorption (P.676)",   f"−{abs_v:.3f}",      -abs_v,       None,                   f"{spec_atten} dB/km × {path_km} km"),
                ("7", "Total Propagation Loss",              f"−{fspl_v+abs_v:.2f} dB", None,   "—",                    "FSPL + Absorption"),
                ("8", "＋ Rx Antenna Gain",                   f"+{rx_gain:.2f}",    rx_gain,      None,                   f"Peak gain toward {tx_name} (dBi)"),
                ("9", "－ Feeder / Connector Loss (Rx)",     f"−{rx_feeder:.2f}",  -rx_feeder,   None,                   "Rx side cable and connectors"),
                ("10","－ Branching / Hybrid Loss (Rx)",     f"−{rx_branch:.2f}",  -rx_branch,   None,                   "Rx combiner losses"),
                ("★", f"RSL at {rx_name}",                   f"{rsl_v:.2f} dBm",   None,         f"{rsl_v:.2f} dBm",     "Power at receiver input"),
            ]
            for snum, label, value, delta, run_override, note in steps:
                if delta is not None:
                    running = round(running + delta, 2)
                run_str = run_override if run_override is not None else f"{running:.2f} dBm"
                rows.append({"Step": snum, "Description": label,
                             "Value": value, "Running Total": run_str, "Notes": note})

            df = pd.DataFrame(rows)

            def style_row(row):
                if row["Step"] == "★" and "EIRP" in row["Description"]:
                    return ["background-color:#1a3560;font-weight:bold;color:#aaddff"]*len(row)
                if row["Step"] == "★" and "RSL" in row["Description"]:
                    return ["background-color:#1a4a1a;font-weight:bold;color:#aaffaa"]*len(row)
                if row["Step"] == "7":
                    return ["background-color:#3a1a1a;color:#ffaaaa"]*len(row)
                return [""]*len(row)

            st.dataframe(df.style.apply(style_row, axis=1),
                         use_container_width=True, hide_index=True)

        with dir_tab_a:
            make_waterfall(tx_pwr_a, ant_gain_a, other_loss_a, branch_a,
                           fspl, abs_db, ant_gain_b, other_loss_b, branch_b,
                           rsl_ab, sA, sB)
        with dir_tab_b:
            make_waterfall(tx_pwr_b, ant_gain_b, other_loss_b, branch_b,
                           fspl, abs_db, ant_gain_a, other_loss_a, branch_a,
                           rsl_ba, sB, sA)

        # ── P.676 SECTION ─────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🌡️ Atmospheric Absorption (ITU-R P.676)")
        ex("Specific attenuation × path length = total absorption loss. At 7 GHz this is small (≈0.6 dB over 60 km). At 23 GHz or 60 GHz it becomes the dominant loss term.")

        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Specific Attenuation", f"{spec_atten:.4f} dB/km",
            help="Change B56 to model different frequencies, temperatures, or humidity")
        p2.metric("Path Length", f"{path_km:.3f} km")
        p3.metric("Total Absorption", f"{abs_db:.3f} dB")
        p4.metric("% of Total Path Loss", f"{100*abs_db/tpl:.1f}%",
            help="Absorption as a fraction of total propagation loss")

        # P.676 reference table
        with st.expander("📋 P.676 Reference — Specific Attenuation at Common Frequencies"):
            st.markdown("""
| Frequency | Specific Attenuation | Notes |
|---|---|---|
| 2 GHz | ~0.003 dB/km | Negligible |
| 7 GHz | ~0.010 dB/km | Standard microwave backbone |
| 11 GHz | ~0.012 dB/km | Common microwave band |
| 15 GHz | ~0.015 dB/km | Upper microwave |
| 23 GHz | ~0.18 dB/km | Rain scatter starts to matter |
| 38 GHz | ~0.12 dB/km | Short-haul millimeter wave |
| 60 GHz | ~14 dB/km | Oxygen absorption peak — very short range |
| 80 GHz | ~0.5 dB/km | E-band license-free |

*Values at standard atmosphere (290K, 50% relative humidity, sea level)*
            """)

        # ── RSL SWEEP CHART ───────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("📉 RSL vs Path Length")
        ex("Shows how RSL degrades as distance increases at the current settings. The vertical marker shows your current path. Use this to find the maximum path length before RSL falls below a minimum threshold.")

        max_sweep = max(path_km * 3, 50.0)
        d_sw = np.linspace(0.1, max_sweep, 500)
        rsl_ab_sw = eirp_a - (20*np.log10(d_sw)+20*np.log10(band_ghz*1000)+32.44) \
                    - spec_atten*d_sw + ant_gain_b - other_loss_b - branch_b
        rsl_ba_sw = eirp_b - (20*np.log10(d_sw)+20*np.log10(band_ghz*1000)+32.44) \
                    - spec_atten*d_sw + ant_gain_a - other_loss_a - branch_a

        min_thresh = st.number_input("Minimum RSL threshold (dBm)",
                                      value=-75.0, step=1.0,
                                      help="Typical minimum receive threshold for operation. Enter your radio's threshold for the chosen modulation.")

        fig_sw, ax_sw = plt.subplots(figsize=(12, 4))
        fig_sw.patch.set_facecolor("#0e1117"); ax_sw.set_facecolor("#0e1117")
        ax_sw.plot(d_sw, rsl_ab_sw, color="#4488ff", lw=2, label=f"RSL at {sB} ({sA}→{sB})")
        if not balanced:
            ax_sw.plot(d_sw, rsl_ba_sw, color="#ff8844", lw=2, ls="--",
                       label=f"RSL at {sA} ({sB}→{sA})")
        ax_sw.axvline(path_km, color="white", lw=1.2, ls=":", label=f"Current: {path_km:.1f} km")
        ax_sw.axhline(rsl_ab,  color="#4488ff", lw=0.8, ls=":", alpha=0.5)
        ax_sw.axhline(min_thresh, color="red", lw=1.2, ls="--", alpha=0.8,
                      label=f"Min threshold: {min_thresh:.0f} dBm")

        # Mark max range at threshold
        try:
            cross_idx = np.where(rsl_ab_sw < min_thresh)[0][0]
            max_range = d_sw[cross_idx]
            ax_sw.axvline(max_range, color="red", lw=1, ls=":", alpha=0.6)
            ax_sw.annotate(f"  Max range\n  {max_range:.1f} km",
                           xy=(max_range, min_thresh+3), color="red", fontsize=8)
        except IndexError:
            pass

        ax_sw.annotate(f"  {rsl_ab:.2f} dBm",
                       xy=(path_km, rsl_ab), color="white", fontsize=8, va="bottom")
        ax_sw.set_xlabel("Path Length (km)", color="white", fontsize=10)
        ax_sw.set_ylabel("RSL (dBm)", color="white", fontsize=10)
        ax_sw.set_title(f"RSL vs Distance — {sA} ↔ {sB}  |  {band_ghz} GHz  |  EIRP={eirp_a:.1f}/{eirp_b:.1f} dBm",
                        color="white", fontsize=11)
        ax_sw.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
        ax_sw.tick_params(colors="white"); ax_sw.grid(color="#333", alpha=0.4)
        for sp in ax_sw.spines.values(): sp.set_color("#444")
        plt.tight_layout()
        st.pyplot(fig_sw)

        # ── SYSTEM SUMMARY CARD ───────────────────────────────────────────────
        st.markdown("---")
        st.subheader("📋 Link Summary")
        ex("Quick-reference card for this link showing all key parameters and results.")

        summary_data = {
            "Link ID": link_id or "—",
            f"Site A": f"{sA}{' — '+location_a if location_a else ''}",
            f"Site B": f"{sB}{' — '+location_b if location_b else ''}",
            "Path Length": f"{path_km:.3f} km",
            "Band / Frequency": f"{band_ghz} GHz  ({freq_a:.1f}/{freq_b:.1f} MHz)",
            "Free Space Path Loss": f"{fspl:.2f} dB",
            "Atmospheric Absorption": f"{abs_db:.3f} dB",
            "Total Propagation Loss": f"{tpl:.2f} dB",
            f"EIRP — {sA}": f"{eirp_a:.2f} dBm  ({ant_model_a or 'antenna'} {ant_gain_a} dBi)",
            f"EIRP — {sB}": f"{eirp_b:.2f} dBm  ({ant_model_b or 'antenna'} {ant_gain_b} dBi)",
            f"RSL at {sB} (A→B)": f"{rsl_ab:.2f} dBm",
            f"RSL at {sA} (B→A)": f"{rsl_ba:.2f} dBm",
            "Link Balance": f"{'Balanced ✓' if balanced else f'Unbalanced — {abs(rsl_ab-rsl_ba):.2f} dB difference'}",
            "Radio": radio_model or "—",
            "Bandwidth / Modulation": f"{bandwidth_mhz:.0f} MHz  /  {modulation}",
            "Bit Rate": f"{bitrate_mbps:.0f} Mb/s",
        }
        if bits:
            summary_data["Spectral Efficiency"] = f"{bitrate_mbps/bandwidth_mhz:.2f} b/s/Hz  (max for {mod_key}: {bits:.0f})"

        for k, v in summary_data.items():
            highlight = "RSL" in k or "EIRP" in k
            bg = "#1a3560" if "EIRP" in k else ("#1a4a1a" if "RSL" in k else "#1a1a2e")
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"padding:5px 12px;margin:2px 0;border-radius:4px;"
                f"background:{bg};border-left:3px solid {'#4488ff' if 'EIRP' in k else '#44bb44' if 'RSL' in k else '#2a2a3a'}'>"
                f"<span style='color:#aaa;font-size:0.85em'>{k}</span>"
                f"<span style='color:white;font-weight:{'bold' if highlight else 'normal'};font-size:0.9em'>{v}</span>"
                f"</div>", unsafe_allow_html=True
            )

        # ── FORMULA REFERENCE ─────────────────────────────────────────────────
        with st.expander("📐 Formula Reference"):
            st.latex(r"\text{EIRP (dBm)} = P_t + G_t - L_{feeder} - L_{branch}")
            st.latex(r"\text{FSPL (dB)} = 20\log_{10}(d_{km}) + 20\log_{10}(f_{MHz}) + 32.44")
            st.latex(r"\text{Absorption (dB)} = \gamma \times d_{km} \quad (\gamma = \text{specific attenuation, dB/km})")
            st.latex(r"\text{RSL (dBm)} = \text{EIRP} - \text{FSPL} - \text{Absorption} + G_r - L_{feeder,Rx} - L_{branch,Rx}")

    else:
        st.info("👆 Enter your path length and band in Section 3 above to see results.")

    # ── SAVE / DELETE ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("💾 Save This Link")
    ex("Save all current parameters to session memory. Reload any saved link from the dropdown at the top.")
    sv1, sv2 = st.columns([3, 1])
    with sv1:
        save_name = st.text_input("Save as:",
            value=f"{(site_a or 'Site A')}–{(site_b or 'Site B')}",
            placeholder="Give this link a name")
    with sv2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("💾 Save", type="primary", use_container_width=True):
            st.session_state.mw_saved_links[save_name] = {
                "site_a": site_a, "site_b": site_b,
                "location_a": location_a, "location_b": location_b,
                "lat_a": lat_a_s, "lon_a": lon_a_s,
                "lat_b": lat_b_s, "lon_b": lon_b_s,
                "elev_a": elev_a, "elev_b": elev_b,
                "struct_a": struct_a, "struct_b": struct_b,
                "path_override": path_override,
                "path_km_manual": float(path_km),
                "band_ghz": band_ghz, "freq_a": freq_a, "freq_b": freq_b,
                "tx_pwr_a": tx_pwr_a, "tx_pwr_b": tx_pwr_b,
                "ant_gain_a": ant_gain_a, "ant_gain_b": ant_gain_b,
                "ant_model_a": ant_model_a, "ant_model_b": ant_model_b,
                "other_loss_a": other_loss_a, "other_loss_b": other_loss_b,
                "branch_a": branch_a, "branch_b": branch_b,
                "radio_model": radio_model, "bandwidth_mhz": bandwidth_mhz,
                "modulation": modulation, "bitrate_mbps": bitrate_mbps,
                "spec_atten": spec_atten, "link_id": link_id,
            }
            ok(f"'{save_name}' saved. Select from the dropdown at top to reload.")

    if saved_names and load_choice != "(new link)":
        if st.button(f"🗑️ Delete '{load_choice}'"):
            del st.session_state.mw_saved_links[load_choice]
            st.rerun()



# ─────────────────────────────────────────────────────────────────────────────
# TAB 13 — ADMIN PANEL (admin only)
# ─────────────────────────────────────────────────────────────────────────────
elif selected_tab == "⚙️ Admin Panel":
    if is_admin():
        show_admin_panel()
    else:
        st.error("⛔ Access denied — admin role required.")

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.caption(
    "Tool uses itur library (P.676, P.618) for ITU-R propagation models. "
    "P.452 and P.528 implementations are simplified analytical approximations. "
    "For full regulatory submissions, validate with ITU-R SoftTools and SEAMCAT."
)
