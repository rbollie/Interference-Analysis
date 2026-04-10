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
        "in_threshold_db": -6,
        "noise_floor_dbm": -120,
        "safety_of_life": True,
        "notes": "En-route navigation and precision approach guidance",
        "rtca_doc": "DO-196",
    },
    "ILS Glide Slope": {
        "f_low_mhz": 328.6, "f_high_mhz": 335.4,
        "system": "ILS Glide Slope",
        "allocation": "ARNS",
        "in_threshold_db": -6,
        "noise_floor_dbm": -115,
        "safety_of_life": True,
        "notes": "Vertical guidance for precision landings (CAT I/II/III)",
        "rtca_doc": "DO-148",
    },
    "DME / TACAN": {
        "f_low_mhz": 960.0, "f_high_mhz": 1215.0,
        "system": "DME / TACAN / SSR / TCAS",
        "allocation": "ARNS + ANS",
        "in_threshold_db": -6,
        "noise_floor_dbm": -106,
        "safety_of_life": True,
        "notes": "Distance measuring, ATC surveillance, collision avoidance",
        "rtca_doc": "DO-189 / DO-185B",
    },
    "ADS-B / Mode-S (1090 MHz)": {
        "f_low_mhz": 1085.0, "f_high_mhz": 1095.0,
        "system": "ADS-B / Mode-S Transponder",
        "allocation": "ARNS + ANS",
        "in_threshold_db": -10,
        "noise_floor_dbm": -100,
        "safety_of_life": True,
        "notes": "1090 MHz squitter — global ATC surveillance backbone",
        "rtca_doc": "DO-260B",
    },
    "GNSS L5 / ARNS": {
        "f_low_mhz": 1164.0, "f_high_mhz": 1215.0,
        "system": "GNSS L5 / Galileo E5",
        "allocation": "ARNS + RNSS",
        "in_threshold_db": -10,
        "noise_floor_dbm": -130,
        "safety_of_life": True,
        "notes": "Safety-of-life GNSS signal; aviation approach procedures",
        "rtca_doc": "DO-292",
    },
    "GPS L1 / GNSS": {
        "f_low_mhz": 1559.0, "f_high_mhz": 1610.0,
        "system": "GPS L1 / GLONASS / Galileo E1",
        "allocation": "RNSS + ARNS",
        "in_threshold_db": -10,
        "noise_floor_dbm": -130,
        "safety_of_life": True,
        "notes": "Primary GNSS band — most sensitive; heavily protected",
        "rtca_doc": "DO-235B / DO-253",
    },
    "En-Route Radar": {
        "f_low_mhz": 2700.0, "f_high_mhz": 2900.0,
        "system": "ATC En-Route Surveillance Radar",
        "allocation": "ARNS + RN",
        "in_threshold_db": -6,
        "noise_floor_dbm": -100,
        "safety_of_life": True,
        "notes": "ASR / ARSR long-range ATC radar",
        "rtca_doc": "N/A (ITU-R M.1849)",
    },
    "Radio Altimeter": {
        "f_low_mhz": 4200.0, "f_high_mhz": 4400.0,
        "system": "Radio Altimeter (Rad Alt)",
        "allocation": "ARNS",
        "in_threshold_db": -6,
        "noise_floor_dbm": -90,
        "safety_of_life": True,
        "notes": "Critical for CAT III landings, TAWS, GPWS, helicopter ops",
        "rtca_doc": "DO-155 / ETSO-C87",
    },
    "ARNS 5 GHz": {
        "f_low_mhz": 5000.0, "f_high_mhz": 5150.0,
        "system": "ARNS / Future Aeronautical Systems",
        "allocation": "ARNS",
        "in_threshold_db": -6,
        "noise_floor_dbm": -95,
        "safety_of_life": True,
        "notes": "Protected for future aeronautical use; under pressure from IMT",
        "rtca_doc": "N/A",
    },
    "Airborne Weather Radar": {
        "f_low_mhz": 9000.0, "f_high_mhz": 9500.0,
        "system": "Airborne / Surface Movement Radar",
        "allocation": "ARNS + RN",
        "in_threshold_db": -6,
        "noise_floor_dbm": -95,
        "safety_of_life": True,
        "notes": "X-band weather radar and airport surface detection",
        "rtca_doc": "DO-220",
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
st.sidebar.markdown("---")

tab_names = [
    "📡 Protected Bands",
    "🔗 Link Budget",
    "📊 Noise & I/N",
    "🌐 Propagation",
    "🎲 Monte Carlo",
    "📋 Contribution Summary",
]
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
    ex("This database captures the key aeronautical allocations you are defending in ITU-R proceedings — your starting reference before any analysis.")

    # Spectrum overview chart
    st.subheader("Spectrum Overview")
    ex("Visual map of all protected FAA bands — use this to quickly spot which new proposals fall near or within these zones.")

    fig, ax = plt.subplots(figsize=(14, 3))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    colors = plt.cm.Set2(np.linspace(0, 1, len(FAA_BANDS)))

    for i, (name, b) in enumerate(FAA_BANDS.items()):
        fl, fh = b["f_low_mhz"], b["f_high_mhz"]
        ax.barh(0, fh - fl, left=fl, height=0.5, color=colors[i],
                alpha=0.85, label=name)
        mid = (fl + fh) / 2
        ax.text(mid, 0.32, name.split("/")[0].strip(),
                ha='center', va='bottom', fontsize=6.5, color='white', rotation=45)

    ax.set_xscale("log")
    ax.set_xlim(100, 12000)
    ax.set_xlabel("Frequency (MHz) — log scale", color='white')
    ax.set_yticks([])
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    for sp in ['top', 'left', 'right']:
        ax.spines[sp].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Band Details")
    ex("I/N threshold is the maximum tolerable interference-to-noise ratio; exceeding it can degrade or deny the aeronautical service.")

    selected_band = st.selectbox("Select a band for details:", list(FAA_BANDS.keys()))
    b = FAA_BANDS[selected_band]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Lower Freq", f"{b['f_low_mhz']} MHz")
    col2.metric("Upper Freq", f"{b['f_high_mhz']} MHz")
    col3.metric("Bandwidth", f"{b['f_high_mhz'] - b['f_low_mhz']:.1f} MHz")
    col4.metric("I/N Threshold", f"{b['in_threshold_db']} dB")

    col5, col6, col7 = st.columns(3)
    col5.metric("Allocation", b["allocation"])
    col6.metric("Noise Floor (est.)", f"{b['noise_floor_dbm']} dBm")
    col7.metric("RTCA Standard", b["rtca_doc"])

    st.markdown(f"**System:** {b['system']}")
    st.markdown(f"**Notes:** {b['notes']}")
    if b["safety_of_life"]:
        st.markdown("🔴 **Safety-of-Life Service** — protected under RR No. 4.10")

    # Full table
    st.subheader("All Bands Summary Table")
    ex("Use this table when drafting a US contribution — it gives you the protection criteria to cite for each system.")
    rows = []
    for name, b in FAA_BANDS.items():
        rows.append({
            "Band": name,
            "Low (MHz)": b["f_low_mhz"],
            "High (MHz)": b["f_high_mhz"],
            "BW (MHz)": round(b["f_high_mhz"] - b["f_low_mhz"], 3),
            "Allocation": b["allocation"],
            "I/N Threshold (dB)": b["in_threshold_db"],
            "Noise Floor (dBm)": b["noise_floor_dbm"],
            "Safety-of-Life": "✅" if b["safety_of_life"] else "—",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — LINK BUDGET
# ─────────────────────────────────────────────────────────────────────────────
elif selected_tab == "🔗 Link Budget":
    st.title("🔗 Link Budget Calculator")
    ex("A link budget accounts for every gain and loss between transmitter and receiver — the foundation of any interference analysis.")

    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.subheader("Transmitter (Interferer)")
        ex("The interferer is the new system whose emissions you are assessing — e.g., a new IMT base station or satellite downlink.")
        tx_power_dbm = st.number_input("Tx Power (dBm)", value=43.0, step=1.0,
            help="e.g., 43 dBm = 20W typical LTE base station")
        tx_gain_dbi = st.number_input("Tx Antenna Gain (dBi)", value=15.0, step=0.5,
            help="Directional gain toward victim receiver")
        cable_loss = st.number_input("Cable / Feeder Loss (dB)", value=2.0, step=0.5,
            help="Losses between PA output and antenna port")
        tx_height_m = st.number_input("Tx Height (m AGL)", value=30.0, step=5.0)

        st.subheader("Channel")
        ex("Path loss describes how signal power dissipates over distance — choose the model that best fits the scenario.")
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
        ex("The victim is your protected aeronautical system — set its parameters using RTCA standards or ITU-R Recommendations.")
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

        ex("Protection Margin = I/N threshold − actual I/N. Positive means protected; negative means the threshold is violated.")

        if margin >= 10:
            ok(f"PROTECTED with {margin:.1f} dB margin — system is well-protected at this distance.")
        elif margin >= 0:
            warn(f"MARGINALLY PROTECTED with only {margin:.1f} dB margin — consider conservative assumptions.")
        else:
            warn(f"THRESHOLD VIOLATED by {abs(margin):.1f} dB — this scenario poses an interference risk to the protected service.")

        # Waterfall chart
        st.subheader("Link Budget Waterfall")
        ex("Each bar represents a gain (+) or loss (−) in the signal path from transmitter to receiver noise floor.")
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
        ex("Sweep shows how much isolation distance is needed to bring I/N below the protection threshold.")
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
    ex("I/N (Interference-to-Noise) is the ITU-R standard metric for compatibility — it tells you how much a new service degrades an existing receiver's noise environment.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Receiver Noise Floor Calculator")
        ex("The noise floor is the minimum signal a receiver can detect — interference must stay well below this to be acceptable.")
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
        ex("Noise density = kT in dBm/Hz; floor rises with bandwidth. GPS receivers use ~290 K; some avionics use lower.")

    with col2:
        st.subheader("I/N Threshold Selector")
        ex("Protection thresholds vary by system type — GNSS is most sensitive (−10 dB) due to the extremely low signal levels from satellites.")
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
    ex("This heatmap shows how I/N changes as receiver bandwidth and interference power vary — helps identify worst-case scenarios.")
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

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — PROPAGATION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
elif selected_tab == "🌐 Propagation":
    st.title("🌐 Propagation Analysis")
    ex("Choosing the right propagation model is critical — using FSPL when P.528 is appropriate can underestimate interference by 10–20 dB.")

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

    ex("Lower path loss = more interference reaching the victim. Always use the most optimistic (lowest) path loss for a conservative protection analysis.")

    # Atmospheric attenuation using itur P.676
    st.subheader("Atmospheric Gaseous Attenuation (ITU-R P.676 via itur)")
    ex("Above ~1 GHz, oxygen and water vapor absorb signal energy — this adds loss that helps protect against terrestrial interference but matters for airborne paths too.")

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
    ex("This decision table follows ITU-R practice — using the wrong model is a common error that reviewers will challenge in Working Party sessions.")
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
    ex("Monte Carlo simulates thousands of random interferer deployments to estimate the probability that aggregate interference violates the I/N threshold — the ITU-R standard methodology per SM.2028.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Interferer Parameters")
        n_interferers = st.number_input("Number of Interferers (N)", value=10, min_value=1, max_value=500, step=5)
        ex("N is the density of new transmitters in a deployment area — a key assumption that drives aggregate interference.")
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

        ex("The 99th percentile I/N is the value to cite in a US contribution — it represents the worst-case aggregate scenario across realistic deployments.")

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

        ex("The CCDF (right plot) is how ITU-R contributions typically present Monte Carlo results — read across from the threshold line to get exceedance probability.")

        # Sensitivity: vary N interferers
        st.subheader("Sensitivity: Violation Probability vs Number of Interferers")
        ex("This shows how quickly risk grows as deployment density increases — use to argue for density limits or coordination zones in a contribution.")
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
    ex("An ITU-R contribution is the formal document submitted by an administration to a Working Party — this module helps you draft the key technical sections.")

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
        ex("This is a starting scaffold — fill in the bracketed placeholders and attach your full analysis figures before submission through NTIA.")

    # Regulatory citation quick reference
    st.subheader("Regulatory Citation Quick Reference")
    ex("Always cite the correct RR article or Resolution — this is what gives your technical argument legal weight in the WRC process.")
    reg_refs = pd.DataFrame([
        ["RR No. 4.10", "No harmful interference to safety-of-life services", "Strongest lever — invoked for ALL FAA safety systems"],
        ["RR No. 5.444", "ARNS protection at 960–1215 MHz", "Use for DME/TACAN/SSR/TCAS/ADS-B bands"],
        ["RR No. 5.328", "ARNS at 108–137 MHz", "VOR/ILS protection basis"],
        ["RR Resolution 233", "Protection of RNSS (GPS/GNSS)", "Use for all GPS/GNSS band defense"],
        ["RR Resolution 750", "IMT and safety services coexistence", "Relevant for all WP 5D IMT proposals"],
        ["ITU-R M.1642", "IMT→ARNS methodology", "Cite as methodology basis for your analysis"],
        ["ITU-R SM.2028", "Monte Carlo methodology", "Cite to validate your simulation approach"],
        ["ITU-R P.528", "Aeronautical propagation model", "Model authority — cite when using P.528 curves"],
        ["ICAO Annex 10", "Aeronautical telecomm standards", "Aligns ITU-R work with ICAO civil aviation requirements"],
    ], columns=["Reference", "Subject", "When to Cite"])
    st.table(reg_refs)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.caption(
    "Tool uses itur library (P.676, P.618) for ITU-R propagation models. "
    "P.452 and P.528 implementations are simplified analytical approximations. "
    "For full regulatory submissions, validate with ITU-R SoftTools and SEAMCAT."
)
