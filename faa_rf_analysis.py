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
    "📚 Tutorial",
    "🤖 Contribution Analyzer",
    "🎓 RF Training",
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
# TAB 7 — TUTORIAL
# ─────────────────────────────────────────────────────────────────────────────
elif selected_tab == "📚 Tutorial":
    st.title("📚 How to Use This Tool")
    ex("This tutorial walks you through the full workflow — from reading a new proposal to producing a defensible US contribution.")

    st.markdown("---")
    st.header("🗺️ The Big Picture — Your Workflow")
    ex("Every interference analysis follows the same five-step arc — this tool supports each step.")

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
            f"- {name}: {b['f_low_mhz']}–{b['f_high_mhz']} MHz ({b['allocation']}), I/N threshold {b['in_threshold_db']} dB, Safety-of-Life: {b['safety_of_life']}"
            for name, b in FAA_BANDS.items()
        ])

        system_prompt = f"""You are a senior RF spectrum policy advisor supporting the FAA and NTIA in ITU-R proceedings. 
Your role is to analyze ITU-R contributions from Working Party 5D (IMT/Mobile) and Working Party 5B (Maritime/Radiodetermination) 
and provide precise, actionable policy guidance to protect US aeronautical interests.

The following FAA frequency bands are protected and must be defended:
{faa_bands_summary}

Key regulatory instruments available:
- RR No. 4.10: No harmful interference to safety-of-life services
- RR No. 5.444: ARNS protection at 960-1215 MHz
- RR Resolution 233/236: RNSS/GNSS protection
- RR Resolution 750: IMT and safety services coexistence
- ITU-R M.1642: Methodology for IMT/ARNS compatibility assessments
- ITU-R SM.2028: Monte Carlo simulation methodology
- ICAO Annex 10: Aeronautical telecommunications standards
- RTCA standards: DO-235B (GNSS), DO-260B (ADS-B), DO-155 (Radio Altimeter)

Your analysis must be technically grounded and policy-actionable. Always be specific about which FAA bands are threatened, 
what the technical mechanism of interference is, and what concrete regulatory language the US should pursue.

{depth_instruction}

Structure your response with clear headers. Use plain language that a policy official can act on immediately.
Flag anything that requires urgent escalation to NTIA or ICAO."""

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
1. THREAT ASSESSMENT — Which FAA protected bands are at risk and how
2. SUBMITTER'S OBJECTIVE — What the submitting administration is actually trying to achieve
3. TECHNICAL CONCERNS — Specific interference mechanisms and vulnerable systems
4. RECOMMENDED US POSITION — Oppose / Support / Propose amendments (with rationale)
5. COUNTER-ARGUMENTS — Technical and regulatory arguments to raise in the Working Party
6. REGULATORY CITATIONS — Specific RR articles, Resolutions, and ITU-R Recommendations to invoke
7. COALITION STRATEGY — Which administrations/organizations to coordinate with
8. REQUIRED ANALYSIS — What technical studies the US should conduct or commission
9. URGENCY & TIMELINE — How quickly must the US respond and through what mechanism
10. DRAFT RESPONSE LANGUAGE — Key phrases/text for the US contribution or intervention"""

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
        ex("Every number in RF engineering is expressed in decibels. Master this and everything else follows.")

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
        ex("Use this to build intuition — change the inputs and watch how the dB math works.")
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
        ex("Frequency determines how a signal behaves — how far it travels, how it's absorbed, and what it can penetrate.")

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
        ex("Each band has different propagation characteristics — lower frequencies travel farther; higher frequencies carry more data but attenuate faster.")
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
        ex("EIRP is the single number that characterizes how much power a transmitter effectively radiates — it's what you put into every interference calculation.")

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
        ex("This is the effective radiated power toward the victim — the starting point of every link budget and interference calculation.")

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
        ex("PFD limits in the Radio Regulations (e.g., RR Appendix 7) are what you cite to protect ground receivers from satellite interference.")

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
        ex("Path loss is the signal power lost as it travels through space — even in a vacuum with no obstructions, power spreads out and weakens with distance.")

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
        ex("This number goes directly into your link budget as the main loss term.")

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
        ex("A link budget is a running total of every gain and loss in a signal path — it's the foundation of all RF system design and interference analysis.")

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
        ex("Adjust each parameter and watch how the received power changes — this is exactly how you assess interference in the Link Budget module.")

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
        ex("The noise floor sets the fundamental limit of what a receiver can detect — interference must stay well below it to be acceptable.")

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
        ex("Every extra MHz of bandwidth raises the noise floor by 10·log(extra_BW) dB — wider receivers are inherently less sensitive.")

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
        ex("I/N is the ITU-R standard currency for interference — master it and you can read, write, and challenge any interference analysis.")

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
        ex("These are the numbers you cite in contributions — they come from RTCA standards and ITU-R Recommendations.")
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
        ex("The propagation model determines how much signal reaches the victim — the biggest single source of variation between optimistic and conservative interference studies.")

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
        ex("Choosing the wrong model is the most common technical error in interference studies — reviewers will catch it.")
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
        ex("Single-scenario analysis tells you about one interferer — Monte Carlo tells you about a realistic population of interferers, which is what ITU-R protection is actually about.")

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
        ex("This live simulation shows you exactly what the Monte Carlo module does under the hood.")

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
        ex("This final lesson shows you how to translate your technical findings into policy language that influences WRC outcomes.")

        st.subheader("The Translation Problem")
        st.markdown("""
RF engineers speak in dB, dBm, I/N, path loss, and propagation models.
Policy officials speak in regulatory text, agenda items, resolutions, and geopolitical interests.

**Your job** is to be fluent in both. A technically perfect analysis that can't be 
communicated as policy language will not change the outcome of a Working Party session.
        """)

        st.subheader("Mapping RF Results to Policy Language")
        ex("This table is your translation guide — use it when writing the policy sections of a US contribution.")
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
        ex("These are the signs that a contribution is using overly optimistic assumptions to reach a compatibility conclusion — your job is to find them.")
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
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.caption(
    "Tool uses itur library (P.676, P.618) for ITU-R propagation models. "
    "P.452 and P.528 implementations are simplified analytical approximations. "
    "For full regulatory submissions, validate with ITU-R SoftTools and SEAMCAT."
)
