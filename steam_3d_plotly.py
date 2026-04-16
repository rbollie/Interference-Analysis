"""
STEAM 3D Science Lab — Teacher Pehpeh by IBT
Chemistry content sourced the same way avhol.com curates:
  Sketchfab (Chemical Bonding collection, VSEPR models, orbitals, crystals)
  3Dmol.org + PubChem (18 real molecular structures)
  PhET Simulations — Univ. Colorado Boulder (CC-BY 4.0)
  RCSB Protein Data Bank (crystallographic structures)
  GeoGebra (3D maths simulations)
  Ohio State Univ. (physics animations)
"""

import streamlit as st
import streamlit.components.v1 as _c


# ── embed helpers ─────────────────────────────────────────────────────────────

def _embed(url, height=650, note=""):
    html = (
        "<!DOCTYPE html><html><head>"
        "<style>*{margin:0;padding:0;box-sizing:border-box;}"
        "body{background:#000;overflow:hidden;}"
        f"iframe{{width:100%;height:{height}px;border:0;}}"
        "</style></head><body>"
        f'<iframe src="{url}" allowfullscreen '
        'allow="fullscreen;accelerometer;autoplay;clipboard-write;'
        'encrypted-media;gyroscope;pointer-lock;picture-in-picture;web-share">'
        "</iframe></body></html>"
    )
    _c.html(html, height=height + 4, scrolling=False)
    if note:
        st.caption(note)


def _sketchfab(model_id, height=640, note=""):
    url = (f"https://sketchfab.com/models/{model_id}/embed?"
           "autostart=1&transparent=0&ui_controls=1&ui_infos=0"
           "&ui_watermark=0&preload=1&ui_ar=0&ui_help=0&ui_vr=0")
    _embed(url, height, note)


def _geogebra(material_id, height=600, note="", mode="m"):
    url = f"https://www.geogebra.org/3d/{material_id}" if mode == "3d" \
          else f"https://www.geogebra.org/m/{material_id}"
    _embed(url, height, note)


def _card(title, body, color="#58AEFF"):
    st.markdown(
        f"<div style='background:rgba(0,0,0,.45);border-left:3px solid {color};"
        "padding:9px 14px;border-radius:0 8px 8px 0;margin:5px 0;'>"
        f"<b style='color:{color};font-size:.88rem;'>{title}</b>"
        f"<div style='color:#8899AA;font-size:.81rem;margin-top:3px;line-height:1.6;'>{body}</div>"
        "</div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════
# VERIFIED SKETCHFAB IDs — Chemical Bonding collection (venkatesh3)
# https://sketchfab.com/venkatesh3/collections/chemical-bonding-...
# ═══════════════════════════════════════════════════════════════════════
#
# ATOMIC ORBITALS (Wieslaw_Kruczala — free download licence)
_ORB1   = "898498896d2b4db58f0e08b87560dfaf"   # Orbitals part1
_ORB3   = "b8b73cc49d1f414684f893ef52d0a56b"   # Orbitals part3
_ORB4   = "5ea70afa38da4fd58591c11cfd040aa4"   # Orbitals part4
_ORB5   = "a4a95b03dc454d889f2d941b53c01f0a"   # Orbitals part5 (animated)
#
# SIGMA / PI BONDING (Michael.Aristov — free)
_SIGMA_BOND = "c51a1a63f92d448884ae178f8ddc7e42"  # H2 sigma bond formation (anim)
_ANTI_BOND  = "e34759cd30d1451ba2460d01b9ddc5f5"  # H2 sigma* antibonding (anim)
_SIGMA_PI   = "6d003a4c635149e79ffb559ef3c59359"  # Sigma vs Pi AO/MO comparison
#
# COVALENT BOND (naveen — animated)
_COV_H2O    = "107d25907a9842c9a843e96de5ed947e"  # Covalent Bond H2O (animated)
#
# MOLECULAR ORBITAL THEORY (arloopa — free)
_MOT        = "37ebc8a06b0d4dcc874c51fe40d5ad8c"  # Molecular Orbital Theory
#
# CRYSTAL STRUCTURES (arloopa — animated)
_NACL_ANIM  = "fbadfd9b0230437ea9343277aab0a210"  # Sodium Chloride (animated)
#
# PERIODIC TABLE 3D (MineralogyPetrographyMuseum)
_PTABLE     = "2cf5f88d759c4440a374ac6fe1baec23"  # Periodic Table Atomic-Ionic Radii
#
# VSEPR GEOMETRY MODELS (orgoly — all free download)
_VSEPR = {
    "Linear (180°)":                "df828649ca1e466e94a5dbafc826959d",
    "Trigonal Bipyramidal (90°/120°)": "8a7db400e7d94d8e9f20c9b5ccc7d0d2",
    "Octahedral (90°)":             "c83ffdcfd7dc48ef95d5ff16fb52cd6b",
    "Square Pyramidal":             "313d418a9f724f228d90a73934017d1f",
    "Square Planar (90°)":          "4ab48b720b5a44149ef6c51162e837fe",
    "T-Shaped":                     "2cf47d25d4a5472aa7137c6a0e009edb",
    "Linear (Octahedral)":          "4cd8eb8edbf543b9bc81da97768ef44e",
}
#
# BIOLOGY — Verified via avhol.com
_HEART_EXT = "a3f0ea2030214a6bbaa97e7357eebd58"   # External anatomy — Univ. of Dundee (CC-BY-NC-SA, English)
_HEART_INT = "26adbbe9c3d34cb698b7f75d7bfb76a6"   # Interior view   — Royal Bolton NHS Hospital (English)
_HEART_BEAT = "d9845afb1ee64ad094adc96320c67d98"   # Beating Heart — avhol.com original (animated, German labels)
_GI     = "26d08389de354277be032be39af5aba4"   # GI Tract (Univ. Groningen)
_EYE    = "307f381d58f34aa3b2772952e248e973"   # Anatomy of Vision
#
# PHYSICS — Verified via avhol.com
_GALV   = "445413d65bd34d3f8b94f5a3b4681ed7"   # Moving Coil Galvanometer
#
# ── NEW ADDITIONS ────────────────────────────────────────────────────────────
#
# BIOLOGY — University of Dundee CAHID (337 models, CC-BY-SA or CC-BY-NC-SA)
#   https://sketchfab.com/anatomy_dundee
_NERVOUS_SYS  = "2e6be1399756494b9f185ce8c5900911"  # Nervous System (CNS + spinal cord)
_CRANIAL_N    = "82d87cb89d6c48f0984a59c4f2a4cf9a"  # 12 Cranial Nerves (annotated)
_SKULL        = "dfc752c0deb34428a3355d9e9a696bff"  # Human Skull
_ABDOMEN      = "959d1974643846418d393da507f1ec20"  # Abdomen & Stomach
_HIP_JOINT    = "da6ecb7f259f45a2baa3e92bbd196ce9"  # Hip Joint — Muscle Origins & Insertions
_TCELL_CANCER = "bdca673b6bf94bfea7d36bc9a5af9707"  # T-cell attacking cancer cell (animated)
_DUNDEE_INT_H = "9f48eaa481cc4a43baeb9e1f03882cff"  # Internal Heart Anatomy (Dundee, English)
_DUNDEE_EXT_H = "10472481071e4375b8233289c277d411"  # External Heart Anatomy (Dundee, English)
#
# CELL BIOLOGY — CC-BY licensed, free embed
_PLANT_CELL   = "e61e7bdf8c8449a583b364f05e70289b"  # Plant Cell Organelles (CC-BY, cvallance01)
_ANIMAL_CELL  = "60ef7d2515b0403986ff9e8b7f234a66"  # Animal/Human Cell (CC-BY, markdragan)
#
# EARTH SCIENCE
_EARTH_LAYERS = "6654eb6fac054d3e87363e56366a5ba7"  # Earth Layers animated (arloopa)
_GEO_XSECTION = "89ab58288c994275945a02aca118724e"  # 3D Geological Cross-Section (Univ. Newcastle)
_TECTONIC_PLT = "7805590c82a54063aab2f2d691a345b8"  # Moving Tectonic Plates
#
# ENGINEERING / TECHNOLOGY — Free Sketchfab models
_ENGINE_2STR  = "6b6810215e944dc5855d91f1c03d52a8"  # Two-Stroke Engine animated (naveen)
_ENGINE_V8    = "b0dbf778b81e4afba4edf11336e2a099"  # Animated V8 Engine (meeww)
_ENGINE_4STR  = "dba271bdb4964a4e8e4dac99dcc7b0aa"  # 4-Stroke Motorcycle Engine (Univ. project)
_STEAM_TURB   = "2471ad99fcf64ec286684784a80f85fe"  # Steam Turbine annotated (CanopyCreative)
_JET_ENGINE   = "a84d4f2efbbd4e13b475eebf2b4b225d"  # Jet Turbine Engine (Fusion 360)
# ═══════════════════════════════════════════════════════════════════════


# ── PubChem CIDs for 3Dmol.org molecule viewer ────────────────────────────────
_MOL_CID = {
    "Water (H2O)":          ("962",      "Bent 104.5°. Universal solvent. H-bonds make ice less dense than liquid water."),
    "Carbon Dioxide (CO2)": ("280",      "Linear 180°. Two C=O double bonds. Greenhouse gas. Dissolves → H2CO3."),
    "Oxygen (O2)":          ("977",      "O=O double bond. Paramagnetic. 21% atmosphere. Essential for respiration."),
    "Nitrogen (N2)":        ("947",      "N≡N triple bond (945 kJ/mol). 78% atmosphere. Fixed by bacteria for plants."),
    "Ammonia (NH3)":        ("222",      "Trigonal pyramidal 107.8°. Lone pair on N. Used in fertilisers."),
    "Methane (CH4)":        ("297",      "Tetrahedral 109.5°. sp3 C. Main natural gas component. GWP 28× CO2."),
    "Ethanol":              ("702",      "OH group enables H-bonding. Bp 78.4°C. Fuel, solvent, beverage."),
    "Benzene":              ("241",      "Planar aromatic ring. Delocalised π electrons. All C-C = 1.40 Å."),
    "Glucose":              ("5793",     "Pyranose ring (chair). Primary energy source. Product of photosynthesis."),
    "Aspirin":              ("2244",     "Acetylsalicylic acid. COX inhibitor. Analgesic and anti-inflammatory."),
    "Caffeine":             ("2519",     "Methylxanthine. Blocks adenosine receptors. Lethal dose ~10 g."),
    "Penicillin G":         ("5904",     "Beta-lactam antibiotic. Fleming 1928. Inhibits cell wall synthesis."),
    "Cholesterol":          ("5997",     "Steroid lipid. Cell membrane component. Precursor to Vitamin D."),
    "ATP":                  ("5957",     "Energy currency of the cell. Hydrolysis releases 30.5 kJ/mol."),
    "Sucrose":              ("5988",     "Glucose + fructose. Non-reducing disaccharide. Hydrolysed by sucrase."),
    "Paracetamol":          ("1983",     "Acetaminophen. Analgesic/antipyretic. Hepatotoxic in overdose."),
    "Ibuprofen":            ("3672",     "NSAID. COX-1/COX-2 inhibitor. Anti-inflammatory and analgesic."),
    "Vitamin C":            ("54670067", "Ascorbic acid. Antioxidant. Collagen cofactor. Deficiency → scurvy."),
}

# ═══════════════════════════════════════════════════════════════════════
# CHEMISTRY 3D LAB
# ═══════════════════════════════════════════════════════════════════════

def render_chemistry_3d_section(api_key=None, eleven_key=None):
    st.markdown(
        "<div style='background:linear-gradient(135deg,#060f18,#0c1a10);border-radius:12px;"
        "padding:16px 22px;margin-bottom:14px;border:1px solid rgba(40,200,120,.16);'>"
        "<h2 style='color:#30E890;margin:0 0 4px;font-size:1.4rem;'>⚗️ Chemistry 3D Laboratory</h2>"
        "<p style='color:#3A7050;margin:0;font-size:.83rem;'>"
        "Curated the same way as avhol.com: Sketchfab 3D models · 3Dmol.org · PhET · RCSB PDB"
        "</p></div>",
        unsafe_allow_html=True,
    )

    lab = st.selectbox("Lab", [
        "🧪 Molecular 3D Viewer  (3Dmol.org + PubChem)",
        "🔗 Chemical Bonding — Sigma & Pi  (Sketchfab animated)",
        "📐 VSEPR Molecular Geometry  (Sketchfab 3D models)",
        "⚛️ Atomic Orbitals  (Sketchfab animated)",
        "🧊 Crystal Structures  (Sketchfab animated)",
        "🌡️ States of Matter  (PhET)",
        "⚡ Chemical Reactions & Rates  (PhET)",
        "🧬 DNA Crystal Structure  (RCSB PDB 1BNA)",
        "🔬 Protein Structures  (RCSB PDB)",
    ], key="chem_lab_select", label_visibility="collapsed")

    if "Molecular 3D Viewer" in lab:
        _mol_viewer(api_key, eleven_key)

    elif "Chemical Bonding" in lab:
        bond_model = st.radio("Model", [
            "H2 Sigma Bond Formation (animated)",
            "H2 Sigma* Antibonding Orbital (animated)",
            "Sigma vs Pi — AO and MO Comparison",
            "Covalent Bond H2O (animated)",
            "Molecular Orbital Theory Overview",
        ], key="bond_sel", label_visibility="collapsed")

        mid_map = {
            "H2 Sigma Bond Formation (animated)":      _SIGMA_BOND,
            "H2 Sigma* Antibonding Orbital (animated)": _ANTI_BOND,
            "Sigma vs Pi — AO and MO Comparison":       _SIGMA_PI,
            "Covalent Bond H2O (animated)":             _COV_H2O,
            "Molecular Orbital Theory Overview":        _MOT,
        }
        _sketchfab(mid_map[bond_model], 640,
                   "Sketchfab 3D model (Chemical Bonding collection). "
                   "Drag to rotate · Scroll to zoom · Click ▶ to play animation")
        c1, c2, c3 = st.columns(3)
        with c1:
            _card("Sigma (σ) Bond",
                  "Head-on orbital overlap along the internuclear axis. "
                  "Single bonds are always σ. Strongest type of covalent bond. "
                  "Free rotation around σ bonds.", "#30E890")
        with c2:
            _card("Pi (π) Bond",
                  "Side-on p-orbital overlap above and below the axis. "
                  "Found in double bonds (1σ+1π) and triple bonds (1σ+2π). "
                  "No free rotation — causes cis/trans isomerism.", "#30E890")
        with c3:
            _card("Antibonding (σ*)",
                  "Out-of-phase combination: node between nuclei. "
                  "Higher energy than atomic orbitals. "
                  "Electrons here weaken/break the bond.", "#30E890")
        teacher_pehpeh_panel(f"Chemical Bonding: {bond_model}", api_key, eleven_key)

    elif "VSEPR" in lab:
        geom = st.selectbox("Geometry", list(_VSEPR.keys()),
                             key="vsepr_sel", label_visibility="collapsed")
        _sketchfab(_VSEPR[geom], 640,
                   f"VSEPR Geometry: {geom} — Sketchfab 3D model by orgoly. "
                   "Drag to rotate and inspect bond angles")
        vsepr_info = {
            "Linear (180°)": ("BeCl2, CO2, HCN, C2H2", "2 bonding, 0 lone pairs. sp hybridised."),
            "Trigonal Bipyramidal (90°/120°)": ("PCl5, PF5, AsF5", "5 bonding, 0 lone pairs. sp3d. Axial 90°, equatorial 120°."),
            "Octahedral (90°)": ("SF6, PF6-, [Co(NH3)6]3+", "6 bonding, 0 lone pairs. sp3d2. All bond angles 90°."),
            "Square Pyramidal": ("BrF5, ClF5, IF5", "5 bonding, 1 lone pair. Lone pair compresses axial angle to ~84.8°."),
            "Square Planar (90°)": ("XeF4, [PtCl4]2-, [Ni(CN)4]2-", "4 bonding, 2 lone pairs. sp3d2. Common in d8 transition metals."),
            "T-Shaped": ("ClF3, BrF3", "3 bonding, 2 lone pairs. Two lone pairs in equatorial positions."),
            "Linear (Octahedral)": ("XeF2, I3-", "2 bonding, 3 lone pairs. Three lone pairs in equatorial plane."),
        }
        eg, desc = vsepr_info.get(geom, ("—", "—"))
        c1, c2 = st.columns(2)
        with c1:
            _card("Examples", eg, "#40D880")
        with c2:
            _card("Description", desc, "#40D880")
        st.info("VSEPR Rule: electron pairs repel each other and arrange to minimise repulsion. "
                "Lone pairs repel MORE than bonding pairs, compressing bond angles.")
        teacher_pehpeh_panel(f"VSEPR Molecular Geometry: {geom}", api_key, eleven_key)

    elif "Atomic Orbitals" in lab:
        orb_part = st.radio("Orbital Set",
                            ["Part 1 — s & p orbitals",
                             "Part 3 — d orbitals",
                             "Part 4 — f orbitals",
                             "Part 5 — Animated orbital filling"],
                            key="orb_sel", label_visibility="collapsed")
        mid_map2 = {
            "Part 1 — s & p orbitals":     _ORB1,
            "Part 3 — d orbitals":         _ORB3,
            "Part 4 — f orbitals":         _ORB4,
            "Part 5 — Animated orbital filling": _ORB5,
        }
        _sketchfab(mid_map2[orb_part], 640,
                   "Atomic Orbitals by Wieslaw Kruczala — Sketchfab free model. "
                   "Drag to rotate and view orbital lobes from any angle")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            _card("s orbital", "Spherical. 1 per sublevel. Nodes: n-1 radial. Max 2 electrons.", "#CC88FF")
        with c2:
            _card("p orbital", "Dumbbell. 3 per sublevel (px, py, pz). Node at nucleus. Max 6 electrons.", "#CC88FF")
        with c3:
            _card("d orbital", "5 shapes (dxy, dxz, dyz, dx2-y2, dz2). Max 10 electrons. Transition metals.", "#CC88FF")
        with c4:
            _card("f orbital", "7 complex shapes. Max 14 electrons. Lanthanides & actinides.", "#CC88FF")
        st.markdown("**Phase:** purple = +ψ · red = −ψ · **Node** = plane where ψ = 0 (zero electron probability)")
        teacher_pehpeh_panel(f"Atomic Orbitals — {orb_part}", api_key, eleven_key)

    elif "Crystal Structures" in lab:
        cryst = st.radio("Structure", [
            "NaCl Ionic Crystal (animated)",
            "Periodic Table — Atomic & Ionic Radii 3D",
        ], key="cryst_sel", label_visibility="collapsed")
        if "NaCl" in cryst:
            _sketchfab(_NACL_ANIM, 640,
                       "Sodium Chloride crystal by arloopa — Sketchfab (animated). "
                       "Click ▶ for animation · Drag to rotate")
            c1, c2, c3 = st.columns(3)
            with c1:
                _card("Rock Salt Structure", "FCC lattice. Coordination number 6:6. Each Na+ surrounded by 6 Cl- and vice versa.", "#FFCC44")
            with c2:
                _card("Ionic Bond", "Na loses e- (IE = 496 kJ/mol). Cl gains e- (EA = -349 kJ/mol). Lattice energy = 787 kJ/mol.", "#FFCC44")
            with c3:
                _card("Properties", "Mp 801°C. Brittle — layers shift → like charges repel → cleaves. Conducts when molten or dissolved.", "#FFCC44")
            teacher_pehpeh_panel("Crystal Structure: NaCl Ionic Crystal", api_key, eleven_key)
        else:
            _sketchfab(_PTABLE, 640,
                       "Periodic Table — Atomic & Ionic Radii by MineralogyPetrographyMuseum — Sketchfab. "
                       "Drag to rotate · Height shows relative atomic radius")
            c1, c2 = st.columns(2)
            with c1:
                _card("Atomic Radius Trends",
                      "Decreases across a period (↑ nuclear charge, same shielding). "
                      "Increases down a group (extra electron shell added).", "#44AAFF")
            with c2:
                _card("Ionic Radius",
                      "Cations smaller than parent atom (lost e- → more nuclear pull). "
                      "Anions larger (gained e- → more repulsion, less nuclear pull per e-).", "#44AAFF")
            teacher_pehpeh_panel("Periodic Table — Atomic and Ionic Radii 3D", api_key, eleven_key)

    elif "States of Matter" in lab:
        _embed("https://phet.colorado.edu/sims/html/states-of-matter/latest/states-of-matter_en.html",
               680, "PhET States of Matter — University of Colorado Boulder (CC-BY 4.0)")
        c1, c2, c3 = st.columns(3)
        with c1:
            _card("Solid", "Particles vibrate in fixed lattice positions. Low KE. Regular structure.", "#88CCFF")
        with c2:
            _card("Liquid", "Particles mobile but close together. IMFs intact. No fixed shape.", "#4488FF")
        with c3:
            _card("Gas", "Particles far apart, random fast motion. Nearly no IMFs. High KE.", "#2255CC")
        teacher_pehpeh_panel("States of Matter — Solid, Liquid, Gas", api_key, eleven_key)

    elif "Chemical Reactions" in lab:
        rxn_tab = st.radio("Simulation", [
            "Reactions & Rates (collision theory)",
            "Build-a-Molecule",
            "Molecular Shapes (VSEPR theory)",
            "Atomic Interactions (IMFs)",
        ], horizontal=True, key="rxn_sub")
        urls = {
            "Reactions & Rates (collision theory)":
                "https://phet.colorado.edu/sims/html/reactions-and-rates/latest/reactions-and-rates_en.html",
            "Build-a-Molecule":
                "https://phet.colorado.edu/sims/html/build-a-molecule/latest/build-a-molecule_en.html",
            "Molecular Shapes (VSEPR theory)":
                "https://phet.colorado.edu/sims/html/molecule-shapes/latest/molecule-shapes_en.html",
            "Atomic Interactions (IMFs)":
                "https://phet.colorado.edu/sims/html/atomic-interactions/latest/atomic-interactions_en.html",
        }
        _embed(urls[rxn_tab], 670,
               "PhET Interactive Simulation — University of Colorado Boulder (CC-BY 4.0)")
        if "Reactions" in rxn_tab:
            _card("Arrhenius Equation",
                  "Rate = A·e^(-Ea/RT). Reactions need: (1) enough energy ≥ Ea and (2) correct orientation. "
                  "Temperature doubles rate every ~10°C. Catalyst lowers Ea without being consumed.", "#FF8844")
            teacher_pehpeh_panel(f"Chemical Reactions: {rxn_tab}", api_key, eleven_key)
        elif "Build" in rxn_tab:
            _card("Goal", "Combine atoms to form molecules. Watch the 3D shape update live. Match the target molecule.", "#30E890")
            teacher_pehpeh_panel(f"Chemical Reactions: {rxn_tab}", api_key, eleven_key)
        elif "Shapes" in rxn_tab:
            _card("VSEPR Core Rule", "Electron pairs repel each other and adopt geometry that minimises repulsion. "
                  "Lone pairs repel more than bonding pairs.", "#30E890")
            teacher_pehpeh_panel(f"Chemical Reactions: {rxn_tab}", api_key, eleven_key)
        else:
            _card("Intermolecular Forces",
                  "London dispersion (all molecules) < Dipole-dipole (polar) < H-bond (N/O/F—H). "
                  "Boiling point rises with stronger IMFs. Noble gases: London dispersion only.", "#58AEFF")
            teacher_pehpeh_panel(f"Chemical Reactions: {rxn_tab}", api_key, eleven_key)

    elif "DNA" in lab:
        _embed("https://www.rcsb.org/3d-view/1BNA?preset=default", 680,
               "RCSB PDB 1BNA — B-DNA dodecamer. Drew & Dickerson 1981. Real X-ray crystal structure at 1.9 Å resolution.")
        c1, c2, c3 = st.columns(3)
        with c1:
            _card("B-DNA", "Right-handed helix. 10 bp/turn. Major groove 12 Å (protein binding sites). Minor groove 6 Å.", "#40D880")
        with c2:
            _card("Base Pairing", "A=T (2 H-bonds). G≡C (3 H-bonds). Antiparallel strands (5'→3' and 3'→5').", "#40D880")
        with c3:
            _card("Discovery", "Watson & Crick 1953 used Franklin's Photo 51 X-ray data. Nobel Prize 1962.", "#40D880")
        teacher_pehpeh_panel("DNA Crystal Structure — RCSB PDB 1BNA", api_key, eleven_key)

    elif "Protein" in lab:
        protein = st.selectbox("Protein", [
            "1HHO — Oxyhaemoglobin (O2 transport)",
            "1LYZ — Lysozyme (antibacterial enzyme)",
            "3GOU — Insulin hexamer",
            "1GZX — Photosystem I (photosynthesis)",
        ], key="prot_sel2", label_visibility="collapsed")
        pdb = protein.split(" — ")[0]
        _embed(f"https://www.rcsb.org/3d-view/{pdb}?preset=default", 660,
               f"RCSB PDB {pdb}. Toolbar: Cartoon / Ball & Stick / Surface / Sequence")
        _card("Reading the structure",
              "Cartoon: α-helices (red ribbons) · β-sheets (yellow arrows) · loops (grey). "
              "Rainbow = N-terminus (blue) → C-terminus (red). "
              "Surface view: hydrophobic core and active-site pockets visible.", "#AADDFF")
        teacher_pehpeh_panel(f"Protein Structure — RCSB PDB {pdb}", api_key, eleven_key)


def _mol_viewer(api_key=None, eleven_key=None):
    c1, c2 = st.columns([3, 1])

    with c2:
        mol = st.selectbox("Molecule", list(_MOL_CID.keys()),
                           key="mol3d_sel", label_visibility="collapsed")
        style = st.radio("Style", ["Ball & Stick", "Space-Fill", "Sticks", "Surface"],
                         key="mol3d_style", label_visibility="collapsed")
        bg = st.radio("Background", ["Black", "White"],
                      key="mol3d_bg", label_visibility="collapsed")
        cid, fact = _MOL_CID[mol]
        bgcolor = "#000000" if bg == "Black" else "#ffffff"
        txtcol  = "#406050" if bg == "Black" else "#204030"

        st.markdown(
            f"<div style='background:#060f0a;border:1px solid #1a5030;border-radius:8px;"
            "padding:10px 12px;margin-top:4px;'>"
            f"<b style='color:#30E890;font-size:.9rem;'>{mol}</b><br>"
            f"<span style='color:#2a5040;font-size:.73rem;'>PubChem CID {cid}</span>"
            f"<div style='color:{txtcol};font-size:.78rem;line-height:1.65;margin-top:6px;'>{fact}</div>"
            "<div style='color:#1a3020;font-size:.72rem;margin-top:8px;line-height:1.9;'>"
            "Left-drag → rotate<br>Scroll → zoom<br>Right-drag → pan<br>"
            "CPK: &#9899;C &#9898;H &#128308;O &#128309;N &#129001;S &#128992;P"
            "</div></div>",
            unsafe_allow_html=True,
        )

    # ── Style JS for 3Dmol.js ────────────────────────────────────────────
    style_js = {
        "Ball & Stick": "viewer.setStyle({}, {sphere:{scale:0.30,colorscheme:'Jmol'}, stick:{radius:0.14,colorscheme:'Jmol'}});",
        "Space-Fill":   "viewer.setStyle({}, {sphere:{colorscheme:'Jmol'}});",
        "Sticks":       "viewer.setStyle({}, {stick:{radius:0.16,colorscheme:'Jmol'}});",
        "Surface":      "viewer.setStyle({}, {sphere:{scale:0.28,colorscheme:'Jmol'}}); viewer.addSurface($3Dmol.SurfaceType.VWS,{opacity:0.65,colorscheme:'whiteCarbon'});",
    }[style]

    # ── Self-contained HTML — 3Dmol.js from cdnjs, molecule from PubChem ─
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.4/jquery.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.4/3Dmol-min.js"></script>
<style>
  *{{margin:0;padding:0;box-sizing:border-box;}}
  body{{background:{bgcolor};overflow:hidden;}}
  #gldiv{{width:100%;height:560px;position:relative;}}
  #msg{{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
        color:#30E890;font:14px/1.6 sans-serif;text-align:center;}}
</style>
</head>
<body>
<div id="gldiv"><div id="msg">Loading {mol}&#8230;</div></div>
<script>
(function(){{
  var viewer = $3Dmol.createViewer('gldiv',{{backgroundColor:'{bgcolor}'}});
  var url = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/SDF?record_type=3d';
  fetch(url)
    .then(function(r){{
      if(!r.ok) throw new Error('HTTP '+r.status);
      return r.text();
    }})
    .then(function(data){{
      document.getElementById('msg').style.display='none';
      viewer.addModel(data,'sdf');
      {style_js}
      viewer.zoomTo();
      viewer.render();
    }})
    .catch(function(e){{
      document.getElementById('msg').innerHTML=
        '<b style="color:#FF6644">Could not load molecule.</b><br>'
        +'<small style="color:#888">PubChem may be unreachable.<br>Try another molecule or refresh.</small>';
    }});
}})();
</script>
</body></html>"""

    with c1:
        _c.html(html, height=564, scrolling=False)
        st.caption(f"3Dmol.js (cdnjs) + PubChem CID {cid} · Real quantum-chemistry 3D conformer · "
                   "Left-drag rotate · Scroll zoom · Right-drag pan")

    teacher_pehpeh_panel(f"Molecular structure: {mol}", api_key, eleven_key)


# ═══════════════════════════════════════════════════════════════════════
# STEAM 3D MODELS LAB
# ═══════════════════════════════════════════════════════════════════════

def render_steam_3d_section(api_key=None, eleven_key=None):
    st.markdown(
        "<div style='background:linear-gradient(135deg,#06101e,#0e1e35);border-radius:12px;"
        "padding:16px 22px;margin-bottom:14px;border:1px solid rgba(80,160,255,.16);'>"
        "<h2 style='color:#58AEFF;margin:0 0 4px;font-size:1.4rem;'>🧊 STEAM 3D Modeling Lab</h2>"
        "<p style='color:#3A6A9A;margin:0;font-size:.83rem;'>"
        "Curated from avhol.com · Sketchfab · GeoGebra · PhET · RCSB PDB · Ohio State"
        "</p></div>",
        unsafe_allow_html=True,
    )

    model = st.selectbox("Model", [
        # ── Original models ───────────────────────────────────────────
        "🫀 Human Heart — Beating / External / Internal (avhol.com + Univ. of Dundee)",
        "🫃 Digestive System — GI Tract (Sketchfab · Univ. Groningen via avhol.com)",
        "👁️ Anatomy of Vision — Eye & Brain (Sketchfab via avhol.com)",
        "⚡ Moving Coil Galvanometer (Sketchfab via avhol.com)",
        "🌌 Solar System & Gravity — PhET",
        "🌊 Wave Interference — PhET",
        "☢️ Nuclear Fission — PhET",
        "⚙️ Forces & Motion — PhET",
        "🔋 Circuit Builder — PhET",
        "📐 Conic Sections 3D — GeoGebra",
        "➡️ Displacement Vector 3D — GeoGebra",
        "🔢 Vector in 3D — GeoGebra",
        "🔵 Sphere & Plane Intersection — GeoGebra",
        "🧬 DNA Double Helix — RCSB PDB 1BNA",
        "🦠 Protein Structure — RCSB PDB",
        # ── NEW: Extended Anatomy (Univ. of Dundee CAHID) ─────────────
        "🧠 Nervous System — Brain, Spinal Cord & Nerves (Univ. Dundee)",
        "🦷 12 Cranial Nerves — Annotated Brain (Univ. Dundee)",
        "💀 Human Skull (Univ. Dundee)",
        "🦴 Hip Joint — Muscle Origins & Insertions (Univ. Dundee)",
        "🔬 T-Cell vs Cancer Cell — Immune Response (Univ. Dundee animated)",
        "🫁 Abdomen & Stomach (Univ. Dundee)",
        # ── NEW: Cell Biology ─────────────────────────────────────────
        "🌿 Plant Cell — All Organelles (CC-BY)",
        "🐾 Animal / Human Cell — All Organelles (CC-BY)",
        # ── NEW: Earth Science ────────────────────────────────────────
        "🌍 Earth Layers — Crust, Mantle, Core (animated)",
        "🪨 Geological Cross-Section 3D (Univ. Newcastle)",
        "🌋 Moving Tectonic Plates (animated)",
        # ── NEW: Engineering / Technology ─────────────────────────────
        "🔧 Two-Stroke Engine — Animated Cutaway",
        "🚗 V8 Engine — Fully Animated",
        "🏍️ 4-Stroke Motorcycle Engine (Univ. Engineering Project)",
        "♨️ Steam Turbine — Annotated Components",
        "✈️ Jet Turbine Engine — Detailed (Fusion 360)",
    ], key="model3d_select", label_visibility="collapsed")

    if "Human Heart" in model:
        heart_view = st.radio("View", [
            "Beating Heart (avhol.com — animated)",
            "External Anatomy (Univ. of Dundee — English)",
            "Interior / Chambers (Royal Bolton NHS — English)",
        ], horizontal=True, key="heart_view")

        if "Beating" in heart_view:
            _sketchfab(_HEART_BEAT, 650,
                       "Beating Heart — Sketchfab (verified via avhol.com/animation/anatomy-of-human-heart/). "
                       "Drag to rotate · Click ▶ to animate. Note: annotations are in German.")
        elif "External" in heart_view:
            _sketchfab(_HEART_EXT, 650,
                       "External Heart Anatomy — University of Dundee CAHID (CC-BY-NC-SA, English). "
                       "Drag to rotate · Click annotations for labels.")
        else:
            _sketchfab(_HEART_INT, 650,
                       "Interior Heart — Royal Bolton NHS Hospital Medical Illustration Dept (English). "
                       "Drag to rotate · inspect chambers, valves, and major vessels.")
        c1, c2, c3 = st.columns(3)
        with c1: _card("Chambers", "4 chambers: RA, RV, LA, LV. Left ventricle wall thickest — pumps to systemic circulation.", "#FF6060")
        with c2: _card("Valves", "Tricuspid, pulmonary, mitral (bicuspid), aortic. Prevent backflow. Sounds 'lub' (AV close) 'dub' (semilunar close).", "#FF6060")
        with c3: _card("Output", "~70 mL/beat × 70 bpm = 4.9 L/min at rest. Up to 25 L/min during exercise (Frank-Starling law).", "#FF6060")
        teacher_pehpeh_panel("Human Heart — Beating 3D Model", api_key, eleven_key)

    elif "Digestive" in model:
        _sketchfab(_GI, 650,
                   "GI Tract — University of Groningen via Sketchfab (verified via avhol.com/animation/gastrointestinal-tract/)")
        c1, c2, c3 = st.columns(3)
        with c1: _card("Small Intestine", "6-7 m. Villi + microvilli: surface area ~200 m2. Absorbs amino acids, fatty acids, glucose into blood.", "#FFA040")
        with c2: _card("Large Intestine", "1.5 m. Absorbs water/electrolytes. Houses ~100 trillion bacteria (microbiome). pH 5.5-7.", "#FFA040")
        with c3: _card("Liver", "500+ functions. Bile production, glycogen storage, detoxification, urea synthesis, clotting factors.", "#FFA040")
        teacher_pehpeh_panel("Digestive System — GI Tract 3D Model", api_key, eleven_key)

    elif "Vision" in model:
        _sketchfab(_EYE, 650,
                   "Anatomy of Vision — Sketchfab (verified via avhol.com/animation/anatomy-of-vision-for-human-eye/)")
        c1, c2, c3 = st.columns(3)
        with c1: _card("Eye layers", "Sclera → Choroid → Retina. Cornea refracts ~70% of incident light. Lens adjusts focus (accommodation).", "#88CCFF")
        with c2: _card("Retina", "~6M cones (colour, fovea, photopic) + ~120M rods (peripheral, scotopic, low light). Fovea = sharpest vision.", "#88CCFF")
        with c3: _card("Visual pathway", "Optic nerve → optic chiasm (nasal fibres cross) → LGN → primary visual cortex (V1, occipital lobe).", "#88CCFF")
        teacher_pehpeh_panel("Anatomy of Vision — Human Eye and Brain", api_key, eleven_key)

    elif "Galvanometer" in model:
        _sketchfab(_GALV, 650,
                   "Moving Coil Galvanometer — Sketchfab (verified via avhol.com/animation/moving-coil-galvanometer/). "
                   "Drag to rotate and inspect components")
        c1, c2, c3 = st.columns(3)
        with c1: _card("Principle", "Torque on current-carrying coil in B-field: τ = NBIA. Coil rotates against restoring spring until τ_spring = τ_magnetic.", "#FFCC44")
        with c2: _card("Components", "Permanent magnet · N-turn rectangular coil · soft iron core (radial field) · phosphor-bronze spring · pointer.", "#FFCC44")
        with c3: _card("Conversions", "Ammeter: low shunt R in parallel. Voltmeter: high series R. Both derived by adjusting full-scale deflection range.", "#FFCC44")
        teacher_pehpeh_panel("Moving Coil Galvanometer — 3D Model", api_key, eleven_key)

    elif "Electron Flow" in model:
        _embed("https://www.asc.ohio-state.edu/orban.14/physics1251_fall2016/conductor/conductor.html",
               650, "Electron Flow in Conductor — Ohio State University (verified via avhol.com)")
        c1, c2, c3 = st.columns(3)
        with c1: _card("Drift Velocity", "vd = I / (nAe). At 1 A in copper: vd ≈ 0.1 mm/s. Signal travels at ~c (EM wave, not electron speed).", "#44CCFF")
        with c2: _card("Electron Density", "Cu: n ≈ 8.5×10^28 /m3. One free e- per atom. Very high density → very low drift velocity needed.", "#44CCFF")
        with c3: _card("Ohm's Law", "V = IR (macroscopic). J = σE (microscopic). Resistivity Cu: ρ = 1.68×10^-8 Ω·m at 20°C.", "#44CCFF")
        teacher_pehpeh_panel("Electron Flow in a Conductor — Drift Velocity Simulation", api_key, eleven_key)

    elif "Solar System" in model:
        sim = st.radio("Sim", ["My Solar System", "Gravity & Orbits"], horizontal=True, key="solar_sim2")
        url = ("https://phet.colorado.edu/sims/html/my-solar-system/latest/my-solar-system_en.html"
               if "My" in sim else
               "https://phet.colorado.edu/sims/html/gravity-and-orbits/latest/gravity-and-orbits_en.html")
        _embed(url, 680, "PhET — University of Colorado Boulder (CC-BY 4.0)")
        c1, c2, c3 = st.columns(3)
        with c1: _card("Kepler 1st", "Orbits are ELLIPSES with the Sun at one focus. Not circles.", "#88AAFF")
        with c2: _card("Kepler 2nd", "Equal areas swept in equal times → faster near perihelion (L conserved).", "#88AAFF")
        with c3: _card("Kepler 3rd", "T² ∝ a³. Earth: 1yr/1AU. Mars: 1.88yr/1.52AU. Verify: 1.88²≈1.52³", "#88AAFF")
        teacher_pehpeh_panel(f"Solar System — {sim} (Kepler's Laws)", api_key, eleven_key)

    elif "Wave" in model:
        _embed("https://phet.colorado.edu/sims/html/wave-interference/latest/wave-interference_en.html",
               680, "PhET Wave Interference — University of Colorado Boulder (CC-BY 4.0)")
        c1, c2 = st.columns(2)
        with c1: _card("Constructive", "Δφ = 0, 2π, 4π… Amplitudes ADD. Bright fringes in double-slit.", "#88DDFF")
        with c2: _card("Destructive", "Δφ = π, 3π… Amplitudes CANCEL. Dark fringes. Noise-cancelling headphones.", "#88DDFF")
        teacher_pehpeh_panel("Wave Interference — Constructive and Destructive", api_key, eleven_key)

    elif "Nuclear" in model:
        _embed("https://phet.colorado.edu/sims/html/nuclear-fission/latest/nuclear-fission_en.html",
               680, "PhET Nuclear Fission — University of Colorado Boulder (CC-BY 4.0)")
        c1, c2 = st.columns(2)
        with c1: _card("Fission", "U-235 + n → Kr-92 + Ba-141 + 3n + ~200 MeV (via E=mc²). Mass defect drives energy release.", "#FF7744")
        with c2: _card("Chain Reaction", "Released neutrons trigger more fissions. Controlled → nuclear power. Critical mass uncontrolled → weapon.", "#FF7744")
        teacher_pehpeh_panel("Nuclear Fission and Chain Reactions", api_key, eleven_key)

    elif "Forces" in model:
        _embed("https://phet.colorado.edu/sims/html/forces-and-motion-basics/latest/forces-and-motion-basics_en.html",
               680, "PhET Forces and Motion — University of Colorado Boulder (CC-BY 4.0)")
        c1, c2 = st.columns(2)
        with c1: _card("Newton's 2nd Law", "F_net = ma. Double force → double acceleration. Double mass → half acceleration.", "#58AEFF")
        with c2: _card("Friction", "f_s ≤ μ_s N (static). f_k = μ_k N (kinetic). Dry steel on steel: μ_k ≈ 0.57. Ice on ice: μ_k ≈ 0.03.", "#58AEFF")
        teacher_pehpeh_panel("Forces and Motion — Newton's Laws", api_key, eleven_key)

    elif "Circuit" in model:
        _embed("https://phet.colorado.edu/sims/html/circuit-construction-kit-dc/latest/circuit-construction-kit-dc_en.html",
               680, "PhET Circuit Construction Kit — University of Colorado Boulder (CC-BY 4.0)")
        c1, c2, c3 = st.columns(3)
        with c1: _card("Ohm's Law", "V = IR. Power P = IV = I²R = V²/R.", "#FFAA44")
        with c2: _card("Series", "I same everywhere. V_total = ΣV_i. R_total = ΣR_i. One bulb fails → all fail.", "#FFAA44")
        with c3: _card("Parallel", "V same across each branch. I_total = ΣI_i. 1/R_total = Σ(1/R_i).", "#FFAA44")
        teacher_pehpeh_panel("Circuit Builder — Series and Parallel Circuits", api_key, eleven_key)

    elif "Conic" in model:
        _geogebra("qadhz2rr", 650,
                  "Conic Sections 3D — GeoGebra (verified via avhol.com/animation/conic-section/)")
        c1, c2, c3, c4 = st.columns(4)
        with c1: _card("Circle (e=0)", "Plane ⊥ axis. x²+y²=r². Equidistant from centre.", "#AAFFAA")
        with c2: _card("Ellipse (0<e<1)", "Oblique cut one nappe. x²/a²+y²/b²=1. Planetary orbits.", "#AAFFAA")
        with c3: _card("Parabola (e=1)", "Plane parallel to slant. y²=4ax. Satellite dishes, projectiles.", "#AAFFAA")
        with c4: _card("Hyperbola (e>1)", "Cuts both nappes. x²/a²-y²/b²=1. Sonic boom wavefront, GPS.", "#AAFFAA")
        teacher_pehpeh_panel("Conic Sections 3D — Circle, Ellipse, Parabola, Hyperbola", api_key, eleven_key)

    elif "Displacement Vector" in model:
        _geogebra("rrjwyvrh", 650,
                  "Displacement Vector 3D — GeoGebra (verified via avhol.com/animation/displacement-vector/)", mode="3d")
        c1, c2 = st.columns(2)
        with c1: _card("Displacement", "Δr = r_final − r_initial. Vector (magnitude + direction). ≠ distance (scalar path length).", "#AACCFF")
        with c2: _card("Components", "Δr = Δx·i + Δy·j + Δz·k. |Δr| = √(Δx²+Δy²+Δz²). Direction cosines: cos α = Δx/|Δr|.", "#AACCFF")
        teacher_pehpeh_panel("Displacement Vector in 3D", api_key, eleven_key)

    elif "Vector in 3D" in model:
        _geogebra("kxfufh37", 650,
                  "Vector in 3D — GeoGebra (verified via avhol.com/animation/vector-in-3d/)")
        c1, c2, c3 = st.columns(3)
        with c1: _card("Components", "A = Ax·i + Ay·j + Az·k. Projections on coordinate axes.", "#AACCFF")
        with c2: _card("Magnitude", "|A| = √(Ax²+Ay²+Az²). Always a positive scalar.", "#AACCFF")
        with c3: _card("Products", "Dot: A·B = |A||B|cosθ (scalar). Cross: A×B = |A||B|sinθ·n̂ (vector ⊥ to both).", "#AACCFF")
        teacher_pehpeh_panel("Vectors in 3D — Components, Magnitude, Dot and Cross Product", api_key, eleven_key)

    elif "Sphere & Plane" in model:
        _geogebra("rqneqdwx", 650,
                  "Sphere & Plane Intersection — GeoGebra (verified via avhol.com/animation/intersection-of-sphere-and-plane/)")
        c1, c2 = st.columns(2)
        with c1: _card("Intersection", "A plane cuts a sphere in a CIRCLE (unless tangent → point, or no intersection → empty). Drag the plane.", "#CCCCFF")
        with c2: _card("Equation", "Sphere: (x-h)²+(y-k)²+(z-l)²=r². Intersection radius ρ = √(r²-d²) where d = distance from centre to plane.", "#CCCCFF")
        teacher_pehpeh_panel("Sphere and Plane Intersection — 3D Geometry", api_key, eleven_key)

    elif "DNA" in model:
        view = st.radio("Structure", [
            "1BNA — B-DNA (classic double helix)",
            "1D66 — DNA + protein complex",
            "4GXY — RNA structure",
        ], horizontal=True, key="dna_view2")
        pdb = view.split(" — ")[0]
        _embed(f"https://www.rcsb.org/3d-view/{pdb}?preset=default", 660,
               f"RCSB PDB {pdb}. Real crystallographic structure. Toolbar: Cartoon / Ball & Stick / Surface.")
        c1, c2, c3 = st.columns(3)
        with c1: _card("B-DNA", "Right-handed. 10 bp/turn. Rise 0.34 nm/bp. Major groove 12 Å, minor groove 6 Å.", "#40D880")
        with c2: _card("Base Pairing", "A=T (2 H-bonds). G≡C (3 H-bonds). GC-rich regions have higher Tm.", "#40D880")
        with c3: _card("History", "Franklin's Photo 51 (1952) showed B-form pattern. Watson & Crick model 1953. Nobel 1962.", "#40D880")
        teacher_pehpeh_panel(f"DNA Double Helix — {view}", api_key, eleven_key)

    elif "Protein" in model:
        prot = st.selectbox("Protein", [
            "1HHO — Oxyhaemoglobin",
            "1LYZ — Lysozyme",
            "3GOU — Insulin hexamer",
            "1GZX — Photosystem I",
        ], key="prot3_sel", label_visibility="collapsed")
        pdb = prot.split(" — ")[0]
        _embed(f"https://www.rcsb.org/3d-view/{pdb}?preset=default", 660,
               f"RCSB PDB {pdb}. Use toolbar: Cartoon (secondary structure) · Surface (binding pockets) · Sequence.")
        _card("Reading protein structure",
              "Cartoon: α-helices (red ribbons) · β-sheets (yellow arrows) · loops (grey). "
              "Rainbow = N-terminus (blue) to C-terminus (red). "
              "Surface view reveals accessible surface area and active-site pockets.", "#AADDFF")
        teacher_pehpeh_panel(f"Protein Structure — RCSB PDB {pdb}", api_key, eleven_key)

    # ── NEW: Extended Anatomy — Univ. of Dundee CAHID ─────────────────
    elif "Nervous System" in model:
        _sketchfab(_NERVOUS_SYS, 660,
                   "The Nervous System — University of Dundee CAHID · CC-BY-SA · "
                   "Brain + spinal cord + spinal nerves + vertebral column. Drag to rotate.")
        c1, c2, c3 = st.columns(3)
        with c1: _card("Central NS", "Brain + spinal cord. Protected by skull and vertebral column. Controls voluntary and involuntary actions.", "#88CCFF")
        with c2: _card("Peripheral NS", "31 spinal nerve pairs + 12 cranial nerves. Sensory (afferent) and motor (efferent) pathways.", "#88CCFF")
        with c3: _card("Neuron", "Cell body + dendrites (input) + axon (output). Action potential: Na+ rushes in → K+ rushes out. Speed up to 120 m/s.", "#88CCFF")
        teacher_pehpeh_panel("Nervous System — Brain, Spinal Cord and Peripheral Nerves", api_key, eleven_key)

    elif "Cranial Nerves" in model:
        _sketchfab(_CRANIAL_N, 660,
                   "12 Cranial Nerves — University of Dundee CAHID · CC-BY-SA · "
                   "Annotated exit points, innervations and functions. Click annotations for details.")
        c1, c2 = st.columns(2)
        with c1:
            _card("CN I–VI", "I Olfactory (smell) · II Optic (vision) · III Oculomotor (eye mvmt) · "
                             "IV Trochlear (superior oblique) · V Trigeminal (face sensation/chewing) · VI Abducens (lateral eye mvmt)", "#88BBFF")
        with c2:
            _card("CN VII–XII", "VII Facial (expression, taste 2/3) · VIII Vestibulocochlear (hearing/balance) · "
                                "IX Glossopharyngeal · X Vagus (heart, gut) · XI Accessory · XII Hypoglossal (tongue mvmt)", "#88BBFF")
        st.caption("Mnemonic (S/M/B type): Some Say Marry Money But My Brother Says Big Brains Matter More")
        teacher_pehpeh_panel("12 Cranial Nerves — Anatomy and Functions", api_key, eleven_key)

    elif "Skull" in model:
        _sketchfab(_SKULL, 660,
                   "Human Skull — University of Dundee CAHID · Drag to rotate and inspect sutures, foramina and processes.")
        c1, c2, c3 = st.columns(3)
        with c1: _card("Cranium (8 bones)", "Frontal · 2 Parietal · Occipital · 2 Temporal · Sphenoid · Ethmoid. Protect brain.", "#DDD0B0")
        with c2: _card("Face (14 bones)", "Mandible (only moveable) · Maxilla · Zygomatic · Nasal · Lacrimal · Palatine · Vomer · Inferior nasal conchae.", "#DDD0B0")
        with c3: _card("Key foramina", "Foramen magnum: spinal cord exits. Optic canal: CN II. Jugular foramen: CN IX, X, XI + jugular vein.", "#DDD0B0")
        teacher_pehpeh_panel("Human Skull — Cranial and Facial Bones", api_key, eleven_key)

    elif "Hip Joint" in model:
        _sketchfab(_HIP_JOINT, 660,
                   "Hip Joint — Muscle Origins & Insertions · University of Dundee CAHID. Click annotations for details.")
        c1, c2, c3 = st.columns(3)
        with c1: _card("Ball & Socket", "Head of femur in acetabulum. ROM: flexion 120°, extension 20°, abduction 45°, adduction 30°.", "#FFCCAA")
        with c2: _card("Key muscles", "Gluteus maximus (extension). Iliopsoas (flexion). Gluteus medius (abduction). Adductor group (adduction).", "#FFCCAA")
        with c3: _card("Ligaments", "Iliofemoral: strongest in body. Pubofemoral. Ischiofemoral. Ligamentum teres carries artery to femoral head.", "#FFCCAA")
        teacher_pehpeh_panel("Hip Joint — Muscle Origins, Insertions and Ligaments", api_key, eleven_key)

    elif "T-Cell" in model:
        _sketchfab(_TCELL_CANCER, 660,
                   "T-Cell attacking Cancer Cell — University of Dundee CAHID (animated). Click play to animate.")
        c1, c2, c3 = st.columns(3)
        with c1: _card("Cytotoxic T-cell (CD8+)", "Recognises MHC-I + antigen on cancer cell. Forms immune synapse. Releases perforin and granzymes to kill.", "#FF6688")
        with c2: _card("Cancer cell death", "Perforin punches holes in membrane. Granzymes trigger caspase cascade → apoptosis (programmed cell death).", "#FF6688")
        with c3: _card("Immunotherapy", "PD-1/PD-L1 checkpoint inhibitors prevent cancer from switching T-cells off. Nobel Prize 2018 (Allison & Honjo).", "#FF6688")
        teacher_pehpeh_panel("T-Cell vs Cancer Cell — Immune Response and Immunotherapy", api_key, eleven_key)

    elif "Abdomen" in model:
        _sketchfab(_ABDOMEN, 660,
                   "Abdomen & Stomach — University of Dundee CAHID. Drag to rotate and explore layers.")
        c1, c2, c3 = st.columns(3)
        with c1: _card("Stomach", "J-shaped muscular bag. Rugae allow expansion to ~1 L. HCl (pH 1.5–2) denatures proteins. Pepsin digests protein.", "#FFAA66")
        with c2: _card("9 Abdominal regions", "Epigastric · Umbilical · Hypogastric + L/R Hypochondriac · L/R Lumbar · L/R Iliac. Used in clinical examination.", "#FFAA66")
        with c3: _card("Peritoneum", "Serous membrane lining abdominal cavity. Visceral layer covers organs. Parietal layer lines wall. Peritonitis is life-threatening.", "#FFAA66")
        teacher_pehpeh_panel("Abdomen and Stomach — Digestive Anatomy", api_key, eleven_key)

    # ── NEW: Cell Biology ──────────────────────────────────────────────
    elif "Plant Cell" in model:
        _sketchfab(_PLANT_CELL, 640,
                   "Plant Cell Organelles — CC-BY · cvallance01 · Sketchfab. Drag to rotate · zoom into organelles.")
        c1, c2, c3 = st.columns(3)
        with c1: _card("Unique to Plant Cells", "Cell wall (cellulose): rigid support. Chloroplasts: photosynthesis (chlorophyll in thylakoids). Large central vacuole: turgor pressure.", "#66BB66")
        with c2: _card("Shared with Animal Cells", "Nucleus, mitochondria, ribosomes, rough/smooth ER, Golgi apparatus, cell membrane (phospholipid bilayer).", "#66BB66")
        with c3: _card("Photosynthesis", "6CO2 + 6H2O + light → C6H12O6 + 6O2. Light reactions in thylakoid (ATP, NADPH). Calvin cycle in stroma (G3P → glucose).", "#66BB66")
        teacher_pehpeh_panel("Plant Cell — Organelles and Photosynthesis", api_key, eleven_key)

    elif "Animal" in model and "Cell" in model:
        _sketchfab(_ANIMAL_CELL, 640,
                   "Animal / Human Cell — CC-BY · markdragan · Sketchfab. Drag to rotate · explore each organelle.")
        c1, c2, c3 = st.columns(3)
        with c1: _card("Nucleus", "Control centre. Double membrane with nuclear pores. Contains 46 chromosomes (humans). Nucleolus assembles ribosomes.", "#AAAAFF")
        with c2: _card("Mitochondria", "ATP synthesis via oxidative phosphorylation. Cristae increase inner membrane surface area. Own DNA (endosymbiosis evidence).", "#AAAAFF")
        with c3: _card("Endomembrane", "Rough ER: membrane proteins. Smooth ER: lipid synthesis, detox. Golgi: sorts and packages proteins into vesicles for secretion.", "#AAAAFF")
        teacher_pehpeh_panel("Animal Cell — Organelles and Cell Biology", api_key, eleven_key)

    # ── NEW: Earth Science ─────────────────────────────────────────────
    elif "Earth Layers" in model:
        _sketchfab(_EARTH_LAYERS, 640,
                   "Earth Layers (animated) — arloopa · Sketchfab. Click play to animate · Drag to rotate.")
        c1, c2, c3, c4 = st.columns(4)
        with c1: _card("Crust (0–70 km)", "Oceanic: basalt, 5–10 km. Continental: granite, up to 70 km. Tectonic plates ride on asthenosphere.", "#CCAA66")
        with c2: _card("Mantle (70–2900 km)", "Silicate rock. Asthenosphere flows (convection drives plates). Lower mantle solid. 1300–3700°C.", "#CC8844")
        with c3: _card("Outer Core (2900–5150 km)", "Liquid iron-nickel. Convection generates Earth's magnetic field (geodynamo). 4000–5000°C.", "#CC4422")
        with c4: _card("Inner Core (5150–6371 km)", "Solid iron-nickel despite ~5500°C — extreme pressure (3.5 million atm) keeps it solid.", "#FF4400")
        teacher_pehpeh_panel("Earth Layers — Crust, Mantle, Outer Core, Inner Core", api_key, eleven_key)

    elif "Geological Cross" in model:
        _sketchfab(_GEO_XSECTION, 640,
                   "3D Geological Cross-Section — Earth Sciences, University of Newcastle. "
                   "Shows how surface map relates to subsurface rock structure.")
        c1, c2 = st.columns(2)
        with c1: _card("Reading Cross-Sections", "Vertical cut through rock layers showing dip angle, fold axes, fault planes, synclines and anticlines. Used in mining and oil exploration.", "#AA8844")
        with c2: _card("Rock Cycle", "Sedimentary (layered: limestone, sandstone). Igneous (cooled magma: granite, basalt). Metamorphic (heat/pressure: marble, quartzite).", "#AA8844")
        teacher_pehpeh_panel("Geological Cross-Section — Rock Layers and Structures", api_key, eleven_key)

    elif "Tectonic" in model:
        _sketchfab(_TECTONIC_PLT, 640,
                   "Moving Tectonic Plates — Sketchfab. Drag to rotate · Shows plate boundaries and movement.")
        c1, c2, c3 = st.columns(3)
        with c1: _card("Convergent", "Plates collide. Oceanic under continental → subduction → volcanoes (Ring of Fire). Two continental plates → mountains (Himalayas).", "#BB7744")
        with c2: _card("Divergent", "Plates pull apart → mid-ocean ridges, new seafloor. Iceland sits on Mid-Atlantic Ridge. East African Rift Valley on land.", "#BB7744")
        with c3: _card("Transform", "Plates slide past each other. No crust created or destroyed. Causes earthquakes. San Andreas Fault is a transform boundary.", "#BB7744")
        teacher_pehpeh_panel("Tectonic Plates — Plate Boundaries and Geological Events", api_key, eleven_key)

    # ── NEW: Engineering / Technology ─────────────────────────────────
    elif "Two-Stroke" in model:
        _sketchfab(_ENGINE_2STR, 640,
                   "Two-Stroke Engine — Animated · naveen · Sketchfab. Click play to animate. "
                   "Used in pehn-pehn, keh-keh, chainsaws, outboard motors.")
        c1, c2, c3 = st.columns(3)
        with c1: _card("Stroke 1 — Power", "Combustion pushes piston DOWN. Exhaust port opens → burned gases exit. Transfer port opens → fresh air-fuel mixture enters.", "#FF8844")
        with c2: _card("Stroke 2 — Compression", "Piston moves UP → compresses charge. Crankcase simultaneously draws fresh mixture below piston.", "#FF8844")
        with c3: _card("vs 4-Stroke", "More power per revolution. Lighter (no valves). But less fuel-efficient, higher emissions. Common in pehn-pehn & keh-keh engines and chainsaws.", "#FF8844")
        teacher_pehpeh_panel("Two-Stroke Engine — How It Works", api_key, eleven_key)

    elif "V8" in model:
        _sketchfab(_ENGINE_V8, 640,
                   "Animated V8 Engine — meeww · Sketchfab · Free use. Click play to animate.")
        c1, c2, c3 = st.columns(3)
        with c1: _card("4-Stroke Cycle", "Intake (piston down, fuel-air in) → Compression (piston up) → Power (spark ignites, piston down) → Exhaust (piston up, gases out).", "#FFAA44")
        with c2: _card("V8 Layout", "8 cylinders in two banks of 4 at ~90°. Staggered firing order gives smooth torque. Used in large trucks and performance cars.", "#FFAA44")
        with c3: _card("Efficiency", "Otto cycle thermal efficiency: eta = 1 - (1/r^(gamma-1)). Higher compression ratio → more efficient. Typical efficiency 25–35%.", "#FFAA44")
        teacher_pehpeh_panel("V8 Engine — Four-Stroke Combustion Cycle", api_key, eleven_key)

    elif "4-Stroke Motorcycle" in model:
        _sketchfab(_ENGINE_4STR, 640,
                   "4-Stroke Motorcycle Engine — University Engineering Project · Miguel Hernández Univ. · Free. "
                   "Drag to rotate · inspect crankshaft, pistons, valves.")
        c1, c2 = st.columns(2)
        with c1: _card("Components", "Piston + connecting rod → crankshaft (converts linear to rotary motion). Camshaft operates valves via timing chain. Spark plug ignites charge.", "#FFCC44")
        with c2: _card("Valve Timing", "Intake valve opens just before TDC. Exhaust valve closes just after TDC. Valve overlap maximises cylinder filling (volumetric efficiency).", "#FFCC44")
        teacher_pehpeh_panel("4-Stroke Motorcycle Engine — Components and Mechanism", api_key, eleven_key)

    elif "Steam Turbine" in model:
        _sketchfab(_STEAM_TURB, 640,
                   "Steam Turbine — Annotated · CanopyCreative · Sketchfab. Click annotations for component details.")
        c1, c2, c3 = st.columns(3)
        with c1: _card("Working Principle", "High-pressure steam expands through nozzles → kinetic energy. Turbine blades extract energy. Each stage drops pressure and temperature.", "#88DDFF")
        with c2: _card("Rankine Cycle", "Pump → Boiler → Turbine → Condenser. Efficiency 30–45%. Combined cycle (gas + steam) reaches ~60%.", "#88DDFF")
        with c3: _card("Applications", "Coal, nuclear and geothermal power stations. Ship propulsion. Oil refinery power generation. Nigeria's Niger Delta power plants.", "#88DDFF")
        teacher_pehpeh_panel("Steam Turbine — Rankine Cycle and Power Generation", api_key, eleven_key)

    elif "Jet Turbine" in model:
        _sketchfab(_JET_ENGINE, 640,
                   "Jet Turbine Engine — Fusion 360 · anandhakumar1002 · Sketchfab · Free. "
                   "Drag to rotate · Shows compressor, combustion chamber, turbine, nozzle.")
        c1, c2, c3 = st.columns(3)
        with c1: _card("Brayton Cycle", "Intake → Compressor (adiabatic compression) → Combustion (heat added at constant pressure) → Turbine → Nozzle (thrust generated).", "#AADDFF")
        with c2: _card("Thrust", "F = m_dot x (v_exit - v_inlet). Turbofan bypasses most air around core → quieter, more fuel-efficient at subsonic speeds.", "#AADDFF")
        with c3: _card("Turbofan vs Turbojet", "Turbofan: large fan + bypass duct → used in all commercial aircraft. Turbojet: all air through core → used in supersonic military jets.", "#AADDFF")
        teacher_pehpeh_panel("Jet Turbine Engine — Brayton Cycle and Thrust", api_key, eleven_key)


# ═══════════════════════════════════════════════════════════════════════
# TEACHER PEHPEH AI ASSISTANT
# ═══════════════════════════════════════════════════════════════════════

import json
import urllib.request
import urllib.error

_TP_SYSTEM = """You are Teacher Pehpeh, a warm, enthusiastic science teacher working in West Africa,
primarily serving students in Liberia, Sierra Leone, Ghana, Guinea, and Nigeria.
You make hard science concepts feel easy by connecting them to everyday life students actually know.

══════════════════════════════════════════════════════
LIBERIAN KOLOQUA — Use these words naturally in explanations
══════════════════════════════════════════════════════

TRANSPORT:
  pehn-pehn    = motorcycle taxi (NOT okada — that is Nigerian)
  keh-keh      = bajaj/tricycle taxi (three-wheeler)
  hottemo      = a car
  kia-moto     = a large pickup truck (used to transport goods)
  holi-holi    = a passenger bus / para-transit bus
  cycle        = bicycle
  coe tar ro   = a paved/asphalt road
  dusty road   = an unpaved dirt road
  skalo        = a free ride/lift in a vehicle
  carboy       = fare collector on a bus/truck

FOOD & DRINK:
  red oil      = palm oil
  bamboo wine  = palm wine (from raffia or oil palm)
  country chop = local stew with various meats/fish served over rice
  palava sauce = green sauce made from jute leaves (served with rice)
  dumboy       = thick cassava dough (swallowed not chewed)
  fufu         = cassava food (similar to dumboy)
  farina       = dried cassava flakes
  plum         = mango
  paw-paw      = papaya
  ground pea   = peanut / groundnut
  boney        = dried herring fish
  tinapaw      = canned mackerel in tomato sauce
  kalla        = doughnuts fried in oil
  kitili       = small bitter garden egg (eggplant)
  argo oil     = vegetable cooking oil
  chicken soup = bouillon/seasoning cube (Maggi)
  coe bo       = cheap street food / small meal
  jollof rice  = rice dish cooked in tomato sauce

EVERYDAY LIFE:
  currenn      = electricity / electric current
  dynamo       = diesel generator
  coh-pa       = coal pot / charcoal cooking stove
  cutlax       = machete
  country medicine = traditional herbal remedies
  druss        = Western / pharmaceutical medicine
  blinger      = mobile phone / cell phone
  me-sheen     = laptop or computer
  barbing saloon = barber shop
  pressing iron = clothes iron (older type uses charcoal)
  comping / susu = rotational savings club (community finance)
  rubba        = rubber (as in rubber plantation / Firestone)
  country soap = traditional handmade soap
  country rope = forest vines
  cattah       = rolled cloth used to balance loads on the head

PEOPLE & COMMUNITY:
  ba           = friend / buddy / peer
  papay        = wealthy older man / boss / elder
  pekin        = a young boy / child
  brabee       = older brother
  antay        = aunt
  elderdo      = title of respect for older man/woman
  kwi          = educated / westernised person
  book people  = intellectuals / educated class
  juju man     = traditional healer / medicine man
  zoe          = Poro or Sande ritual elder / specialist
  gronna boy   = juvenile delinquent / rough youth
  big man      = government official / important person

EXPRESSIONS:
  now-now      = right now / immediately
  jus na       = right away / just now
  small-small  = gradually / little by little
  different different = several varieties / all kinds
  correh       = correct / good quality / upstanding
  fuan-fuan    = trouble / problem / headache
  wahala       = big problem / contentious matter
  gbelleh      = foolish / stupid
  flakajay     = fake / substandard / foolish
  saka         = crazy / mentally unstable
  dux          = to ace something / top performer
  haat clean   = to have integrity / honest intentions
  big book     = educated / using big/difficult words
  know book    = to be formally educated
  gapping      = to be very hungry / suffering
  bluff        = to show off / flaunt
  humbug       = to bother / annoy
  crackay      = stubborn / argumentative person
  vex          = angry
  che!         = expression of surprise / disbelief
  ineh?        = isn't it? / right?

LIBERIA-SPECIFIC SCIENCE CONTEXTS:
  - Firestone / Bridgestone rubber plantation (Margibi / Harbel) — polymer chemistry, botany
  - Iron ore mining in Nimba County — metals, oxidation, geology
  - St. Paul River, Mesurado River, Cavalla River — water, ecosystems, flow
  - Charcoal production — combustion, carbon chemistry, incomplete vs complete burning
  - Laterite (red soil) — iron oxide chemistry, weathering, geology
  - Dense rainforest in Lofa and Nimba — photosynthesis, biodiversity, ecosystems
  - Solar panels being installed in rural Bong County — circuits, energy, photoelectric effect
  - Kalla (fried doughnuts) cooking in hot red oil — heat transfer, convection, states of matter
  - Coh-pa (charcoal stove) — combustion, thermodynamics
  - Palava sauce cooking — chemistry of cooking, denaturation
  - Cane juice (sugarcane liquor) fermentation — biochemistry, enzymes, yeast
  - Atlantic fishing on the Liberian coast — waves, buoyancy, biology
  - Dynamo/generator — electromagnetism, Faraday's law
  - Pehn-pehn engine — two-stroke combustion, forces, friction
  - Keh-keh three-wheeler — torque, balance, centre of mass
  - Currenn (electricity) in Monrovia — circuits, AC/DC, Ohm's law
  - Logging trucks on dusty roads — friction, Newton's laws
  - Waterside market in Monrovia — acids/bases (lime, tomato), mixtures
  - Susu savings club — mathematical patterns, interest, percentage
  - Country rope (forest vines) — tensile strength, material science
  - Dumboy pounding in mortar — pressure, force, mechanical advantage

══════════════════════════════════════════════════════
YOUR TEACHING STYLE
══════════════════════════════════════════════════════

- Friendly, warm, encouraging — like a favourite teacher
- WASSCE / CAPE / Senior Secondary level (SS1–SS3)
- ALWAYS give ONE local Liberian analogy FIRST, then the science
- Use Koloqua words naturally — e.g. "the pehn-pehn engine", "the coh-pa stove", "the keh-keh"
- Suggest PROJECT-BASED LEARNING using local/cheap materials:
  empty tins, palm fronds, cassava starch, river water, charcoal,
  local seeds, clay, plastic bottles, rubber strips, groundnut shells,
  empty Argo oil containers, country rope, coh-pa ash, laterite soil
- Break every concept into 2–3 simple steps
- END every response with: "💡 Try This!" — a hands-on project using local materials
- Keep responses encouraging and not too long

When given a 3D model/simulation topic:
  1. SHORT plain-language explanation of what they see (2–3 sentences)
  2. Connect it to something familiar from Liberian / West African daily life
  3. Suggest a project using local materials
  4. End with 💡 Try This!

Speak clearly and warmly. Do not use heavy slang — keep accessible to all West African students."""


def _ask_pehpeh(question: str, topic: str, api_key: str) -> str:
    """Call Claude API and return Teacher Pehpeh's response."""
    payload = json.dumps({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 600,
        "system": _TP_SYSTEM,
        "messages": [
            {
                "role": "user",
                "content": (
                    f"The student is looking at the 3D model/simulation: **{topic}**\n\n"
                    f"Student question or request: {question}"
                )
            }
        ]
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["content"][0]["text"]
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        return f"⚠️ API error {e.code}: {body[:200]}"
    except Exception as exc:
        return f"⚠️ Could not reach Teacher Pehpeh: {exc}"


# ── ElevenLabs TTS (Teacher Pehpeh voice) ────────────────────────────────────
# Warm, friendly female voice — "Rachel" (default ElevenLabs voice).
# To use a different voice, replace with any voice_id from your ElevenLabs library.
_ELEVEN_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"
_ELEVEN_MODEL_ID = "eleven_multilingual_v2"   # high-quality, handles Koloqua terms well


def _elevenlabs_tts(text: str, api_key: str):
    """Call ElevenLabs TTS and return MP3 bytes, or an error string."""
    payload = json.dumps({
        "text": text,
        "model_id": _ELEVEN_MODEL_ID,
        "voice_settings": {
            "stability": 0.45,
            "similarity_boost": 0.75,
            "style": 0.25,
            "use_speaker_boost": True,
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        f"https://api.elevenlabs.io/v1/text-to-speech/{_ELEVEN_VOICE_ID}",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
            "xi-api-key": api_key,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        return f"⚠️ ElevenLabs error {e.code}: {body[:200]}"
    except Exception as exc:
        return f"⚠️ Could not reach ElevenLabs: {exc}"


def teacher_pehpeh_panel(topic: str, api_key: str = None, eleven_key: str = None) -> None:
    """
    Drop-in panel that renders below any 3D model.
    Call with the topic label (e.g. 'Water H2O molecular structure'),
    the Claude api_key, and optionally the ElevenLabs eleven_key for spoken audio.
    """
    # session-state key unique to this topic so chats don't bleed between tabs
    safe_key = "tp_" + "".join(c if c.isalnum() else "_" for c in topic)[:48]
    hist_key  = safe_key + "_hist"
    input_key = safe_key + "_inp"

    if hist_key not in st.session_state:
        st.session_state[hist_key] = []

    st.markdown(
        "<div style='margin-top:18px;border-top:1px solid rgba(255,200,80,.12);'></div>",
        unsafe_allow_html=True,
    )

    with st.expander("🎓 Ask Teacher Pehpeh", expanded=False):
        # Pre-load browser voices so first click doesn't stutter
        _c.html(
            "<script>if(window.speechSynthesis){window.speechSynthesis.getVoices();}</script>",
            height=0,
        )
        # ── header ────────────────────────────────────────────────────
        st.markdown(
            "<div style='background:linear-gradient(135deg,#1a0e00,#2a1a00);"
            "border-radius:10px;padding:12px 16px;margin-bottom:10px;"
            "border:1px solid rgba(255,180,40,.2);'>"
            "<span style='font-size:1.1rem;'>👩🏽‍🏫</span> "
            "<b style='color:#FFD060;font-size:.95rem;'>Teacher Pehpeh</b> "
            "<span style='color:#A07030;font-size:.8rem;'>"
            "— simplifying science with local West African examples &amp; project ideas</span>"
            "</div>",
            unsafe_allow_html=True,
        )

        # ── Audio explanation of the 3D model ─────────────────────────
        audio_key  = safe_key + "_audio_bytes"
        script_key = safe_key + "_audio_script"

        col_btn, col_info = st.columns([1, 2])
        with col_btn:
            gen_audio = st.button("🎙️ Hear Teacher Pehpeh explain this",
                                  key=safe_key + "_gen_audio",
                                  use_container_width=True)
        with col_info:
            st.markdown("<p style='color:#705030;font-size:.78rem;margin:6px 0 0;'>"
                        "Short spoken overview of this 3D model — in Teacher Pehpeh's voice.</p>",
                        unsafe_allow_html=True)

        if gen_audio:
            if not api_key:
                st.warning("🔑 Claude API key is required to generate the script.")
            elif not eleven_key:
                st.warning("🔑 ElevenLabs API key is required for spoken audio. "
                           "Please add it in the app settings.")
            else:
                with st.spinner("Teacher Pehpeh is preparing her explanation..."):
                    # 1. Get a short Koloqua-style script from Claude (reuses _TP_SYSTEM)
                    script = _ask_pehpeh(
                        "Give me a short, warm spoken explanation of this 3D model "
                        "(about 90–120 words). One local Liberian analogy, then the science, "
                        "then end with a quick 'Try This!' project idea. "
                        "Plain prose only — no markdown, no bullet points, no emojis, "
                        "no section headings. Speak as if recording audio for the student.",
                        topic,
                        api_key,
                    )
                    if script.startswith("⚠️"):
                        st.error(script)
                    else:
                        # Strip any stray markdown the model might emit anyway
                        import re as _re_a
                        clean = _re_a.sub(r"[*_`#>|~]", "", script)
                        clean = _re_a.sub(r"\s+", " ", clean).strip()

                        # 2. Send to ElevenLabs
                        audio = _elevenlabs_tts(clean, eleven_key)
                        if isinstance(audio, str):       # error string
                            st.error(audio)
                        else:
                            st.session_state[audio_key]  = audio
                            st.session_state[script_key] = clean

        # Replay cached audio + show transcript
        if st.session_state.get(audio_key):
            st.audio(st.session_state[audio_key], format="audio/mp3")
            with st.expander("📄 Transcript", expanded=False):
                st.markdown(
                    f"<div style='color:#E0C080;font-size:.85rem;line-height:1.65;'>"
                    f"{st.session_state[script_key]}</div>",
                    unsafe_allow_html=True,
                )

        st.markdown(
            "<div style='margin:10px 0;border-top:1px solid rgba(255,200,80,.08);'></div>",
            unsafe_allow_html=True,
        )

        # ── quick-start buttons ───────────────────────────────────────
        if not st.session_state[hist_key]:
            st.markdown("<p style='color:#705030;font-size:.82rem;margin:0 0 6px;'>"
                        "Not sure where to start? Try one of these:</p>",
                        unsafe_allow_html=True)
            col_a, col_b, col_c = st.columns(3)
            prompts = {
                "🤔 Explain it simply": "Can you explain this to me in simple terms, like I'm hearing it for the first time?",
                "🌴 Local example":     "Give me a local West African example that relates to this topic.",
                "🔬 Project idea":      "Suggest a hands-on project I can do at home or school using local materials.",
            }
            triggered = None
            for col, (label, prompt) in zip([col_a, col_b, col_c], prompts.items()):
                with col:
                    if st.button(label, key=safe_key + label, use_container_width=True):
                        triggered = prompt
            if triggered:
                st.session_state[hist_key].append({"role": "user", "content": triggered})
                st.rerun()

        # ── chat history ──────────────────────────────────────────────
        for msg_idx, msg in enumerate(st.session_state[hist_key]):
            if msg["role"] == "user":
                st.markdown(
                    f"<div style='background:rgba(255,200,80,.08);border-radius:8px;"
                    "padding:8px 12px;margin:4px 0;text-align:right;'>"
                    f"<span style='color:#FFD080;font-size:.85rem;'>{msg['content']}</span></div>",
                    unsafe_allow_html=True,
                )
            else:
                # Clean text for speech — strip markdown symbols
                import re as _re
                speech_text = _re.sub(r"[*_`#>|~]", "", msg["content"])
                speech_text = _re.sub(r"\s+", " ", speech_text).strip()
                # Escape for JS string — replace backslash, then quotes, then newlines
                speech_safe = (speech_text
                               .replace("\\", "\\\\")
                               .replace("'", "\\'")
                               .replace('"', '\\"')
                               .replace("\n", " "))

                audio_id = f"tp_audio_{safe_key}_{msg_idx}"
                st.markdown(
                    f"<div style='background:rgba(40,25,0,.5);border:1px solid rgba(255,180,40,.15);"
                    "border-radius:8px;padding:10px 14px;margin:4px 0;'>"
                    "<span style='color:#FFCC60;font-size:.8rem;font-weight:700;'>👩🏽‍🏫 Teacher Pehpeh</span>"
                    f"<span style='float:right;'>"
                    f"<button id='{audio_id}_play' onclick=\"(function(){{"
                    f"  if(window.speechSynthesis.speaking){{window.speechSynthesis.cancel();}}  "
                    f"  var u=new SpeechSynthesisUtterance('{speech_safe}');"
                    f"  u.rate=0.92; u.pitch=1.05; u.volume=1.0;"
                    f"  var voices=window.speechSynthesis.getVoices();"
                    f"  var pick=voices.find(v=>v.lang.startsWith('en'));"
                    f"  if(pick) u.voice=pick;"
                    f"  window.speechSynthesis.speak(u);"
                    f"  document.getElementById('{audio_id}_play').style.display='none';"
                    f"  document.getElementById('{audio_id}_stop').style.display='inline';"
                    f"  u.onend=function(){{document.getElementById('{audio_id}_play').style.display='inline';"
                    f"  document.getElementById('{audio_id}_stop').style.display='none';}};"
                    f"}})()\" "
                    "style='background:rgba(255,180,40,.15);border:1px solid rgba(255,180,40,.3);"
                    "color:#FFD060;border-radius:6px;padding:2px 10px;font-size:.75rem;"
                    "cursor:pointer;margin-left:6px;'>🔊 Listen</button>"
                    f"<button id='{audio_id}_stop' onclick=\"window.speechSynthesis.cancel();"
                    f"document.getElementById('{audio_id}_play').style.display='inline';"
                    f"document.getElementById('{audio_id}_stop').style.display='none';\" "
                    "style='display:none;background:rgba(255,80,40,.15);border:1px solid rgba(255,80,40,.3);"
                    "color:#FF8060;border-radius:6px;padding:2px 10px;font-size:.75rem;"
                    "cursor:pointer;margin-left:6px;'>⏹ Stop</button>"
                    "</span><br>"
                    f"<span style='color:#E0C080;font-size:.86rem;line-height:1.65;'>{msg['content']}</span></div>",
                    unsafe_allow_html=True,
                )

        # ── reply pending ─────────────────────────────────────────────
        history = st.session_state[hist_key]
        if history and history[-1]["role"] == "user":
            if not api_key:
                reply = ("🔑 Teacher Pehpeh needs an API key to answer. "
                         "Please add your Anthropic API key in the app settings.")
            else:
                with st.spinner("Teacher Pehpeh is thinking..."):
                    reply = _ask_pehpeh(history[-1]["content"], topic, api_key)
            st.session_state[hist_key].append({"role": "assistant", "content": reply})
            st.rerun()

        # ── input box ─────────────────────────────────────────────────
        user_input = st.chat_input(
            "Ask Teacher Pehpeh anything about this topic...",
            key=input_key,
        )
        if user_input and user_input.strip():
            st.session_state[hist_key].append({"role": "user", "content": user_input.strip()})
            st.rerun()

        # ── clear chat ────────────────────────────────────────────────
        if st.session_state[hist_key]:
            if st.button("🗑️ Clear chat", key=safe_key + "_clear",
                         use_container_width=False):
                st.session_state[hist_key] = []
                st.rerun()
