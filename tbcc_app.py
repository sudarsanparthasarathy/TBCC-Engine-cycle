"""
TBCC Engine Lab  -  interactive Turbine-Based Combined Cycle explorer
Run with:   streamlit run tbcc_app.py
"""
from __future__ import annotations

from dataclasses import astuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

import tbcc_model as tm

st.set_page_config(page_title="TBCC Engine Lab", page_icon="🛩️", layout="wide")

# --------------------------------------------------------------------------- #
# Theme
# --------------------------------------------------------------------------- #
C_TJ, C_TR, C_RJ = "#38bdf8", "#fbbf24", "#fb7185"
INK, MUTED = "#e2e8f0", "#94a3b8"

st.markdown(
    """
<style>
.stApp {background: radial-gradient(1200px 600px at 15% -10%, #12274a 0%, #0b1220 45%, #070b14 100%);}
header[data-testid="stHeader"] {background: transparent;}
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 1.4rem; max-width: 1500px;}
section[data-testid="stSidebar"] {background: rgba(10,16,30,.85); border-right: 1px solid rgba(148,163,184,.12);}
h1, h2, h3, h4 {letter-spacing: .01em;}
.hero-title {font-size: 2.1rem; font-weight: 800; line-height: 1.1;
  background: linear-gradient(90deg, #38bdf8, #fbbf24 55%, #fb7185); -webkit-background-clip: text;
  background-clip: text; color: transparent;}
.hero-sub {color: #94a3b8; margin-top: .25rem; margin-bottom: 1rem;}
.card {background: linear-gradient(160deg, rgba(30,41,59,.65), rgba(15,23,42,.65));
  border: 1px solid rgba(148,163,184,.16); border-radius: 18px; padding: 14px 18px;
  backdrop-filter: blur(6px); box-shadow: 0 10px 30px rgba(0,0,0,.35);}
div[data-testid="stMetric"] {background: linear-gradient(160deg, rgba(30,41,59,.7), rgba(15,23,42,.7));
  border: 1px solid rgba(148,163,184,.16); border-radius: 16px; padding: 14px 16px;
  box-shadow: 0 8px 24px rgba(0,0,0,.30);}
div[data-testid="stMetricLabel"] p {color: #94a3b8; font-size: .78rem; text-transform: uppercase; letter-spacing: .08em;}
div[data-testid="stMetricValue"] {font-weight: 750;}
.badge {display:inline-block; padding: 6px 16px; border-radius: 999px; font-weight: 800;
  letter-spacing: .12em; font-size: .85rem; border: 1px solid; }
.mach-label {color:#94a3b8; text-transform:uppercase; letter-spacing:.14em; font-size:.75rem;}
.stTabs [data-baseweb="tab"] {font-weight: 600;}
</style>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# State - slider and number box stay in sync
# --------------------------------------------------------------------------- #
ss = st.session_state
ss.setdefault("mach_slider", 0.80)
ss.setdefault("mach_num", 0.80)


def _from_slider():
    ss.mach_num = ss.mach_slider


def _from_num():
    ss.mach_slider = ss.mach_num


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### ⚙️ Cycle solver")
    options = ["pyCycle (OpenMDAO)", "NumPy (built-in)"] if tm.PYCYCLE_OK else ["NumPy (built-in)"]
    solver_choice = st.radio("Back-end", options, label_visibility="collapsed")
    backend = "pycycle" if solver_choice.startswith("pyCycle") else "numpy"
    if not tm.PYCYCLE_OK:
        st.caption("pyCycle / OpenMDAO not importable - running the NumPy model.")
        with st.expander("Import error"):
            st.code(tm.PYCYCLE_ERR)

    st.markdown("### 🌍 Flight condition")
    use_q = st.toggle("Constant dynamic-pressure trajectory", value=False,
                      help="Altitude follows the Mach number so that q stays constant.")
    if use_q:
        q_kpa = st.slider("Dynamic pressure q  [kPa]", 10.0, 90.0, 50.0, 1.0)
        alt_km_manual = 0.0
    else:
        alt_km_manual = st.slider("Altitude  [km]", 0.0, 30.0, 10.0, 0.5)
        q_kpa = 50.0

    st.markdown("### 🔥 Operation")
    throttle = st.slider("Throttle (T4 setting)", 0.50, 1.00, 1.00, 0.01)
    m_t0, m_t1 = st.slider("Mode transition band  [Mach]", 1.5, 5.0, (2.5, 3.5), 0.1)

    with st.expander("Turbojet flowpath", expanded=False):
        w_corr = st.slider("Corrected airflow SLS  [kg/s]", 20.0, 90.0, 45.0, 1.0)
        a_tj = st.slider("Capture area  [m²]", 0.2, 1.2, 0.45, 0.01)
        pr_des = st.slider("Max compressor PR", 4.0, 20.0, 10.0, 0.5)
        t4_tj = st.slider("Max turbine-inlet T4  [K]", 1300, 2000, 1650, 10)
        t3_max = st.slider("Compressor-exit T3 limit  [K]", 700, 1100, 900, 10)
    with st.expander("Ramjet / dual-mode flowpath", expanded=False):
        a_rj = st.slider("Capture area  [m²] ", 0.2, 1.6, 0.60, 0.01)
        t4_rj = st.slider("Max burner-exit T4  [K]", 1600, 2600, 2200, 10)

params = tm.EngineParams(w_corr=w_corr, a_tj=a_tj, pr_des=pr_des, t4_tj=float(t4_tj),
                         t3_max=float(t3_max), a_rj=a_rj, t4_rj=float(t4_rj),
                         m_t0=float(m_t0), m_t1=float(m_t1), throttle=throttle)


def altitude_m(mach: float) -> float:
    return tm.alt_for_q(mach, q_kpa * 1e3) if use_q else alt_km_manual * 1e3


# --------------------------------------------------------------------------- #
# Header + Mach control (number box  <->  slider)
# --------------------------------------------------------------------------- #
st.markdown('<div class="hero-title">TBCC Engine Lab</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Turbine-Based Combined Cycle · turbojet → ramjet mode transition · '
            'OpenMDAO + pyCycle + NumPy</div>', unsafe_allow_html=True)

c_num, c_sld = st.columns([1, 4], vertical_alignment="bottom")
with c_num:
    st.number_input("Flight Mach number", min_value=0.0, max_value=6.0, step=0.05,
                    format="%.2f", key="mach_num", on_change=_from_num)
with c_sld:
    st.slider("Mach", 0.0, 6.0, step=0.05, key="mach_slider", on_change=_from_slider,
              label_visibility="collapsed")

mach = float(ss.mach_num)
alt_m = altitude_m(mach)

with st.spinner("Solving cycle…"):
    pt = tm.evaluate(mach, alt_m, params, backend)

for n in pt.notes:
    st.warning(n, icon="⚠️")

# --------------------------------------------------------------------------- #
# Engine schematic (SVG)
# --------------------------------------------------------------------------- #
def schematic(w_tj: float, w_rj: float, mach: float) -> str:
    tj_o = 0.22 + 0.78 * w_tj
    rj_o = 0.22 + 0.78 * w_rj
    flame_tj, flame_rj = 0.15 + 0.85 * w_tj, 0.15 + 0.85 * w_rj
    blades = "".join(
        f'<rect x="{232 + i * 19}" y="{80 + i * 1.5}" width="6" height="{34 - i * 3}" rx="2" fill="{C_TJ}"/>'
        for i in range(6))
    tblades = "".join(
        f'<rect x="{500 + i * 17}" y="{84 - i * 1.5}" width="6" height="{26 + i * 3}" rx="2" fill="{C_RJ}"/>'
        for i in range(4))
    holders = "".join(
        f'<rect x="{330 + i * 55}" y="{190 + (i % 2) * 6}" width="5" height="{34 - (i % 2) * 12}" rx="2" fill="#fde68a"/>'
        for i in range(7))
    speed = max(0.6, 3.2 - mach * 0.45)
    return f"""
<svg viewBox="0 0 960 290" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:Inter,Segoe UI,sans-serif">
<defs>
 <linearGradient id="gtj" x1="0" x2="1"><stop offset="0" stop-color="{C_TJ}" stop-opacity=".30"/><stop offset="1" stop-color="{C_TJ}" stop-opacity=".08"/></linearGradient>
 <linearGradient id="grj" x1="0" x2="1"><stop offset="0" stop-color="{C_RJ}" stop-opacity=".10"/><stop offset="1" stop-color="{C_RJ}" stop-opacity=".32"/></linearGradient>
 <radialGradient id="fl"><stop offset="0" stop-color="#fff7ae"/><stop offset=".5" stop-color="#fb923c"/><stop offset="1" stop-color="#ef4444" stop-opacity="0"/></radialGradient>
 <linearGradient id="plume" x1="0" x2="1"><stop offset="0" stop-color="#fdba74" stop-opacity=".9"/><stop offset="1" stop-color="#fb7185" stop-opacity="0"/></linearGradient>
</defs>
<style>
 .flow{{stroke-dasharray:10 14;animation:dash {speed:.2f}s linear infinite}}
 @keyframes dash{{to{{stroke-dashoffset:-48}}}}
 .lbl{{fill:{MUTED};font-size:11px;letter-spacing:.08em}}
 .ttl{{font-size:13px;font-weight:700;letter-spacing:.14em}}
</style>
<!-- turbojet flowpath -->
<g opacity="{tj_o:.2f}">
 <path d="M40,46 L210,70 L700,70 L785,82 L915,58 L915,134 L785,110 L700,120 L210,120 L40,144 Z" fill="url(#gtj)" stroke="{C_TJ}" stroke-width="1.6"/>
 {blades}
 <rect x="352" y="76" width="120" height="38" rx="14" fill="#7c2d12" stroke="#fb923c"/>
 <ellipse cx="412" cy="95" rx="52" ry="15" fill="url(#fl)" opacity="{flame_tj:.2f}"/>
 {tblades}
 <path d="M790,92 L915,92 L915,100 L790,100 Z" fill="url(#plume)" opacity="{flame_tj:.2f}"/>
 <path class="flow" d="M30,95 L900,95" stroke="{C_TJ}" stroke-width="2" fill="none" opacity=".55"/>
 <text class="ttl" x="42" y="32" fill="{C_TJ}">TURBOJET  ·  {w_tj*100:.0f}%</text>
 <text class="lbl" x="232" y="140">COMPRESSOR</text><text class="lbl" x="372" y="140">BURNER</text><text class="lbl" x="498" y="140">TURBINE</text><text class="lbl" x="800" y="152">C-D NOZZLE</text>
</g>
<!-- ramjet flowpath -->
<g opacity="{rj_o:.2f}">
 <path d="M40,176 L250,198 L700,198 L785,208 L915,184 L915,266 L785,244 L700,242 L250,242 L40,268 Z" fill="url(#grj)" stroke="{C_RJ}" stroke-width="1.6"/>
 <rect x="300" y="204" width="400" height="32" rx="12" fill="#7c2d12" opacity=".55"/>
 <ellipse cx="500" cy="220" rx="190" ry="14" fill="url(#fl)" opacity="{flame_rj:.2f}"/>
 {holders}
 <path d="M790,216 L915,216 L915,230 L790,230 Z" fill="url(#plume)" opacity="{flame_rj:.2f}"/>
 <path class="flow" d="M30,221 L900,221" stroke="{C_RJ}" stroke-width="2" fill="none" opacity=".55"/>
 <text class="ttl" x="42" y="166" fill="{C_RJ}">RAMJET / DUAL-MODE  ·  {w_rj*100:.0f}%</text>
 <text class="lbl" x="300" y="262">COMBUSTOR + FLAMEHOLDERS</text><text class="lbl" x="800" y="278">C-D NOZZLE</text>
</g>
</svg>"""


# --------------------------------------------------------------------------- #
# Hero row: Mach gauge + schematic + mode badge
# --------------------------------------------------------------------------- #
def gauge(mach: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=mach, number={"valueformat": ".2f", "font": {"size": 54, "color": INK}},
        gauge={"axis": {"range": [0, 6], "tickcolor": MUTED, "tickfont": {"color": MUTED}, "dtick": 1},
               "bar": {"color": "#f8fafc", "thickness": 0.22}, "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
               "steps": [{"range": [0, params.m_t0], "color": "rgba(56,189,248,.55)"},
                         {"range": [params.m_t0, params.m_t1], "color": "rgba(251,191,36,.55)"},
                         {"range": [params.m_t1, 6], "color": "rgba(251,113,133,.55)"}],
               "threshold": {"line": {"color": "#ffffff", "width": 4}, "thickness": 0.9, "value": mach}},
        domain={"x": [0, 1], "y": [0, 1]}))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=0), paper_bgcolor="rgba(0,0,0,0)",
                      font={"color": INK})
    return fig


badge_col = {"TURBOJET": C_TJ, "MODE TRANSITION": C_TR, "RAMJET": C_RJ}[pt.mode]
h1, h2 = st.columns([1, 2.3], gap="medium")
with h1:
    st.markdown('<div class="card"><div class="mach-label">Mach speed</div>', unsafe_allow_html=True)
    st.plotly_chart(gauge(mach), use_container_width=True, config={"displayModeBar": False})
    st.markdown(
        f'<div style="text-align:center;margin-bottom:6px">'
        f'<span class="badge" style="color:{badge_col};border-color:{badge_col};'
        f'background:{badge_col}1f">{pt.mode}</span></div></div>', unsafe_allow_html=True)
with h2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    components.html(f'<body style="margin:0;background:transparent">{schematic(pt.w_tj, pt.w_rj, mach)}</body>',
                    height=300)
    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# KPIs
# --------------------------------------------------------------------------- #
k = st.columns(6)
k[0].metric("Net thrust", f"{pt.F / 1e3:,.1f} kN")
k[1].metric("Specific impulse", f"{pt.isp:,.0f} s")
k[2].metric("TSFC", f"{pt.tsfc:,.1f} g/kN·s")
k[3].metric("Air flow", f"{pt.W:,.1f} kg/s")
k[4].metric("Fuel flow", f"{pt.mdot_f:,.2f} kg/s")
k[5].metric("Airspeed", f"{pt.fl.V:,.0f} m/s", f"{pt.fl.V * 3.6:,.0f} km/h", delta_color="off")

k2 = st.columns(6)
k2[0].metric("Altitude", f"{pt.fl.alt / 1e3:,.1f} km", f"{pt.fl.alt * 3.28084:,.0f} ft", delta_color="off")
k2[1].metric("Dynamic pressure", f"{pt.fl.q / 1e3:,.1f} kPa")
k2[2].metric("Ambient T / P", f"{pt.fl.T0:,.0f} K", f"{pt.fl.P0 / 1e3:,.2f} kPa", delta_color="off")
k2[3].metric("Total temp Tt0", f"{pt.fl.Tt0:,.0f} K")
k2[4].metric("Turbojet share", f"{pt.w_tj * 100:,.0f} %")
k2[5].metric("Solver", pt.backend)

# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False, max_entries=24)
def cached_sweep(params_t: tuple, use_q_: bool, q_kpa_: float, alt_km_: float, backend_: str, n: int):
    p = tm.EngineParams(*params_t)
    fn = (lambda m: tm.alt_for_q(m, q_kpa_ * 1e3)) if use_q_ else (lambda m: alt_km_ * 1e3)
    return tm.sweep(np.linspace(0.05, 6.0, n), fn, p, backend_)


def style(fig: go.Figure, h: int = 360) -> go.Figure:
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(color=INK),
                      legend=dict(orientation="h", y=1.12, x=0),
                      xaxis=dict(gridcolor="rgba(148,163,184,.12)", zeroline=False),
                      yaxis=dict(gridcolor="rgba(148,163,184,.12)", zeroline=False))
    return fig


def band(fig: go.Figure):
    fig.add_vrect(x0=params.m_t0, x1=params.m_t1, fillcolor=C_TR, opacity=0.10, line_width=0)
    fig.add_vline(x=mach, line_color="#ffffff", line_width=2, line_dash="dot")


tab1, tab2, tab3 = st.tabs(["📈 Performance sweep", "🌡️ Cycle stations", "ℹ️ Model notes"])

with tab1:
    n_pts = 36 if backend == "pycycle" else 120
    with st.spinner("Sweeping Mach range…"):
        sw = cached_sweep(astuple(params), use_q, q_kpa, alt_km_manual, backend, n_pts)
    for n in sw["notes"][:2]:
        st.warning(n, icon="⚠️")
    M = sw["M"]
    col_a, col_b = st.columns(2)

    fa = go.Figure()
    fa.add_trace(go.Scatter(x=M, y=sw["F_tj"] / 1e3, name="Turbojet", mode="lines", stackgroup="one",
                            line=dict(color=C_TJ, width=1.5), fillcolor="rgba(56,189,248,.35)"))
    fa.add_trace(go.Scatter(x=M, y=sw["F_rj"] / 1e3, name="Ramjet", mode="lines", stackgroup="one",
                            line=dict(color=C_RJ, width=1.5), fillcolor="rgba(251,113,133,.35)"))
    fa.add_trace(go.Scatter(x=M, y=sw["F"] / 1e3, name="Total", mode="lines", line=dict(color="#f8fafc", width=2.5)))
    fa.add_trace(go.Scatter(x=[mach], y=[pt.F / 1e3], mode="markers", name="Now",
                            marker=dict(size=12, color="#fff", line=dict(color=badge_col, width=3))))
    band(fa)
    fa.update_layout(title="Net thrust vs Mach", xaxis_title="Mach", yaxis_title="Thrust [kN]")
    col_a.plotly_chart(style(fa), use_container_width=True, config={"displayModeBar": False})

    fb = go.Figure()
    fb.add_trace(go.Scatter(x=M, y=sw["isp"], name="Isp", mode="lines", line=dict(color=C_TR, width=3)))
    fb.add_trace(go.Scatter(x=[mach], y=[pt.isp], mode="markers", name="Now",
                            marker=dict(size=12, color="#fff", line=dict(color=badge_col, width=3))))
    band(fb)
    fb.update_layout(title="Specific impulse vs Mach", xaxis_title="Mach", yaxis_title="Isp [s]")
    col_b.plotly_chart(style(fb), use_container_width=True, config={"displayModeBar": False})

    col_c, col_d = st.columns(2)
    fc = go.Figure(go.Scatter(x=M, y=sw["mdot_f"], mode="lines", line=dict(color="#a78bfa", width=3),
                              fill="tozeroy", fillcolor="rgba(167,139,250,.18)", name="Fuel flow"))
    band(fc)
    fc.update_layout(title="Fuel flow vs Mach", xaxis_title="Mach", yaxis_title="kg/s")
    col_c.plotly_chart(style(fc, 300), use_container_width=True, config={"displayModeBar": False})

    fd = go.Figure(go.Scatter(x=M, y=sw["alt"] / 1e3, mode="lines", line=dict(color="#34d399", width=3), name="Altitude"))
    band(fd)
    fd.update_layout(title="Flight corridor", xaxis_title="Mach", yaxis_title="Altitude [km]")
    col_d.plotly_chart(style(fd, 300), use_container_width=True, config={"displayModeBar": False})

with tab2:
    ft = go.Figure()
    rows = []
    for name, res, w, colr in (("Turbojet", pt.tj, pt.w_tj, C_TJ), ("Ramjet", pt.rj, pt.w_rj, C_RJ)):
        if res is None:
            continue
        labels = [s[0] for s in res.stations]
        ft.add_trace(go.Scatter(x=labels, y=[s[1] for s in res.stations], mode="lines+markers",
                                name=f"{name} ({w * 100:.0f}%)", line=dict(color=colr, width=3),
                                marker=dict(size=10), opacity=0.35 + 0.65 * w))
        for s in res.stations:
            rows.append({"Flowpath": name, "Station": s[0], "Tt [K]": round(s[1], 1),
                         "Pt [kPa]": round(s[2], 2)})
    ft.update_layout(title="Total temperature through the engine", yaxis_title="Tt [K]")
    cs1, cs2 = st.columns([1.5, 1])
    cs1.plotly_chart(style(ft, 400), use_container_width=True, config={"displayModeBar": False})
    cs2.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=400)
    e1, e2 = st.columns(2)
    if pt.tj:
        e1.metric("Turbojet exhaust velocity", f"{pt.tj.Ve:,.0f} m/s")
    if pt.rj:
        e2.metric("Ramjet exhaust velocity", f"{pt.rj.Ve:,.0f} m/s")

with tab3:
    st.markdown(
        """
**Architecture** – two parallel flowpaths share the airframe inlet. A smoothstep schedule across the
*mode-transition band* moves airflow from the turbojet to the ramjet; thrust and fuel flow are the
airflow-weighted sum of both.

| Layer | Role |
|---|---|
| **NumPy** | ISA atmosphere, MIL-E-5007D ram recovery, flow & compressor-ratio schedules, blending, Mach sweeps, built-in cycle model (warm start + fallback) |
| **OpenMDAO** | Newton solver and `BalanceComp`s: burner-exit temperature (→ fuel/air ratio) and shaft power balance (→ turbine PR) |
| **pyCycle** | Real-gas (CEA, JP-7) `FlightConditions`, `Inlet`, `Compressor`, `Combustor`, `Turbine`, C-D `Nozzle`, `Performance` |

**Assumptions** – fully-expanded variable-geometry nozzles, no installation losses or bypass drag, constant
efficiencies, compressor PR limited by the T3 ceiling, no dissociation in the NumPy model. Scramjet
(supersonic-combustion) operation is out of scope, so Mach is limited to 6.
"""
    )
