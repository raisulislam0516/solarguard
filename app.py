"""
SolarGuard — Solar Panel Dust & Crack Checker
------------------------------------------------
Upload solar panel image(s) and this app reports, per image:
  1) Soiling level (% dust coverage), estimated via HSV analysis
  2) Whether a crack is likely present, via Canny edge density
  3) An overall Health Score (gauge)
  4) A running history of everything checked this session
  5) A downloadable PDF summary report

No ML model download required — pure OpenCV image processing.

Run with:
    streamlit run app.py
"""

from datetime import datetime

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from fpdf import FPDF

st.set_page_config(page_title="SolarGuard", page_icon="☀️", layout="wide")

# ---------------------- Styling ---------------------- #
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>
    html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }

    .stApp {
        background:
            radial-gradient(circle at 15% 0%, rgba(245,158,11,0.10), transparent 40%),
            radial-gradient(circle at 85% 10%, rgba(59,130,246,0.10), transparent 40%),
            #0b1220;
    }

    .block-container { padding-top: 1.2rem; max-width: 1100px; }

    @keyframes fadeInDown {
        0% { opacity: 0; transform: translateY(-12px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    .hero {
        text-align: center; padding: 1.2rem 1rem 1rem 1rem;
        animation: fadeInDown 0.6s ease-out;
    }
    .hero .badge-chip {
        display: inline-block; background: rgba(245,158,11,0.12);
        color: #fbbf24; border: 1px solid rgba(245,158,11,0.35);
        padding: 0.25rem 0.9rem; border-radius: 999px;
        font-size: 0.75rem; font-weight: 600; letter-spacing: 0.06em;
        text-transform: uppercase; margin-bottom: 0.7rem;
    }
    .hero h1 {
        font-size: 2.8rem; margin: 0.2rem 0; font-weight: 800;
        background: linear-gradient(90deg, #f59e0b, #fbbf24, #fde68a, #f59e0b);
        background-size: 200% auto;
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        animation: shine 6s linear infinite;
    }
    @keyframes shine { to { background-position: 200% center; } }
    .hero p { color: #94a3b8; font-size: 1.05rem; max-width: 520px; margin: 0.3rem auto 0 auto; }

    div[data-testid="stMetric"] {
        background: linear-gradient(155deg, #1e293b, #172033);
        border: 1px solid rgba(148,163,184,0.12);
        border-radius: 16px; padding: 1.1rem;
        box-shadow: 0 6px 18px rgba(0,0,0,0.3);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 24px rgba(0,0,0,0.4);
    }
    div[data-testid="stMetricLabel"] p {
        color: #e2e8f0 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }
    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 800 !important;
        font-size: 2rem !important;
    }
    h3, .stMarkdown h3 { color: #f8fafc !important; font-weight: 700 !important; }
    .stMarkdown p, .stMarkdown li { color: #e2e8f0 !important; }
    section[data-testid="stSidebar"] {
        background: #0f172a !important;
        border-right: 1px solid rgba(148,163,184,0.12);
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li,
    section[data-testid="stSidebar"] h2 {
        color: #f1f5f9 !important;
    }
    section[data-testid="stSidebar"] strong { color: #fbbf24 !important; }

    .badge {
        display: inline-block; padding: 0.4rem 1rem; border-radius: 999px;
        font-weight: 600; font-size: 0.85rem; margin: 4px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.25);
    }
    .badge-danger { background: linear-gradient(135deg,#7f1d1d,#991b1b); color: #fecaca; }
    .badge-safe { background: linear-gradient(135deg,#14532d,#166534); color: #bbf7d0; }
    .badge-warn { background: linear-gradient(135deg,#78350f,#92400e); color: #fde68a; }

    .img-caption { text-align: center; color: #94a3b8; font-size: 0.82rem; margin-top: 0.4rem; letter-spacing: 0.03em; }
    .footer-note { text-align: center; color: #64748b; font-size: 0.8rem; margin-top: 2.5rem; padding-bottom: 1rem; }

    [data-testid="stFileUploaderDropzone"] {
        border-radius: 16px !important;
        border: 1.5px dashed rgba(245,158,11,0.4) !important;
        background: rgba(30,41,59,0.4) !important;
    }

    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0; padding: 0.6rem 1.2rem;
        background: rgba(30,41,59,0.5); font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #f59e0b, #fbbf24) !important;
        color: #0b1220 !important;
    }

    img { border-radius: 12px; }

    div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>☀️ SolarGuard</h1>
    <p>AI-assisted dust &amp; crack inspection dashboard for solar panels — instant health scoring from a single photo.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("ℹ️ How it works")
    st.markdown("""
    **Soiling estimation** — clean panels are darker with higher
    color saturation; dust makes the surface lighter/grayer,
    measured in HSV color space.

    **Crack detection** — Canny edge detection finds sharp,
    irregular lines. High edge density suggests possible damage.

    **Health Score** — combines soiling and crack confidence
    into a single 0–100 rating.
    """)
    st.divider()
    st.caption("Built with OpenCV, Plotly & Streamlit")
    if st.button("🗑️ Clear history", use_container_width=True):
        st.session_state["history"] = []
        st.rerun()


# ---------- Core analysis functions ---------- #
def calculate_soiling(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].astype(np.float32)
    brightness = hsv[:, :, 2].astype(np.float32)
    dust_score = np.mean((255 - saturation) / 255 * (brightness / 255))
    return round(float(dust_score) * 100, 1)


def detect_crack(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    edge_density = np.count_nonzero(edges) / edges.size
    crack_detected = edge_density > 0.045
    confidence = round(min(edge_density * 800, 99), 1) if crack_detected else round((1 - edge_density) * 100, 1)
    return crack_detected, confidence, edges


def health_score(soiling, crack_found, confidence):
    penalty = confidence if crack_found else 0
    score = 100 - (soiling * 0.5) - (penalty * 0.5)
    return round(float(np.clip(score, 0, 100)), 1)


def make_gauge(value, title, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"size": 16, "color": "#94a3b8"}},
        number={"suffix": "%", "font": {"size": 30, "color": "#f8fafc"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#475569"},
            "bar": {"color": color},
            "bgcolor": "#0f172a",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40], "color": "#1e293b"},
                {"range": [40, 70], "color": "#1e293b"},
                {"range": [70, 100], "color": "#1e293b"},
            ],
        },
    ))
    fig.update_layout(
        height=220, margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font={"color": "#f8fafc"},
    )
    return fig


def generate_pdf_report(history):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "SolarGuard - Inspection Report", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(4)
    for i, item in enumerate(history, 1):
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, f"{i}. {item['name']}", ln=True)
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 7, f"   Soiling: {item['soiling']}%  |  Health Score: {item['health']}%", ln=True)
        crack_txt = f"Crack: {'Yes' if item['crack'] else 'No'} ({item['confidence']}% confidence)"
        pdf.cell(0, 7, f"   {crack_txt}", ln=True)
        pdf.ln(2)
    return bytes(pdf.output())


if "history" not in st.session_state:
    st.session_state["history"] = []

# ---------------------- Tabs ---------------------- #
tab_inspect, tab_history = st.tabs(["🔍 Inspect", "📊 History & Report"])

with tab_inspect:
    uploaded_files = st.file_uploader(
        "Upload one or more solar panel images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            soiling = calculate_soiling(image)
            crack_found, confidence, edge_image = detect_crack(image)
            health = health_score(soiling, crack_found, confidence)

            st.session_state["history"].append({
                "name": uploaded_file.name, "soiling": soiling,
                "crack": crack_found, "confidence": confidence, "health": health,
            })

            st.markdown(f"### 📷 {uploaded_file.name}")

            img_col, edge_col, gauge_col = st.columns([1, 1, 1.2])
            with img_col:
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
                st.markdown('<div class="img-caption">Original</div>', unsafe_allow_html=True)
            with edge_col:
                st.image(edge_image, use_container_width=True)
                st.markdown('<div class="img-caption">Edge Detection</div>', unsafe_allow_html=True)
            with gauge_col:
                gauge_color = "#22c55e" if health >= 70 else ("#f59e0b" if health >= 40 else "#ef4444")
                st.plotly_chart(make_gauge(health, "Health Score", gauge_color), use_container_width=True)

            m1, m2, m3 = st.columns(3)
            m1.metric("Soiling Level", f"{soiling}%")
            m2.metric("Crack Confidence", f"{confidence}%")
            m3.metric("Health Score", f"{health}%")

            badges = []
            badges.append(
                f'<span class="badge badge-danger">⚠️ Crack likely</span>' if crack_found
                else '<span class="badge badge-safe">✅ No crack detected</span>'
            )
            badges.append(
                '<span class="badge badge-warn">🧹 Needs cleaning</span>' if soiling > 20
                else '<span class="badge badge-safe">✨ Reasonably clean</span>'
            )
            st.markdown(" ".join(badges), unsafe_allow_html=True)
            st.divider()
    else:
        st.info("👆 Upload one or more images above to get started.")

with tab_history:
    if st.session_state["history"]:
        st.markdown("### 📋 Session History")
        st.dataframe(
            [{"Image": h["name"], "Soiling %": h["soiling"], "Health %": h["health"],
              "Crack": "Yes" if h["crack"] else "No", "Confidence %": h["confidence"]}
             for h in st.session_state["history"]],
            use_container_width=True,
        )

        avg_health = round(np.mean([h["health"] for h in st.session_state["history"]]), 1)
        cracked = sum(1 for h in st.session_state["history"] if h["crack"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Panels Checked", len(st.session_state["history"]))
        c2.metric("Avg. Health Score", f"{avg_health}%")
        c3.metric("Panels with Cracks", cracked)

        pdf_bytes = generate_pdf_report(st.session_state["history"])
        st.download_button(
            "📄 Download PDF Report", data=pdf_bytes,
            file_name="solarguard_report.pdf", mime="application/pdf",
            use_container_width=True,
        )
    else:
        st.info("No panels checked yet — go to the Inspect tab to get started.")

st.markdown('<div class="footer-note">SolarGuard • OpenCV-based image analysis prototype</div>', unsafe_allow_html=True)
