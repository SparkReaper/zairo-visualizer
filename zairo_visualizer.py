import streamlit as st
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import soundfile as sf
import librosa
from datetime import datetime

st.set_page_config(page_title="Zairo Analysis Visualizer", layout="wide")
st.title("🧪 Zairo Analysis Visualizer")
st.caption("Load TestReport JSONs + Guide TXT → visual comparison of cuts & swing behaviour")

# ====================== FILE LOADING ======================
@st.cache_data
def load_test_report(uploaded_file):
    """Handle both Streamlit UploadedFile and real file paths."""
    if uploaded_file is None:
        return None
    # Streamlit UploadedFile case
    if hasattr(uploaded_file, "getvalue"):
        content = uploaded_file.getvalue().decode("utf-8")
        data = json.loads(content)
    else:
        # Fallback for normal file path
        with open(uploaded_file, "r", encoding="utf-8") as f:
            data = json.load(f)

    # Handle the wrapper format you have (with "tests" key)
    if isinstance(data, dict) and "tests" in data:
        return data["tests"]
    return [data] if isinstance(data, dict) else data


@st.cache_data
def load_guide_txt(uploaded_file):
    """Handle both Streamlit UploadedFile and real file paths."""
    if uploaded_file is None:
        return ""
    if hasattr(uploaded_file, "getvalue"):
        return uploaded_file.getvalue().decode("utf-8")
    else:
        with open(uploaded_file, "r", encoding="utf-8") as f:
            return f.read()

# Sidebar — file selection
st.sidebar.header("📂 Load Data")
report_files = st.sidebar.file_uploader(
    "Select one or more TestReport JSON files", 
    type=["json"], 
    accept_multiple_files=True
)

guide_file = st.sidebar.file_uploader("Select corresponding Guide TXT (optional)", type=["txt"])

wav_file = st.sidebar.file_uploader("Optional: Original WAV for accurate waveform", type=["wav"])

if not report_files:
    st.info("👉 Upload at least one `zairo_testreport_*.json` file to begin")
    st.stop()

# Load reports
reports = []
for f in report_files:
    tests = load_test_report(f)
    if tests:
        reports.extend(tests)

st.sidebar.success(f"Loaded {len(reports)} test reports")

# Test selector
test_options = [f"Test {i+1} — {r.get('style','?')} / swing:{r.get('swing_level','?')}" for i, r in enumerate(reports)]
selected_idx = st.selectbox("Select test to visualise", range(len(reports)), format_func=lambda i: test_options[i])
selected = reports[selected_idx]

# ====================== METADATA HEADER ======================
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Style", selected.get("style", "hybrid"))
with col2:
    st.metric("Swing Level", selected.get("swing_level", "off"))
with col3:
    st.metric("Window Duration", f"{selected.get('duration_sec', 0):.2f}s")

st.subheader(f"Test {selected_idx+1} — {selected.get('style')} / {selected.get('swing_level')}")

# ====================== WAVEFORM VISUALISATION ======================
st.subheader("📈 Waveform + Cuts")

# Load audio if provided
if wav_file:
    audio, sr = sf.read(wav_file)
else:
    st.warning("No WAV loaded — waveform will be placeholder. Upload the original file for accurate view.")
    audio = np.zeros(44100 * 40)  # dummy
    sr = 44100

# Downsample for plotting
target_sr = 4410  # 10x downsample for smooth plotting
if len(audio) > 0:
    audio_down = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
else:
    audio_down = np.zeros(1000)

times = np.linspace(0, len(audio_down)/target_sr, len(audio_down))

# Extract cuts
cuts = selected.get("report", {}).get("final_cuts_detailed", []) if "report" in selected else selected.get("final_cuts_detailed", [])

fig = go.Figure()

# Waveform
fig.add_trace(go.Scatter(
    x=times, y=audio_down,
    mode='lines', name='Waveform',
    line=dict(color='#1f77b4', width=1),
    opacity=0.7
))

# Engine cuts
for c in cuts:
    t = c.get("local_time_seconds") or c.get("time", 0)
    score = c.get("score", 0)
    color = "#2ca02c" if c.get("wave") == "primary" else "#ff7f0e"
    if c.get("swing_reason") == "ghost_note":
        color = "#d62728"
    fig.add_vline(x=t, line_dash="dash", line_color=color, line_width=2, annotation_text=f"{score:.2f}", annotation_position="top")

# Guide cuts (if loaded)
if guide_file:
    guide_text = load_guide_txt(guide_file)
    # Very simple parser for your guide format
    guide_cuts = []
    for line in guide_text.splitlines():
        if line.strip() and line[0].isdigit():
            try:
                t = float(line.split("s")[0].strip())
                guide_cuts.append(t)
            except:
                pass
    for t in guide_cuts:
        fig.add_vline(x=t, line_dash="dot", line_color="#9467bd", line_width=1.5, annotation_text="G", annotation_position="bottom")

fig.update_layout(
    title="Waveform + Engine Cuts (solid) vs Guide (dotted)",
    xaxis_title="Time (seconds)",
    yaxis_title="Amplitude",
    height=500,
    template="plotly_dark"
)

st.plotly_chart(fig, use_container_width=True)

# ====================== CUT TABLE ======================
st.subheader("📋 Final Cuts Comparison")

df = pd.DataFrame([
    {
        "time": c.get("local_time_seconds") or c.get("time", 0),
        "score": round(c.get("score", 0), 3),
        "wave": c.get("wave", "primary"),
        "swing_reason": c.get("swing_reason", ""),
        "beat_phase": round(c.get("beat_phase", -1), 3),
        "perc_harm": round(c.get("perc_harm_ratio", 1), 3),
        "reasons": " + ".join(c.get("reasons", []))
    }
    for c in cuts
])

st.dataframe(df, use_container_width=True)

# Rejection summary
if "report" in selected and "rejection_summary" in selected["report"]:
    st.subheader("🚫 Top Rejected Candidates")
    rej = pd.DataFrame(selected["report"]["rejection_summary"])
    st.dataframe(rej, use_container_width=True)

st.success("✅ Visualizer ready. Upload more JSONs or change the test selector to compare swing levels visually.")

st.caption("Built as part of the Zairo robust engine project — no shortcuts.")