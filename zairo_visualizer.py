import streamlit as st
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import soundfile as sf
from datetime import datetime

st.set_page_config(page_title="Zairo Analysis Visualizer", layout="wide")
st.title("🧪 Zairo Analysis Visualizer")
st.caption("Load TestReport JSONs + Guide TXT → visual comparison of cuts & swing behaviour")

# ====================== HELPERS ======================
def parse_reports(uploaded_files):
    reports = []
    for f in uploaded_files:
        content = f.getvalue().decode("utf-8")
        data = json.loads(content)
        if isinstance(data, dict) and "tests" in data:
            reports.extend(data["tests"])
        elif isinstance(data, dict):
            reports.append(data)
        elif isinstance(data, list):
            reports.extend(data)
    return reports


def parse_guide(uploaded_file):
    if uploaded_file is None:
        return []
    text = uploaded_file.getvalue().decode("utf-8")
    cuts = []
    for line in text.splitlines():
        if line.strip() and line[0].isdigit():
            try:
                t = float(line.split("s")[0].strip())
                cuts.append(t)
            except Exception:
                pass
    return cuts


def load_audio(uploaded_file):
    if uploaded_file is None:
        return None, None
    audio, sr = sf.read(uploaded_file)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def get_report(test):
    return test.get("report", test)


def get_window(test):
    report = get_report(test)
    meta = report.get("input_metadata", {})
    start = meta.get("start_sec", 0.0)
    end = meta.get("end_sec", None)
    if end is None:
        end = test.get("duration_sec", 0)
        if end == 0:
            cuts = report.get("final_cuts_detailed", [])
            if cuts:
                end = max(c.get("local_time_seconds", 0) for c in cuts) + 1.0
    return start, end

# ====================== SIDEBAR — FILE LOADING ======================
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

# Visualize button — load once, store in session state
if st.sidebar.button("Visualize", type="primary", use_container_width=True):
    with st.spinner("Loading files..."):
        st.session_state["reports"] = parse_reports(report_files)
        st.session_state["guide_cuts"] = parse_guide(guide_file)
        audio, sr = load_audio(wav_file)
        if audio is not None:
            step = max(1, sr // 4410)
            st.session_state["audio_down"] = audio[::step]
            st.session_state["audio_sr"] = sr / step
        else:
            st.session_state["audio_down"] = None
            st.session_state["audio_sr"] = None

if "reports" not in st.session_state:
    st.info("Upload your files above, then press **Visualize** when ready.")
    st.stop()

reports = st.session_state["reports"]
guide_cuts = st.session_state["guide_cuts"]
audio_down = st.session_state["audio_down"]
audio_sr = st.session_state["audio_sr"]

st.sidebar.success(f"Loaded {len(reports)} test reports")

# ====================== TEST SELECTOR ======================
test_options = [
    f"Test {i+1} — {r.get('style','?')} / swing:{r.get('swing_level','?')}"
    for i, r in enumerate(reports)
]
selected_idx = st.selectbox(
    "Select test to visualise", range(len(reports)),
    format_func=lambda i: test_options[i]
)
selected = reports[selected_idx]

# ====================== METADATA HEADER ======================
col1, col2, col3, col4 = st.columns(4)
win_start, win_end = get_window(selected)
with col1:
    st.metric("Style", selected.get("style", "hybrid"))
with col2:
    st.metric("Swing Level", selected.get("swing_level", "off"))
with col3:
    st.metric("Window Duration", f"{selected.get('duration_sec', 0):.2f}s")
with col4:
    st.metric("Window", f"{win_start:.1f}s → {win_end:.1f}s")

st.subheader(f"Test {selected_idx+1} — {selected.get('style')} / {selected.get('swing_level')}")

# ====================== WAVEFORM VISUALISATION ======================
st.subheader("📈 Waveform + Cuts")

filter_col1, filter_col2 = st.columns(2)
with filter_col1:
    crop_to_window = st.checkbox("Crop to analysis window", value=True)
with filter_col2:
    show_rejected = st.checkbox("Show rejected candidates", value=False)

report = get_report(selected)
cuts = report.get("final_cuts_detailed", [])
rejected = report.get("rejection_summary", [])

# Build waveform trace
if audio_down is not None and audio_sr:
    full_times = np.linspace(0, len(audio_down) / audio_sr, len(audio_down))

    if crop_to_window and win_end > win_start:
        mask = (full_times >= win_start) & (full_times <= win_end)
        plot_times = full_times[mask]
        plot_audio = audio_down[mask]
    else:
        plot_times = full_times
        plot_audio = audio_down
else:
    duration = win_end - win_start if crop_to_window else selected.get("duration_sec", 30)
    plot_times = np.linspace(win_start if crop_to_window else 0, win_start + duration, 1000)
    plot_audio = np.zeros(1000)
    if audio_down is None:
        st.warning("No WAV loaded — waveform is a placeholder.")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=plot_times, y=plot_audio,
    mode='lines', name='Waveform',
    line=dict(color='#1f77b4', width=1),
    opacity=0.7
))

# Engine cuts
for c in cuts:
    t_local = c.get("local_time_seconds") or c.get("time", 0)
    t_abs = t_local + win_start
    t = t_abs if (audio_down is not None or not crop_to_window) else t_local
    if crop_to_window and (t < win_start or t > win_end):
        continue
    score = c.get("score", 0)
    color = "#2ca02c" if c.get("wave") == "primary" else "#ff7f0e"
    if c.get("swing_reason") == "ghost_note":
        color = "#d62728"
    fig.add_vline(
        x=t, line_dash="dash", line_color=color, line_width=2,
        annotation_text=f"{score:.2f}", annotation_position="top"
    )

# Rejected candidates (dimmed, only when toggled)
if show_rejected:
    for r in rejected:
        t_r = r.get("time", 0)
        if crop_to_window and (t_r < win_start or t_r > win_end):
            continue
        fig.add_vline(
            x=t_r, line_dash="dot", line_color="#555555", line_width=1,
            annotation_text=f"✗{r.get('score', 0):.2f}",
            annotation_position="bottom",
            annotation_font_color="#777777",
        )

# Guide cuts
for t in guide_cuts:
    if crop_to_window and (t < win_start or t > win_end):
        continue
    fig.add_vline(
        x=t, line_dash="dot", line_color="#9467bd", line_width=1.5,
        annotation_text="G", annotation_position="bottom"
    )

# Analysis window markers (when not cropped)
if not crop_to_window and win_end > win_start:
    fig.add_vrect(x0=win_start, x1=win_end, fillcolor="#2ca02c", opacity=0.08,
                  annotation_text="Analysis Window", annotation_position="top left")

fig.update_layout(
    title="Waveform — Green: primary | Orange: swing | Red: ghost | Purple: guide TXT",
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
if "rejection_summary" in report:
    st.subheader("🚫 Top Rejected Candidates")
    rej = pd.DataFrame(report["rejection_summary"])
    st.dataframe(rej, use_container_width=True)

st.success("✅ Switch tests freely above — no reload needed.")
st.caption("Built as part of the Zairo robust engine project — no shortcuts.")
