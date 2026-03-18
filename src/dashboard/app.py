"""
AIOps RL Agent -- Real-Time Dashboard (Week 3)
================================================
Streamlit app that integrates the metrics simulator, anomaly detector,
and trained DQN agent into a single interactive dashboard.

Run with:
    streamlit run src/dashboard/app.py
"""

import sys
import os
import time

# ---------------------------------------------------------------------------
# Project path setup (so imports work from any working directory)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from src.rl_agent.environment import AIOpsEnv, ACTION_NAMES
from src.utils.logger import log_metrics

# Try importing Stable-Baselines3
try:
    from stable_baselines3 import DQN
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "dqn_aiops_model.zip")
LOG_FILE   = os.path.join(PROJECT_ROOT, "data", "metrics_log.csv")
MAX_HISTORY = 50      # Rolling window for charts


# ===================================================================
# PAGE CONFIG
# ===================================================================
st.set_page_config(
    page_title="AIOps RL Agent Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ===================================================================
# CUSTOM CSS -- premium dark theme styling
# ===================================================================
st.markdown("""
<style>
    /* ---------- Global ---------- */
    .main { background-color: #0e1117; }

    /* ---------- Metric cards ---------- */
    .metric-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #16192b 100%);
        border: 1px solid #2d3348;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-bottom: 8px;
    }
    .metric-card h3 {
        color: #8b95b0;
        font-size: 0.85rem;
        margin: 0 0 6px 0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }

    /* ---------- Status badges ---------- */
    .badge-healthy {
        background: linear-gradient(135deg, #00c853, #00e676);
        color: #0e1117;
        padding: 6px 20px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1rem;
        display: inline-block;
    }
    .badge-warning {
        background: linear-gradient(135deg, #ff9100, #ffc400);
        color: #0e1117;
        padding: 6px 20px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1rem;
        display: inline-block;
    }
    .badge-critical {
        background: linear-gradient(135deg, #ff1744, #ff5252);
        color: #ffffff;
        padding: 6px 20px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1rem;
        display: inline-block;
    }

    /* ---------- Anomaly indicator ---------- */
    .anomaly-normal {
        background: linear-gradient(135deg, #00c853, #00e676);
        color: #0e1117;
        padding: 16px;
        border-radius: 12px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 700;
    }
    .anomaly-detected {
        background: linear-gradient(135deg, #ff1744, #ff5252);
        color: #ffffff;
        padding: 16px;
        border-radius: 12px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 700;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }

    /* ---------- Action badge ---------- */
    .action-badge {
        background: linear-gradient(135deg, #7c4dff, #536dfe);
        color: #ffffff;
        padding: 12px 24px;
        border-radius: 12px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: 700;
    }

    /* ---------- Log table ---------- */
    .log-container {
        max-height: 300px;
        overflow-y: auto;
        border-radius: 8px;
    }

    /* ---------- Section headers ---------- */
    .section-header {
        color: #8b95b0;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ===================================================================
# SESSION STATE INITIALIZATION
# ===================================================================
def init_session_state():
    """Initialise all session-state keys on first run."""
    defaults = {
        "running":          False,
        "step":             0,
        "env":              None,
        "model":            None,
        "obs":              None,
        # History lists (rolling window)
        "cpu_history":      [],
        "mem_history":      [],
        "lat_history":      [],
        "disk_history":     [],
        "reward_history":   [],
        "cum_reward":       [],
        "action_history":   [],
        "anomaly_history":  [],
        # Latest values
        "current_metrics":  {"cpu": 0, "memory": 0, "latency": 0, "disk_io": 0},
        "current_anomaly":  0,
        "current_action":   "---",
        "current_reward":   0.0,
        "total_reward":     0.0,
        "anomalies_count":  0,
        "action_counts":    {i: 0 for i in range(5)},
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()


# ===================================================================
# ENVIRONMENT & MODEL SETUP
# ===================================================================
@st.cache_resource(show_spinner="Loading RL environment & model...")
def load_env_and_model():
    """Create environment and load trained DQN model (cached)."""
    env = AIOpsEnv()
    model = None
    if SB3_AVAILABLE and os.path.isfile(MODEL_PATH):
        model = DQN.load(MODEL_PATH)
    return env, model


# ===================================================================
# HELPER FUNCTIONS
# ===================================================================
def get_system_health(metrics: dict, anomaly: int) -> str:
    """Determine overall system health from metrics and anomaly flag."""
    cpu, mem, lat, disk = (
        metrics["cpu"], metrics["memory"],
        metrics["latency"], metrics["disk_io"],
    )
    # Critical: any metric near failure thresholds
    if cpu >= 95 or mem >= 95 or lat >= 450 or disk >= 90 or anomaly == 1:
        return "Critical"
    # Warning: elevated but not critical
    if cpu >= 75 or mem >= 75 or lat >= 200 or disk >= 60:
        return "Warning"
    return "Healthy"


def make_line_chart(y_data, title, y_label, color, range_y=None):
    """Build a Plotly line chart for metric history."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=y_data,
        mode="lines+markers",
        line=dict(color=color, width=2),
        marker=dict(size=4),
        fill="tozeroy",
        fillcolor=color.replace("1)", "0.1)"),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#8b95b0")),
        xaxis=dict(
            title="Step", showgrid=False,
            color="#4a5068", tickfont=dict(size=10),
        ),
        yaxis=dict(
            title=y_label, showgrid=True, gridcolor="#1e2235",
            color="#4a5068", tickfont=dict(size=10),
            range=range_y,
        ),
        height=220,
        margin=dict(l=40, r=20, t=40, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def make_reward_chart(rewards, cum_rewards):
    """Build a dual-axis reward chart."""
    fig = go.Figure()
    # Per-step reward as bars
    colors = ["rgba(0,200,83,0.7)" if r >= 0 else "rgba(255,23,68,0.7)"
              for r in rewards]
    fig.add_trace(go.Bar(
        y=rewards, name="Step Reward",
        marker_color=colors, opacity=0.8,
    ))
    # Cumulative reward as line
    fig.add_trace(go.Scatter(
        y=cum_rewards, name="Cumulative",
        mode="lines", line=dict(color="rgba(124,77,255,1)", width=3),
        yaxis="y2",
    ))
    fig.update_layout(
        title=dict(text="Reward Trend", font=dict(size=14, color="#8b95b0")),
        xaxis=dict(title="Step", showgrid=False, color="#4a5068"),
        yaxis=dict(
            title="Step Reward", showgrid=True, gridcolor="#1e2235",
            color="#4a5068",
        ),
        yaxis2=dict(
            title="Cumulative", overlaying="y", side="right",
            showgrid=False, color="#7c4dff",
        ),
        height=260,
        margin=dict(l=40, r=40, t=40, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            font=dict(color="#8b95b0"),
        ),
        barmode="relative",
    )
    return fig


def make_action_pie(action_counts):
    """Build a donut chart of action distribution."""
    labels = [ACTION_NAMES[i] for i in range(5)]
    values = [action_counts[i] for i in range(5)]
    colors = ["#636efa", "#ef553b", "#00cc96", "#ffa15a", "#ab63fa"]
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.5,
        marker=dict(colors=colors),
        textinfo="percent+label",
        textfont=dict(size=11),
    ))
    fig.update_layout(
        height=240,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(color="#8b95b0", size=10)),
        showlegend=False,
    )
    return fig


# ===================================================================
# SIMULATION STEP
# ===================================================================
def run_one_step():
    """Execute one simulation step and update session state."""
    s = st.session_state

    # Ensure env is set up
    if s["env"] is None:
        env, model = load_env_and_model()
        s["env"] = env
        s["model"] = model
        s["obs"], _ = env.reset()

    env   = s["env"]
    model = s["model"]
    obs   = s["obs"]

    # Choose action
    if model is not None:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
    else:
        action = env.action_space.sample()

    # Step the environment
    obs, reward, terminated, truncated, info = env.step(action)
    s["obs"] = obs
    s["step"] += 1

    # Extract info
    m = info["metrics"]
    s["current_metrics"] = m
    s["current_anomaly"] = info["anomaly"]
    s["current_action"]  = ACTION_NAMES[action]
    s["current_reward"]  = reward
    s["total_reward"]   += reward
    s["action_counts"][action] += 1
    if info["anomaly"]:
        s["anomalies_count"] += 1

    # Append to histories (capped at MAX_HISTORY)
    s["cpu_history"].append(m["cpu"])
    s["mem_history"].append(m["memory"])
    s["lat_history"].append(m["latency"])
    s["disk_history"].append(m["disk_io"])
    s["reward_history"].append(reward)
    s["cum_reward"].append(s["total_reward"])
    s["action_history"].append(ACTION_NAMES[action])
    s["anomaly_history"].append(info["anomaly"])

    for key in ["cpu_history", "mem_history", "lat_history",
                "disk_history", "reward_history", "cum_reward",
                "action_history", "anomaly_history"]:
        if len(s[key]) > MAX_HISTORY:
            s[key] = s[key][-MAX_HISTORY:]

    # Log to CSV
    status_str = "Anomaly Detected" if info["anomaly"] else "Normal"
    log_metrics(m, status_str, filepath=LOG_FILE)

    # Reset env if episode ended
    if terminated or truncated:
        s["obs"], _ = env.reset()


# ===================================================================
# SIDEBAR
# ===================================================================
with st.sidebar:
    st.markdown("## AIOps RL Agent")
    st.markdown("**Week 3 Dashboard**")
    st.markdown("---")

    # Model status
    if SB3_AVAILABLE and os.path.isfile(MODEL_PATH):
        st.success("DQN Model Loaded")
    else:
        st.warning("No model -- using random actions")

    st.markdown("---")

    # Controls
    speed = st.slider(
        "Simulation Speed (seconds)", 0.5, 3.0, 1.0, 0.5,
        help="Delay between each simulation step",
    )

    col_a, col_b = st.columns(2)
    with col_a:
        start = st.button("Start", use_container_width=True, type="primary")
    with col_b:
        stop = st.button("Stop", use_container_width=True)

    step_once = st.button("Step Once", use_container_width=True)

    if start:
        st.session_state["running"] = True
    if stop:
        st.session_state["running"] = False

    st.markdown("---")
    st.markdown(f"**Step:** {st.session_state['step']}")
    st.markdown(f"**Total Reward:** {st.session_state['total_reward']:+.1f}")
    st.markdown(f"**Anomalies:** {st.session_state['anomalies_count']}")


# ===================================================================
# MAIN LAYOUT
# ===================================================================
st.markdown(
    "<h1 style='text-align:center; color:#e0e0e0;'>"
    "AIOps RL Agent Dashboard"
    "</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center; color:#6b7394; margin-top:-10px;'>"
    "Autonomous AI Operations Manager -- Real-Time Monitoring"
    "</p>",
    unsafe_allow_html=True,
)

# ── Row 1: Current metrics cards + anomaly + health ──────────────
r1c1, r1c2, r1c3, r1c4, r1c5, r1c6 = st.columns([1, 1, 1, 1, 1, 1])

m = st.session_state["current_metrics"]

with r1c1:
    st.markdown(
        f'<div class="metric-card"><h3>CPU Usage</h3>'
        f'<p class="value" style="color:#636efa">{m["cpu"]:.1f}%</p></div>',
        unsafe_allow_html=True,
    )

with r1c2:
    st.markdown(
        f'<div class="metric-card"><h3>Memory</h3>'
        f'<p class="value" style="color:#ef553b">{m["memory"]:.1f}%</p></div>',
        unsafe_allow_html=True,
    )

with r1c3:
    st.markdown(
        f'<div class="metric-card"><h3>Latency</h3>'
        f'<p class="value" style="color:#00cc96">{m["latency"]:.1f}ms</p></div>',
        unsafe_allow_html=True,
    )

with r1c4:
    st.markdown(
        f'<div class="metric-card"><h3>Disk I/O</h3>'
        f'<p class="value" style="color:#ffa15a">{m["disk_io"]:.1f}MB/s</p></div>',
        unsafe_allow_html=True,
    )

with r1c5:
    # Anomaly indicator
    if st.session_state["current_anomaly"]:
        st.markdown(
            '<div class="anomaly-detected">ANOMALY DETECTED</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="anomaly-normal">SYSTEM NORMAL</div>',
            unsafe_allow_html=True,
        )

with r1c6:
    # System health
    health = get_system_health(m, st.session_state["current_anomaly"])
    badge_class = {
        "Healthy": "badge-healthy",
        "Warning": "badge-warning",
        "Critical": "badge-critical",
    }[health]
    st.markdown(
        f'<div class="metric-card">'
        f'<h3>System Health</h3>'
        f'<p style="margin-top:8px"><span class="{badge_class}">{health}</span></p>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── Row 2: Metrics line charts ───────────────────────────────────
st.markdown('<p class="section-header">Live Metrics</p>',
            unsafe_allow_html=True)

ch1, ch2, ch3, ch4 = st.columns(4)

with ch1:
    st.plotly_chart(
        make_line_chart(st.session_state["cpu_history"],
                        "CPU Usage (%)", "%", "rgba(99,110,250,1)",
                        range_y=[0, 105]),
        use_container_width=True, key="cpu_chart",
    )
with ch2:
    st.plotly_chart(
        make_line_chart(st.session_state["mem_history"],
                        "Memory (%)", "%", "rgba(239,85,59,1)",
                        range_y=[0, 105]),
        use_container_width=True, key="mem_chart",
    )
with ch3:
    st.plotly_chart(
        make_line_chart(st.session_state["lat_history"],
                        "Latency (ms)", "ms", "rgba(0,204,150,1)",
                        range_y=[0, 520]),
        use_container_width=True, key="lat_chart",
    )
with ch4:
    st.plotly_chart(
        make_line_chart(st.session_state["disk_history"],
                        "Disk I/O (MB/s)", "MB/s", "rgba(255,161,90,1)",
                        range_y=[0, 105]),
        use_container_width=True, key="disk_chart",
    )


# ── Row 3: RL Agent Actions + Reward Chart ───────────────────────
r3c1, r3c2, r3c3 = st.columns([1, 1, 2])

with r3c1:
    st.markdown('<p class="section-header">RL Agent Action</p>',
                unsafe_allow_html=True)
    st.markdown(
        f'<div class="action-badge">'
        f'{st.session_state["current_action"]}'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p style="text-align:center; color:#6b7394; margin-top:8px;">'
        f'Reward: {st.session_state["current_reward"]:+.1f}</p>',
        unsafe_allow_html=True,
    )
    # Action history (last 10)
    st.markdown("**Recent Actions:**")
    recent = st.session_state["action_history"][-10:]
    for i, act in enumerate(reversed(recent)):
        anom = st.session_state["anomaly_history"][-(i + 1)]
        dot = "🔴" if anom else "🟢"
        st.markdown(f"{dot} {act}")

with r3c2:
    st.markdown('<p class="section-header">Action Distribution</p>',
                unsafe_allow_html=True)
    if sum(st.session_state["action_counts"].values()) > 0:
        st.plotly_chart(
            make_action_pie(st.session_state["action_counts"]),
            use_container_width=True, key="pie_chart",
        )
    else:
        st.info("No actions yet. Click **Start** or **Step Once**.")

with r3c3:
    st.markdown('<p class="section-header">Reward Trend</p>',
                unsafe_allow_html=True)
    if len(st.session_state["reward_history"]) > 0:
        st.plotly_chart(
            make_reward_chart(
                st.session_state["reward_history"],
                st.session_state["cum_reward"],
            ),
            use_container_width=True, key="reward_chart",
        )
    else:
        st.info("No data yet. Click **Start** or **Step Once**.")


# ── Row 4: Logs Panel ────────────────────────────────────────────
st.markdown('<p class="section-header">Simulation Logs (last 20 steps)</p>',
            unsafe_allow_html=True)

if st.session_state["step"] > 0:
    n = min(len(st.session_state["cpu_history"]), 20)
    log_data = {
        "Step":    list(range(
            st.session_state["step"] - n + 1,
            st.session_state["step"] + 1,
        )),
        "CPU (%)":     st.session_state["cpu_history"][-n:],
        "Memory (%)":  st.session_state["mem_history"][-n:],
        "Latency (ms)":st.session_state["lat_history"][-n:],
        "Disk I/O":    st.session_state["disk_history"][-n:],
        "Status":      [
            "Anomaly" if a else "Normal"
            for a in st.session_state["anomaly_history"][-n:]
        ],
        "Action":      st.session_state["action_history"][-n:],
        "Reward":      st.session_state["reward_history"][-n:],
    }
    df = pd.DataFrame(log_data)
    # Show newest first
    st.dataframe(
        df.iloc[::-1].reset_index(drop=True),
        use_container_width=True,
        height=300,
    )
else:
    st.info("No simulation data yet. Click **Start** or **Step Once** "
            "to begin.")


# ===================================================================
# AUTO-RUN LOOP / STEP-ONCE
# ===================================================================
if step_once:
    run_one_step()
    st.rerun()

if st.session_state["running"]:
    run_one_step()
    time.sleep(speed)
    st.rerun()
