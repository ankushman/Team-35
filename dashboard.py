"""
dashboard.py - Flask Web Dashboard

Serves a live web UI that displays:
  - Current CPU / Memory / Disk gauges
  - System status (Normal / Anomaly)
  - Last executed action badge
  - Live action execution log (auto-updates via JS polling)

Run independently:
    python dashboard.py [--simulate] [--port 5000]
"""

import threading
from flask import Flask, jsonify, render_template_string
from monitor import get_metrics
from anomaly_detector import detect_anomaly
import action_log

app = Flask(__name__)

# Global flag — set to True to enable CPU spike simulation
SIMULATE_SPIKE = False

# ── HTML template ──────────────────────────────────────────────────────────────
DASHBOARD_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>AIOps Self-Healing Dashboard</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-body: #080a12;
      --bg-card: rgba(17,20,34,0.85);
      --bg-card-solid: #111422;
      --border: rgba(255,255,255,0.06);
      --text-1: #eef0fa;
      --text-2: #6b7394;
      --text-3: #3a3f58;
      --accent: #00d4ff;
      --accent-glow: rgba(0,212,255,0.15);
      --green: #4caf50;
      --orange: #ff9800;
      --red: #f44336;
      --purple: #bb86fc;
      --cyan: #4fc3f7;
      --status-ok-bg: rgba(76,175,80,0.1);
      --status-bad-bg: rgba(244,67,54,0.1);
      --gauge-track: rgba(255,255,255,0.06);
      --row-hover: rgba(255,255,255,0.03);
      --toggle-bg: #1e2235;
      --toggle-knob: #00d4ff;
    }
    [data-theme="light"] {
      --bg-body: #f0f2f7;
      --bg-card: rgba(255,255,255,0.8);
      --bg-card-solid: #ffffff;
      --border: rgba(0,0,0,0.08);
      --text-1: #1a1d2e;
      --text-2: #6b7394;
      --text-3: #b0b8d0;
      --accent: #0078b4;
      --accent-glow: rgba(0,120,180,0.1);
      --status-ok-bg: rgba(76,175,80,0.12);
      --status-bad-bg: rgba(244,67,54,0.1);
      --gauge-track: rgba(0,0,0,0.07);
      --row-hover: rgba(0,0,0,0.02);
      --toggle-bg: #d8dce6;
      --toggle-knob: #ff9800;
    }

    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: 'Inter', system-ui, sans-serif;
      background: var(--bg-body);
      color: var(--text-1);
      min-height: 100vh;
      padding: 0;
      transition: background 0.4s, color 0.4s;
    }

    /* ── Top bar ── */
    .topbar {
      display: flex; align-items: center; justify-content: space-between;
      padding: 16px 40px;
      border-bottom: 1px solid var(--border);
      background: var(--bg-card);
      backdrop-filter: blur(16px);
      position: sticky; top: 0; z-index: 100;
      transition: background 0.4s, border-color 0.4s;
    }
    .topbar-left { display: flex; align-items: center; gap: 14px; }
    .topbar-logo {
      font-size: 1.25rem; font-weight: 800; color: var(--accent);
      letter-spacing: 1.5px; transition: color 0.4s;
    }
    .topbar-divider { width: 1px; height: 22px; background: var(--border); }
    .topbar-status {
      display: flex; align-items: center; gap: 8px;
      font-size: 0.78rem; color: var(--text-2); transition: color 0.4s;
    }
    .live-dot {
      width: 7px; height: 7px; background: var(--green); border-radius: 50%;
      animation: blink 1.4s ease-in-out infinite;
    }
    @keyframes blink { 0%,100%{opacity:1} 50%{opacity:.2} }

    .topbar-right { display: flex; align-items: center; gap: 20px; }
    .topbar-ts { font-size: 0.75rem; color: var(--text-2); transition: color 0.4s; }

    /* ── Theme toggle ── */
    .theme-toggle {
      display: flex; align-items: center; gap: 8px;
      cursor: pointer; user-select: none;
    }
    .toggle-icon { font-size: 0.95rem; color: var(--text-2); transition: color 0.4s; }
    .toggle-track {
      width: 44px; height: 24px; background: var(--toggle-bg); border-radius: 12px;
      position: relative; transition: background 0.4s; border: 1px solid var(--border);
    }
    .toggle-knob {
      width: 18px; height: 18px; background: var(--toggle-knob); border-radius: 50%;
      position: absolute; top: 2px; left: 3px;
      transition: transform 0.35s cubic-bezier(.4,0,.2,1), background 0.4s;
      box-shadow: 0 2px 6px rgba(0,0,0,0.25);
    }
    [data-theme="light"] .toggle-knob { transform: translateX(19px); }

    /* ── Container ── */
    .container { max-width: 1200px; margin: 0 auto; padding: 28px 32px 60px; }

    /* ── Section title ── */
    .section-title {
      font-size: 0.7rem; font-weight: 600; color: var(--text-2);
      text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 16px;
      transition: color 0.4s;
    }

    /* ── Gauge cards ── */
    .gauge-row {
      display: grid; grid-template-columns: repeat(3, 1fr);
      gap: 20px; margin-bottom: 24px;
    }
    .gauge-card {
      background: var(--bg-card); backdrop-filter: blur(12px);
      border: 1px solid var(--border); border-radius: 16px;
      padding: 28px 20px 24px; text-align: center;
      transition: background 0.4s, border-color 0.4s, box-shadow 0.3s;
    }
    .gauge-card:hover { box-shadow: 0 8px 32px var(--accent-glow); }
    .gauge-label {
      font-size: 0.7rem; font-weight: 600; color: var(--text-2);
      text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 16px;
      transition: color 0.4s;
    }
    .gauge-svg { width: 120px; height: 120px; margin: 0 auto 12px; display: block; }
    .gauge-track-circle { fill: none; stroke: var(--gauge-track); stroke-width: 8; }
    .gauge-fill {
      fill: none; stroke-width: 8; stroke-linecap: round;
      transition: stroke-dashoffset 0.8s cubic-bezier(.4,0,.2,1), stroke 0.4s;
      transform: rotate(-90deg); transform-origin: center;
    }
    .gauge-text {
      font-size: 1.5rem; font-weight: 700; fill: var(--text-1);
      dominant-baseline: central; text-anchor: middle; transition: fill 0.4s;
    }
    .gauge-sub {
      font-size: 0.65rem; fill: var(--text-2);
      dominant-baseline: central; text-anchor: middle;
    }

    /* ── Status banner ── */
    .status-banner {
      padding: 14px 28px; border-radius: 12px; text-align: center;
      font-size: 0.95rem; font-weight: 600; letter-spacing: 0.8px;
      margin-bottom: 24px;
      transition: background 0.4s, border-color 0.4s, color 0.4s;
    }
    .status-normal {
      background: var(--status-ok-bg); border: 1px solid var(--green); color: var(--green);
    }
    .status-anomaly {
      background: var(--status-bad-bg); border: 1px solid var(--red); color: var(--red);
      animation: pulse 1.4s ease-in-out infinite;
    }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.5} }

    /* ── Bottom grid ── */
    .bottom-grid { display: grid; grid-template-columns: 340px 1fr; gap: 20px; align-items: start; }

    /* ── Panel ── */
    .panel {
      background: var(--bg-card); backdrop-filter: blur(12px);
      border: 1px solid var(--border); border-radius: 16px; padding: 24px;
      transition: background 0.4s, border-color 0.4s;
    }
    .panel-title {
      font-size: 0.7rem; font-weight: 600; color: var(--text-2);
      text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 18px;
      transition: color 0.4s;
    }

    /* ── Last action ── */
    #last-action-box { display: flex; flex-direction: column; gap: 10px; }
    .action-badge {
      display: inline-block; padding: 6px 16px; border-radius: 20px;
      font-size: 0.82rem; font-weight: 600; width: fit-content;
      transition: background 0.4s, border-color 0.4s, color 0.4s;
    }
    .badge-restart { background: rgba(187,134,252,0.12); border: 1px solid var(--purple); color: var(--purple); }
    .badge-cache   { background: rgba(79,195,247,0.12);  border: 1px solid var(--cyan);   color: var(--cyan); }
    .badge-nothing { background: rgba(76,175,80,0.12);   border: 1px solid var(--green);  color: var(--green); }
    .badge-waiting { background: rgba(107,115,148,0.1);  border: 1px solid var(--text-2); color: var(--text-2); }
    .action-meta    { font-size: 0.75rem; color: var(--text-2); }
    .action-trigger { font-size: 0.8rem;  color: var(--text-2); font-style: italic; }
    .action-result  { font-size: 0.8rem;  color: var(--green); }

    /* ── Table ── */
    #action-table-wrap {
      max-height: 340px; overflow-y: auto;
      scrollbar-width: thin; scrollbar-color: var(--border) transparent;
    }
    table { width: 100%; border-collapse: collapse; font-size: 0.8rem; }
    thead th {
      color: var(--text-2); text-align: left; padding: 8px 12px;
      border-bottom: 1px solid var(--border); font-weight: 500;
      position: sticky; top: 0; background: var(--bg-card-solid);
      transition: background 0.4s, color 0.4s;
    }
    tbody tr { border-bottom: 1px solid var(--border); transition: background 0.15s; }
    tbody tr:hover { background: var(--row-hover); }
    tbody td { padding: 10px 12px; vertical-align: top; }
    .td-time { color: var(--text-2); white-space: nowrap; }
    .td-action-restart { color: var(--purple); font-weight: 600; }
    .td-action-cache   { color: var(--cyan);   font-weight: 600; }
    .td-trigger { color: var(--text-2); font-style: italic; }
    .td-metrics { color: var(--text-3); white-space: nowrap; }
    .td-result  { color: var(--green); }
    #empty-state {
      text-align: center; color: var(--text-2);
      padding: 32px 0; font-size: 0.82rem; display: none;
    }

    footer {
      text-align: center; margin-top: 40px;
      color: var(--text-3); font-size: 0.72rem; transition: color 0.4s;
    }

    @media (max-width: 800px) {
      .topbar { padding: 14px 16px; flex-wrap: wrap; gap: 10px; }
      .container { padding: 20px 16px 40px; }
      .gauge-row { grid-template-columns: 1fr; }
      .bottom-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>

  <!-- ── Top bar ── -->
  <div class="topbar">
    <div class="topbar-left">
      <span class="topbar-logo">&#9881; AIOps</span>
      <div class="topbar-divider"></div>
      <div class="topbar-status">
        <span class="live-dot"></span>
        <span>Live</span>
        <span>&nbsp;·&nbsp;</span>
        <span>Simulation: <strong>{{ simulate }}</strong></span>
      </div>
    </div>
    <div class="topbar-right">
      <span class="topbar-ts" id="ts">{{ timestamp }}</span>
      <div class="theme-toggle" id="theme-toggle" title="Toggle dark / light mode">
        <span class="toggle-icon">&#9790;</span>
        <div class="toggle-track"><div class="toggle-knob"></div></div>
        <span class="toggle-icon">&#9728;</span>
      </div>
    </div>
  </div>

  <!-- ── Main content ── -->
  <div class="container">

    <!-- Gauge cards -->
    <div class="section-title">System Metrics</div>
    <div class="gauge-row">
      {% set gauges = [
        ('CPU Usage',    cpu,    'cpu',  60, 80),
        ('Memory Usage', memory, 'mem',  60, 85),
        ('Disk Usage',   disk,   'disk', 70, 90)
      ] %}
      {% for label, val, key, lo, hi in gauges %}
      {% set color = '#4caf50' if val < lo else ('#ff9800' if val < hi else '#f44336') %}
      <div class="gauge-card" id="card-{{ key }}">
        <div class="gauge-label">{{ label }}</div>
        <svg class="gauge-svg" viewBox="0 0 120 120">
          <circle class="gauge-track-circle" cx="60" cy="60" r="50" />
          <circle class="gauge-fill" id="gf-{{ key }}"
                  cx="60" cy="60" r="50"
                  stroke="{{ color }}"
                  stroke-dasharray="314.16"
                  stroke-dashoffset="{{ 314.16 - (314.16 * val / 100) }}" />
          <text class="gauge-text" id="gt-{{ key }}" x="60" y="56">{{ val }}%</text>
          <text class="gauge-sub" x="60" y="76">of 100%</text>
        </svg>
      </div>
      {% endfor %}
    </div>

    <!-- Status banner -->
    <div id="status-banner" class="status-banner {{ 'status-anomaly' if anomaly else 'status-normal' }}">
      {% if anomaly %}
        &#9888; ANOMALY DETECTED &mdash; Self-healing in progress
      {% else %}
        &#10003; System Normal
      {% endif %}
    </div>

    <!-- Bottom panels -->
    <div class="bottom-grid">

      <!-- Last action panel -->
      <div class="panel">
        <div class="panel-title">Last Executed Action</div>
        <div id="last-action-box">
          <span class="action-badge badge-waiting" id="la-badge">Waiting for anomaly&hellip;</span>
          <span class="action-meta" id="la-meta"></span>
          <span class="action-trigger" id="la-trigger"></span>
          <span class="action-result" id="la-result"></span>
        </div>
      </div>

      <!-- Action history -->
      <div class="panel">
        <div class="panel-title">Action Execution Log</div>
        <div id="action-table-wrap">
          <table>
            <thead>
              <tr>
                <th>Time</th><th>Action</th><th>Trigger</th><th>Metrics</th><th>Result</th>
              </tr>
            </thead>
            <tbody id="action-tbody"></tbody>
          </table>
          <div id="empty-state">No actions executed yet.<br>Waiting for an anomaly&hellip;</div>
        </div>
      </div>

    </div><!-- /bottom-grid -->
  </div><!-- /container -->

  <footer>AIOps Self-Healing Monitor &copy; 2026</footer>

<script>
  /* ── Constants ── */
  const CIRC = 314.16; // 2 * PI * 50

  /* ── Gauge helpers ── */
  function gaugeColor(v, lo, hi) {
    return v < lo ? '#4caf50' : v < hi ? '#ff9800' : '#f44336';
  }

  function setGauge(key, value, lo, hi) {
    const fill = document.getElementById('gf-' + key);
    const txt  = document.getElementById('gt-' + key);
    if (!fill || !txt) return;
    const c = gaugeColor(value, lo, hi);
    fill.style.strokeDashoffset = CIRC - (CIRC * Math.min(value, 100) / 100);
    fill.style.stroke = c;
    txt.textContent = value + '%';
  }

  /* ── Badge helpers ── */
  function actionBadgeClass(a) {
    if (a === 'restart_service') return 'badge-restart';
    if (a === 'clear_cache')     return 'badge-cache';
    return 'badge-nothing';
  }
  function actionLabel(a) {
    if (a === 'restart_service') return '&#9881; Restart Service';
    if (a === 'clear_cache')     return '&#128465; Clear Cache';
    return '&#10003; Do Nothing';
  }
  function actionTdClass(a) {
    if (a === 'restart_service') return 'td-action-restart';
    if (a === 'clear_cache')     return 'td-action-cache';
    return '';
  }

  /* ── Poll metrics ── */
  async function pollMetrics() {
    try {
      const r = await fetch('/api/metrics');
      const d = await r.json();

      setGauge('cpu',  d.cpu,    60, 80);
      setGauge('mem',  d.memory, 60, 85);
      setGauge('disk', d.disk,   70, 90);

      const banner = document.getElementById('status-banner');
      if (d.anomaly) {
        banner.className = 'status-banner status-anomaly';
        banner.innerHTML = '&#9888; ANOMALY DETECTED &mdash; Self-healing in progress';
      } else {
        banner.className = 'status-banner status-normal';
        banner.innerHTML = '&#10003; System Normal';
      }

      document.getElementById('ts').textContent = new Date().toLocaleTimeString();
    } catch(e) {}
  }

  /* ── Poll actions ── */
  let _seenCount = 0;

  async function pollActions() {
    try {
      const r = await fetch('/api/actions');
      const actions = await r.json();

      const tbody = document.getElementById('action-tbody');
      const empty = document.getElementById('empty-state');

      if (actions.length === 0) { empty.style.display = 'block'; return; }
      empty.style.display = 'none';

      if (actions.length === _seenCount) return;
      _seenCount = actions.length;

      tbody.innerHTML = actions.map(a => `
        <tr>
          <td class="td-time">${a.timestamp}</td>
          <td class="${actionTdClass(a.action)}">${a.action.replace(/_/g,' ')}</td>
          <td class="td-trigger">${a.trigger}</td>
          <td class="td-metrics">CPU ${a.cpu}% / Mem ${a.memory}%</td>
          <td class="td-result">${a.result}</td>
        </tr>
      `).join('');

      const la = actions[0];
      const badge = document.getElementById('la-badge');
      badge.className = 'action-badge ' + actionBadgeClass(la.action);
      badge.innerHTML = actionLabel(la.action);
      document.getElementById('la-meta').textContent    = '@ ' + la.timestamp;
      document.getElementById('la-trigger').textContent = la.trigger;
      document.getElementById('la-result').textContent  = la.result;

      if (tbody.firstElementChild) {
        tbody.firstElementChild.style.background = 'var(--accent-glow)';
        setTimeout(() => {
          if (tbody.firstElementChild) tbody.firstElementChild.style.background = '';
        }, 1200);
      }
    } catch(e) {}
  }

  /* ── Theme toggle ── */
  (function() {
    const toggle = document.getElementById('theme-toggle');
    const root = document.documentElement;
    const saved = localStorage.getItem('aiops-theme') || 'dark';
    root.setAttribute('data-theme', saved);

    toggle.addEventListener('click', function() {
      const next = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
      root.setAttribute('data-theme', next);
      localStorage.setItem('aiops-theme', next);
    });
  })();

  /* ── Start polling ── */
  pollMetrics();
  pollActions();
  setInterval(pollMetrics, 3000);
  setInterval(pollActions, 2000);
</script>
</body>
</html>
"""


# ── Flask routes ───────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Render the main dashboard page (initial server-side render)."""
    from datetime import datetime
    metrics = get_metrics(simulate_spike=SIMULATE_SPIKE)
    anomaly = detect_anomaly(metrics["cpu"], metrics["memory"])
    return render_template_string(
        DASHBOARD_HTML,
        cpu=metrics["cpu"],
        memory=metrics["memory"],
        disk=metrics["disk"],
        anomaly=anomaly,
        simulate="ON" if SIMULATE_SPIKE else "OFF",
        timestamp=datetime.now().strftime("%H:%M:%S"),
    )


@app.route("/api/metrics")
def api_metrics():
    """JSON endpoint — live system metrics + anomaly flag."""
    metrics = get_metrics(simulate_spike=SIMULATE_SPIKE)
    metrics["anomaly"] = detect_anomaly(metrics["cpu"], metrics["memory"])
    return jsonify(metrics)


@app.route("/api/actions")
def api_actions():
    """JSON endpoint — list of all executed corrective actions (newest first)."""
    return jsonify(action_log.get_all())


# ── Runner ─────────────────────────────────────────────────────────────────────

def run_dashboard(host: str = "0.0.0.0", port: int = 5000,
                  simulate: bool = False, debug: bool = False) -> None:
    """
    Start the Flask dashboard.

    Args:
        host:     Interface to listen on.
        port:     TCP port.
        simulate: Enable CPU spike simulation.
        debug:    Enable Flask debug mode (do not use in production).
    """
    global SIMULATE_SPIKE
    SIMULATE_SPIKE = simulate
    print(f"[DASHBOARD] Starting on http://{host}:{port}  (simulate={simulate})")
    app.run(host=host, port=port, debug=debug, use_reloader=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AIOps Web Dashboard")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--simulate", action="store_true",
                        help="Enable CPU spike simulation")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    run_dashboard(port=args.port, simulate=args.simulate, debug=args.debug)
