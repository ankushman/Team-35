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
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      background: #0b0d14;
      color: #dde1f0;
      min-height: 100vh;
      padding: 32px 20px 60px;
    }

    /* ── Header ── */
    header {
      text-align: center;
      margin-bottom: 36px;
    }
    header h1 {
      font-size: 1.9rem;
      color: #00d4ff;
      letter-spacing: 2px;
    }
    #subtitle {
      color: #555e80;
      margin-top: 6px;
      font-size: 0.85rem;
    }

    /* ── Grid layout ── */
    .layout {
      display: grid;
      grid-template-columns: 1fr 1fr;
      grid-template-rows: auto auto;
      gap: 24px;
      max-width: 1100px;
      margin: 0 auto;
    }

    /* metrics panel spans full width */
    .metrics-panel { grid-column: 1 / -1; }

    /* ── Metric cards row ── */
    .cards {
      display: flex;
      flex-wrap: wrap;
      gap: 20px;
      justify-content: center;
    }

    .card {
      background: #13162100;
      border: 1px solid #1e2235;
      border-radius: 14px;
      padding: 24px 28px;
      flex: 1 1 200px;
      max-width: 240px;
      text-align: center;
      background: #131621;
      transition: box-shadow 0.3s;
    }
    .card:hover { box-shadow: 0 4px 28px rgba(0,212,255,0.12); }

    .card .label {
      font-size: 0.75rem;
      color: #555e80;
      text-transform: uppercase;
      letter-spacing: 1px;
      margin-bottom: 12px;
    }
    .card .value { font-size: 2.6rem; font-weight: 700; }
    .card .bar-wrap {
      background: #1e2235;
      border-radius: 8px;
      height: 7px;
      margin-top: 14px;
      overflow: hidden;
    }
    .card .bar { height: 100%; border-radius: 8px; transition: width 0.6s ease; }

    .low  { color: #4caf50; } .bar.low  { background: #4caf50; }
    .mid  { color: #ff9800; } .bar.mid  { background: #ff9800; }
    .high { color: #f44336; } .bar.high { background: #f44336; }

    /* ── Status banner ── */
    .status-banner {
      grid-column: 1 / -1;
      padding: 16px 32px;
      border-radius: 10px;
      font-size: 1.1rem;
      font-weight: 600;
      letter-spacing: 1px;
      text-align: center;
    }
    .status-normal  { background:#152a1e; border:1px solid #4caf50; color:#4caf50; }
    .status-anomaly { background:#2a1515; border:1px solid #f44336; color:#f44336;
                      animation: pulse 1.4s ease-in-out infinite; }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.55} }

    /* ── Last-action badge ── */
    .last-action-panel {
      background: #131621;
      border: 1px solid #1e2235;
      border-radius: 14px;
      padding: 22px 26px;
    }
    .last-action-panel h2 {
      font-size: 0.75rem;
      color: #555e80;
      text-transform: uppercase;
      letter-spacing: 1px;
      margin-bottom: 14px;
    }
    #last-action-box {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .action-badge {
      display: inline-block;
      padding: 5px 14px;
      border-radius: 20px;
      font-size: 0.85rem;
      font-weight: 600;
      letter-spacing: 0.5px;
      width: fit-content;
    }
    .badge-restart { background:#2a1a38; border:1px solid #bb86fc; color:#bb86fc; }
    .badge-cache   { background:#1a2838; border:1px solid #4fc3f7; color:#4fc3f7; }
    .badge-nothing { background:#1a2a1a; border:1px solid #4caf50; color:#4caf50; }
    .badge-waiting { background:#1e2235; border:1px solid #555e80; color:#555e80; }

    .action-meta { font-size: 0.78rem; color: #555e80; }
    .action-trigger { font-size: 0.82rem; color: #aab; font-style: italic; }
    .action-result  { font-size: 0.82rem; color: #8ef5a0; }

    /* ── Action history table ── */
    .history-panel {
      background: #131621;
      border: 1px solid #1e2235;
      border-radius: 14px;
      padding: 22px 26px;
      overflow: hidden;
    }
    .history-panel h2 {
      font-size: 0.75rem;
      color: #555e80;
      text-transform: uppercase;
      letter-spacing: 1px;
      margin-bottom: 14px;
    }
    #action-table-wrap {
      max-height: 320px;
      overflow-y: auto;
      scrollbar-width: thin;
      scrollbar-color: #1e2235 transparent;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.82rem;
    }
    thead th {
      color: #555e80;
      text-align: left;
      padding: 6px 10px;
      border-bottom: 1px solid #1e2235;
      font-weight: 500;
      position: sticky;
      top: 0;
      background: #131621;
    }
    tbody tr { border-bottom: 1px solid #0f1118; transition: background 0.15s; }
    tbody tr:hover { background: #1a1e2e; }
    tbody td { padding: 8px 10px; vertical-align: top; }
    .td-time  { color: #555e80; white-space: nowrap; }
    .td-action-restart { color: #bb86fc; font-weight: 600; }
    .td-action-cache   { color: #4fc3f7; font-weight: 600; }
    .td-trigger { color: #aab; font-style: italic; }
    .td-metrics { color: #778; white-space: nowrap; }
    .td-result  { color: #8ef5a0; }

    #empty-state {
      text-align: center;
      color: #555e80;
      padding: 28px 0;
      font-size: 0.85rem;
      display: none;
    }

    /* ── Live indicator ── */
    .live-dot {
      display: inline-block;
      width: 8px; height: 8px;
      background: #4caf50;
      border-radius: 50%;
      margin-right: 6px;
      animation: blink 1.2s ease-in-out infinite;
    }
    @keyframes blink { 0%,100%{opacity:1} 50%{opacity:.25} }

    /* ── Footer ── */
    footer {
      text-align: center;
      margin-top: 40px;
      color: #2e3450;
      font-size: 0.75rem;
    }

    /* Responsive */
    @media (max-width: 700px) {
      .layout { grid-template-columns: 1fr; }
      .last-action-panel, .history-panel { grid-column: 1; }
    }
  </style>
</head>
<body>
  <header>
    <h1>&#9881; AIOps Self-Healing Dashboard</h1>
    <p id="subtitle">
      <span class="live-dot"></span>
      Live &nbsp;|&nbsp; Simulation: <strong>{{ simulate }}</strong>
      &nbsp;|&nbsp; Last update: <span id="ts">{{ timestamp }}</span>
    </p>
  </header>

  <div class="layout">

    <!-- ── Metric cards ── -->
    <div class="metrics-panel">
      <div class="cards">
        <!-- CPU -->
        {% set cpu_cls = 'low' if cpu < 60 else ('mid' if cpu < 80 else 'high') %}
        <div class="card" id="card-cpu">
          <div class="label">CPU Usage</div>
          <div class="value {{ cpu_cls }}" id="val-cpu">{{ cpu }}%</div>
          <div class="bar-wrap">
            <div class="bar {{ cpu_cls }}" id="bar-cpu" style="width:{{ cpu }}%"></div>
          </div>
        </div>

        <!-- Memory -->
        {% set mem_cls = 'low' if memory < 60 else ('mid' if memory < 85 else 'high') %}
        <div class="card" id="card-mem">
          <div class="label">Memory Usage</div>
          <div class="value {{ mem_cls }}" id="val-mem">{{ memory }}%</div>
          <div class="bar-wrap">
            <div class="bar {{ mem_cls }}" id="bar-mem" style="width:{{ memory }}%"></div>
          </div>
        </div>

        <!-- Disk -->
        {% set disk_cls = 'low' if disk < 70 else ('mid' if disk < 90 else 'high') %}
        <div class="card" id="card-disk">
          <div class="label">Disk Usage</div>
          <div class="value {{ disk_cls }}" id="val-disk">{{ disk }}%</div>
          <div class="bar-wrap">
            <div class="bar {{ disk_cls }}" id="bar-disk" style="width:{{ disk }}%"></div>
          </div>
        </div>
      </div>
    </div>

    <!-- ── Status banner ── -->
    <div id="status-banner" class="status-banner {{ 'status-anomaly' if anomaly else 'status-normal' }}">
      {% if anomaly %}
        &#9888; ANOMALY DETECTED &mdash; Self-healing in progress
      {% else %}
        &#10003; System Normal
      {% endif %}
    </div>

    <!-- ── Last action ── -->
    <div class="last-action-panel">
      <h2>Last Executed Action</h2>
      <div id="last-action-box">
        <span class="action-badge badge-waiting" id="la-badge">Waiting for anomaly&hellip;</span>
        <span class="action-meta" id="la-meta"></span>
        <span class="action-trigger" id="la-trigger"></span>
        <span class="action-result" id="la-result"></span>
      </div>
    </div>

    <!-- ── Action history ── -->
    <div class="history-panel">
      <h2>Automatic Action Execution Log</h2>
      <div id="action-table-wrap">
        <table>
          <thead>
            <tr>
              <th>Time</th>
              <th>Action</th>
              <th>Trigger</th>
              <th>Metrics</th>
              <th>Result</th>
            </tr>
          </thead>
          <tbody id="action-tbody"></tbody>
        </table>
        <div id="empty-state">No actions executed yet.<br>Waiting for an anomaly&hellip;</div>
      </div>
    </div>

  </div>

  <footer>AIOps Demo &copy; 2026</footer>

<script>
  /* ── Utility helpers ── */
  function cls(v, low, mid) {
    return v < low ? 'low' : v < mid ? 'mid' : 'high';
  }

  function setBar(id, value, low, mid) {
    const el = document.getElementById(id);
    if (!el) return;
    const c = cls(value, low, mid);
    el.className = 'bar ' + c;
    el.style.width = Math.min(value, 100) + '%';
  }

  function setVal(id, value, low, mid) {
    const el = document.getElementById(id);
    if (!el) return;
    el.textContent = value + '%';
    el.className = 'value ' + cls(value, low, mid);
  }

  function actionBadgeClass(action) {
    if (action === 'restart_service') return 'badge-restart';
    if (action === 'clear_cache')     return 'badge-cache';
    return 'badge-nothing';
  }

  function actionLabel(action) {
    if (action === 'restart_service') return '&#9881; Restart Service';
    if (action === 'clear_cache')     return '&#128465; Clear Cache';
    return '&#10003; Do Nothing';
  }

  function actionTdClass(action) {
    if (action === 'restart_service') return 'td-action-restart';
    if (action === 'clear_cache')     return 'td-action-cache';
    return '';
  }

  /* ── Poll /api/metrics every 3 s ── */
  async function pollMetrics() {
    try {
      const r = await fetch('/api/metrics');
      const d = await r.json();

      setVal('val-cpu',  d.cpu,    60, 80);
      setVal('val-mem',  d.memory, 60, 85);
      setVal('val-disk', d.disk,   70, 90);
      setBar('bar-cpu',  d.cpu,    60, 80);
      setBar('bar-mem',  d.memory, 60, 85);
      setBar('bar-disk', d.disk,   70, 90);

      const banner = document.getElementById('status-banner');
      if (d.anomaly) {
        banner.className = 'status-banner status-anomaly';
        banner.innerHTML = '&#9888; ANOMALY DETECTED &mdash; Self-healing in progress';
      } else {
        banner.className = 'status-banner status-normal';
        banner.innerHTML = '&#10003; System Normal';
      }

      document.getElementById('ts').textContent = new Date().toLocaleTimeString();
    } catch(e) { /* server might be momentarily busy */ }
  }

  /* ── Poll /api/actions every 2 s ── */
  let _seenCount = 0;

  async function pollActions() {
    try {
      const r = await fetch('/api/actions');
      const actions = await r.json();   // array, newest first

      const tbody = document.getElementById('action-tbody');
      const empty = document.getElementById('empty-state');

      if (actions.length === 0) {
        empty.style.display = 'block';
        return;
      }
      empty.style.display = 'none';

      // Only re-render if count changed (new actions arrived)
      if (actions.length === _seenCount) return;
      _seenCount = actions.length;

      // Rebuild table rows
      tbody.innerHTML = actions.map(a => `
        <tr>
          <td class="td-time">${a.timestamp}</td>
          <td class="${actionTdClass(a.action)}">${a.action.replace(/_/g,' ')}</td>
          <td class="td-trigger">${a.trigger}</td>
          <td class="td-metrics">CPU ${a.cpu}% / Mem ${a.memory}%</td>
          <td class="td-result">${a.result}</td>
        </tr>
      `).join('');

      // Update last-action badge
      const la = actions[0];
      const badge = document.getElementById('la-badge');
      badge.className = 'action-badge ' + actionBadgeClass(la.action);
      badge.innerHTML = actionLabel(la.action);
      document.getElementById('la-meta').textContent    = `@ ${la.timestamp}`;
      document.getElementById('la-trigger').textContent = la.trigger;
      document.getElementById('la-result').textContent  = la.result;

      // Flash highlight on the newest row
      if (tbody.firstElementChild) {
        tbody.firstElementChild.style.background = '#252040';
        setTimeout(() => {
          if (tbody.firstElementChild)
            tbody.firstElementChild.style.background = '';
        }, 1200);
      }
    } catch(e) {}
  }

  // Kick off polling
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
