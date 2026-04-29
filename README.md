# AIOps Self-Healing System Monitor

A miniature **Autonomous AI Operations (AIOps)** demo written in Python.  
The system watches CPU and memory usage, detects anomalies, and automatically applies corrective actions — all while serving a live web dashboard.

---

## What is AIOps?

AIOps (Artificial Intelligence for IT Operations) applies machine learning and automation to IT operations tasks such as event correlation, anomaly detection, and self-healing.  
This demo distils those concepts into a small, readable Python project.

---

## How It Works

```
┌──────────┐    metrics    ┌──────────────────┐   anomaly?  ┌─────────┐   action  ┌────────┐
│ monitor  │ ────────────► │ anomaly_detector │ ──────────► │  agent  │ ─────────► │ fixer  │
└──────────┘               └──────────────────┘             └─────────┘            └────────┘
      │                                                                                  │
      └──────────────────────── Flask Dashboard (dashboard.py) ◄────────────────────────┘
```

1. **monitor.py** — Collects CPU, memory, and disk usage via `psutil`.  
2. **anomaly_detector.py** — Flags high CPU (>80%) or high memory (>85%) as anomalies.  
3. **agent.py** — Rule-based AI agent that picks the best corrective action.  
4. **fixer.py** — Executes the chosen action (simulated in demo mode).  
5. **dashboard.py** — Flask web UI that auto-refreshes every 5 seconds.  
6. **main.py** — Wires everything together; runs the monitor loop and the dashboard simultaneously.

---

## Project Structure

```
aiops_demo/
├── main.py               # Entry point — runs monitor + dashboard
├── monitor.py            # Collects system metrics
├── anomaly_detector.py   # Threshold-based anomaly detection
├── agent.py              # AI decision agent
├── fixer.py              # Self-healing corrective actions
├── dashboard.py          # Flask web dashboard
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Full system (monitor + web dashboard)

```bash
python main.py
```

Open your browser at **http://localhost:5000**

### With CPU spike simulation (demonstrates anomaly detection without stressing your machine)

```bash
python main.py --simulate
```

### Terminal only (no web dashboard)

```bash
python main.py --no-web
```

### Web dashboard only

```bash
python dashboard.py
python dashboard.py --simulate   # with simulation
```

### Command-line options

| Flag | Default | Description |
|------|---------|-------------|
| `--simulate` | off | Generate fake CPU spikes (85–95%) |
| `--no-web` | off | Disable the Flask dashboard |
| `--port` | 5000 | Dashboard port |
| `--interval` | 3 | Monitoring poll interval (seconds) |

---

## Example Terminal Output

```
──────────────────────────────────────────────────
  AIOps Self-Healing Monitor
  Simulation mode : ON  (CPU spikes 85–95%)
  Poll interval   : 3s
──────────────────────────────────────────────────

══════════════════════════════════════════════════
[10:42:01] [INFO] Cycle #1 — Collecting system metrics
  CPU    : 91.3%
  Memory : 62.1%
  Disk   : 48.5%
[10:42:02] [WARNING] Anomaly detected!
  ↳ High CPU: 91.3% (threshold 80.0%)
[10:42:02] [AI] Decision: restart_service
[ACTION] Restarting service to reduce system load...
[ACTION] Service restarted successfully.
```

---

## Extending the Project

- Replace threshold rules with a trained `IsolationForest` or `OneClassSVM` from **scikit-learn**.  
- Integrate real remediation (e.g. `systemctl restart`, Redis `FLUSHALL`, Kubernetes rollout).  
- Persist metrics to SQLite or InfluxDB and render historical charts on the dashboard.  
- Add alerting via email or Slack webhook when anomalies are detected.




# Install deps
pip install -r aiops_demo/requirements.txt

# Full system (monitor + dashboard at http://localhost:5000)
python aiops_demo/main.py

# With CPU spike simulation (recommended for demo)
python aiops_demo/main.py --simulate

# Dashboard only
python aiops_demo/dashboard.py --simulate