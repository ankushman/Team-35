<<<<<<< HEAD
# Team-35
Mini Project 2k26
=======
# 🤖 AIOps RL Agent — Autonomous AI Operations Manager

> **B.Tech Mini Project** — Week 1: Foundation

An autonomous system that simulates IT infrastructure metrics, detects anomalies using machine learning, and (in future weeks) uses reinforcement learning to take automated corrective actions.

---

## 📁 Project Structure

```
aiops-rl-agent/
│
├── src/
│   ├── simulator/
│   │   └── metrics_simulator.py    # Generates simulated infra metrics
│   ├── anomaly_detection/
│   │   └── detector.py             # Isolation Forest anomaly detector
│   ├── utils/
│   │   └── logger.py               # CSV data logger
│   └── main.py                     # Main entry point
│
├── data/
│   └── metrics_log.csv             # Auto-generated metrics log
│
├── tests/                          # Unit tests (Week 2+)
├── notebooks/                      # Jupyter notebooks (Week 2+)
├── docs/                           # Documentation
│
├── .github/workflows/
│   └── ci.yml                      # GitHub Actions CI pipeline
│
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/aiops-rl-agent.git
cd aiops-rl-agent
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the system

```bash
python src/main.py
```

### Expected output

```
============================================================
  AIOps RL Agent — Week 1 Demo
============================================================

[INFO] Anomaly detector trained on 1000 normal samples.

  CPU |   MEM |   LAT |  DISK | STATUS
--------------------------------------------------
CPU:  45.2 | Memory:  55.0 | Latency:  80.3 | Disk:  25.1 | Status: Normal
CPU:  98.1 | Memory:  92.4 | Latency: 350.0 | Disk:  85.7 | Status: Anomaly Detected
...
```

A CSV log file is saved automatically to `data/metrics_log.csv`.

---

## 🔬 Modules

| Module | Description |
|--------|-------------|
| `metrics_simulator.py` | Generates random CPU, memory, latency, disk I/O metrics with ~15% anomaly injection |
| `detector.py` | Trains an Isolation Forest on normal data and classifies new samples |
| `logger.py` | Appends timestamped metrics + status to a CSV file |
| `main.py` | Orchestrates the full pipeline |

---

## ⚙️ CI/CD

GitHub Actions runs on every push:
1. Installs Python 3.11 + dependencies
2. Executes `python src/main.py`
3. Verifies `data/metrics_log.csv` was created

---

## 📅 Roadmap

| Week | Milestone |
|------|-----------|
| **1** ✅ | Simulator, anomaly detection, logging, CI/CD |
| **2** | Reinforcement learning agent (Gymnasium) |
| **3** | Streamlit dashboard + visualization |
| **4** | Integration, testing, deployment |

---

## 📄 License

This project is for academic purposes (B.Tech Mini Project).
>>>>>>> b303812 (Week 1: AIOps RL Agent foundation - simulator, anomaly detection, logging, CI/CD)
