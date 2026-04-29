"""
main.py - AIOps Main Controller

Ties together all modules into a continuous monitoring loop:
  1. Collect system metrics (monitor.py)
  2. Detect anomalies    (anomaly_detector.py)
  3. Decide on action    (agent.py)
  4. Execute the action  (fixer.py)

Also starts the Flask dashboard in a background thread so both
the terminal monitor and the web UI run simultaneously.

Usage:
    python main.py              # real metrics
    python main.py --simulate   # CPU spike simulation
    python main.py --no-web     # terminal only, no dashboard
"""

import os
import sys
import time
import argparse
import threading
from datetime import datetime

# Ensure imports always resolve relative to this file's directory so that
# 'import action_log' in fixer.py and dashboard.py land on the same object.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.chdir(_HERE)

from monitor import get_metrics
from anomaly_detector import detect_anomaly, anomaly_reasons
from agent import decide_action, ACTION_DO_NOTHING
from fixer import execute_action
import dashboard  # Flask app lives here


# ── Helpers ────────────────────────────────────────────────────────────────────

SEPARATOR = "─" * 50


def log(level: str, message: str) -> None:
    """Print a timestamped log line."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {message}")


# ── Monitoring loop ────────────────────────────────────────────────────────────

def monitoring_loop(simulate_spike: bool = False, interval: int = 3) -> None:
    """
    Infinite loop that collects metrics, detects anomalies,
    and triggers corrective actions.

    Args:
        simulate_spike: Forward the flag to the metrics collector.
        interval:       Seconds to wait between each cycle.
    """
    print(SEPARATOR)
    print("  AIOps Self-Healing Monitor")
    print(f"  Simulation mode : {'ON  (CPU spikes 85–95%)' if simulate_spike else 'OFF (real metrics)'}")
    print(f"  Poll interval   : {interval}s")
    print(SEPARATOR)

    iteration = 0

    while True:
        iteration += 1
        print(f"\n{'═'*50}")
        log("INFO", f"Cycle #{iteration} — Collecting system metrics")

        # ── Step 1: Collect ───────────────────────────────────────────────────
        metrics = get_metrics(simulate_spike=simulate_spike)
        print(f"  CPU    : {metrics['cpu']}%")
        print(f"  Memory : {metrics['memory']}%")
        print(f"  Disk   : {metrics['disk']}%")

        # ── Step 2: Detect ────────────────────────────────────────────────────
        is_anomaly = detect_anomaly(metrics["cpu"], metrics["memory"])

        if is_anomaly:
            log("WARNING", "Anomaly detected!")
            for reason in anomaly_reasons(metrics["cpu"], metrics["memory"]):
                print(f"  ↳ {reason}")

            # ── Step 3: Decide ────────────────────────────────────────────────
            action = decide_action(metrics["cpu"], metrics["memory"])
            log("AI", f"Decision: {action}")

            # ── Step 4: Execute ───────────────────────────────────────────────
            execute_action(action, cpu=metrics["cpu"], memory=metrics["memory"])
        else:
            log("INFO", "System status: NORMAL — no action required.")

        time.sleep(interval)


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AIOps Self-Healing System Monitor"
    )
    parser.add_argument(
        "--simulate", action="store_true",
        help="Simulate CPU spikes (85–95%%) for demonstration purposes"
    )
    parser.add_argument(
        "--no-web", action="store_true",
        help="Disable the web dashboard (terminal output only)"
    )
    parser.add_argument(
        "--port", type=int, default=5000,
        help="Port for the web dashboard (default: 5000)"
    )
    parser.add_argument(
        "--interval", type=int, default=3,
        help="Monitoring poll interval in seconds (default: 3)"
    )
    args = parser.parse_args()

    # Start Flask dashboard in a background daemon thread
    if not args.no_web:
        dashboard.SIMULATE_SPIKE = args.simulate
        web_thread = threading.Thread(
            target=dashboard.run_dashboard,
            kwargs={
                "host": "0.0.0.0",
                "port": args.port,
                "simulate": args.simulate,
                "debug": False,
            },
            daemon=True,
            name="DashboardThread",
        )
        web_thread.start()
        print(f"[DASHBOARD] Web UI running at http://localhost:{args.port}")

    # Run the monitoring loop in the main thread
    try:
        monitoring_loop(simulate_spike=args.simulate, interval=args.interval)
    except KeyboardInterrupt:
        print("\n[INFO] Monitoring stopped by user.")


if __name__ == "__main__":
    main()
