"""
AIOps RL Agent — Main Entry Point (Week 1)
===========================================
1. Trains the anomaly detector on synthetic normal data.
2. Runs the metrics simulator in a loop.
3. Detects anomalies in each sample.
4. Logs every record to CSV.
5. Prints a summary line to the terminal.
"""

import sys
import os

# ---------------------------------------------------------------------------
# Make sure the project root is on the Python path so that imports work
# regardless of where the script is invoked from.
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.simulator.metrics_simulator import generate_metrics
from src.anomaly_detection.detector import AnomalyDetector
from src.utils.logger import log_metrics


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NUM_ITERATIONS = 20                          # Number of metric samples to generate
LOG_FILE = os.path.join(PROJECT_ROOT, "data", "metrics_log.csv")


def main() -> None:
    """Run one cycle of the AIOps monitoring pipeline."""

    # ── Step 1: Train the anomaly detector ─────────────────────────────
    print("=" * 60)
    print("  AIOps RL Agent — Week 1 Demo")
    print("=" * 60)
    print()

    detector = AnomalyDetector()
    detector.train(n_samples=1000)
    print()

    # ── Step 2: Simulate, detect, log, print ───────────────────────────
    print(f"{'CPU':>5} | {'MEM':>5} | {'LAT':>5} | {'DISK':>5} | STATUS")
    print("-" * 50)

    for _ in range(NUM_ITERATIONS):
        # Generate metrics
        metrics = generate_metrics()

        # Detect anomaly
        status = detector.detect(metrics)

        # Log to CSV
        log_metrics(metrics, status, filepath=LOG_FILE)

        # Print to terminal
        print(
            f"CPU: {metrics['cpu']:>5} | "
            f"Memory: {metrics['memory']:>5} | "
            f"Latency: {metrics['latency']:>5} | "
            f"Disk: {metrics['disk_io']:>5} | "
            f"Status: {status}"
        )

    print()
    print(f"[INFO] {NUM_ITERATIONS} records logged to {LOG_FILE}")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
