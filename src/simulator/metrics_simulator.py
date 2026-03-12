"""
Metrics Simulator Module
========================
Simulates IT infrastructure metrics (CPU, memory, network latency, disk I/O).
Occasionally injects anomalies to mimic real-world infrastructure issues.
"""

import random


# ---------------------------------------------------------------------------
# Normal operating ranges for each metric
# ---------------------------------------------------------------------------
NORMAL_RANGES = {
    "cpu":      (10, 75),      # CPU usage in %
    "memory":   (20, 75),      # Memory usage in %
    "latency":  (5, 150),      # Network latency in ms
    "disk_io":  (5, 50),       # Disk I/O in MB/s
}

# Anomalous ranges — values that indicate a problem
ANOMALY_RANGES = {
    "cpu":      (85, 100),
    "memory":   (85, 100),
    "latency":  (250, 500),
    "disk_io":  (70, 100),
}

# Probability that any given sample is anomalous (~15 %)
ANOMALY_PROBABILITY = 0.15


def generate_metrics() -> dict:
    """
    Generate a single sample of infrastructure metrics.

    Returns
    -------
    dict
        Keys: cpu, memory, latency, disk_io — each a float rounded to 1 dp.
    """
    is_anomaly = random.random() < ANOMALY_PROBABILITY

    # Pick ranges based on whether we are injecting an anomaly
    ranges = ANOMALY_RANGES if is_anomaly else NORMAL_RANGES

    metrics = {
        metric: round(random.uniform(lo, hi), 1)
        for metric, (lo, hi) in ranges.items()
    }

    return metrics


# ---------------------------------------------------------------------------
# Quick smoke-test when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    for i in range(10):
        sample = generate_metrics()
        print(f"Sample {i + 1}: {sample}")
