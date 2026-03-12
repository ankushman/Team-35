"""
Data Logging Utility
====================
Logs infrastructure metrics and their anomaly status to a CSV file,
making it easy to review historical data and feed it into dashboards.
"""

import os
import csv
from datetime import datetime


# CSV column headers
FIELDNAMES = ["timestamp", "cpu", "memory", "latency", "disk_io", "status"]


def log_metrics(metrics: dict, status: str,
                filepath: str = "data/metrics_log.csv") -> None:
    """
    Append a single metrics record to the CSV log file.

    Parameters
    ----------
    metrics : dict
        Must contain keys: cpu, memory, latency, disk_io.
    status : str
        Anomaly detection result, e.g. "Normal" or "Anomaly Detected".
    filepath : str
        Path to the CSV file (created automatically if missing).
    """
    # Ensure the parent directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Check whether the file already exists (to decide on writing headers)
    file_exists = os.path.isfile(filepath)

    with open(filepath, mode="a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)

        # Write header only once
        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "cpu":       metrics["cpu"],
            "memory":    metrics["memory"],
            "latency":   metrics["latency"],
            "disk_io":   metrics["disk_io"],
            "status":    status,
        })


# ---------------------------------------------------------------------------
# Quick smoke-test when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample = {"cpu": 55.2, "memory": 63.0, "latency": 90, "disk_io": 30}
    log_metrics(sample, "Normal", filepath="data/test_log.csv")
    print("Sample logged successfully → data/test_log.csv")
