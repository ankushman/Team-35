"""
anomaly_detector.py - Anomaly Detection Module

Applies simple threshold-based rules to decide whether the system
is in an anomalous state.  Thresholds are kept as module-level
constants so they are easy to tune.
"""

# Thresholds (percentage)
CPU_THRESHOLD = 80.0
MEMORY_THRESHOLD = 85.0


def detect_anomaly(cpu: float, memory: float) -> bool:
    """
    Determine whether the given metrics represent an anomaly.

    Rules:
        - CPU usage > CPU_THRESHOLD  → anomaly
        - Memory usage > MEMORY_THRESHOLD → anomaly

    Args:
        cpu:    CPU usage percentage (0–100).
        memory: Memory usage percentage (0–100).

    Returns:
        True if an anomaly is detected, False otherwise.
    """
    if cpu > CPU_THRESHOLD:
        return True
    if memory > MEMORY_THRESHOLD:
        return True
    return False


def anomaly_reasons(cpu: float, memory: float) -> list[str]:
    """Return a human-readable list of triggered anomaly conditions."""
    reasons = []
    if cpu > CPU_THRESHOLD:
        reasons.append(f"High CPU: {cpu}% (threshold {CPU_THRESHOLD}%)")
    if memory > MEMORY_THRESHOLD:
        reasons.append(f"High Memory: {memory}% (threshold {MEMORY_THRESHOLD}%)")
    return reasons
