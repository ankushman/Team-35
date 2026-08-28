"""
monitor.py - System Metrics Collection Module

Collects real-time CPU, memory, and disk usage using psutil.
Supports a simulation mode that generates high CPU values to
trigger anomaly detection without stressing the actual machine.

Falls back to simulated metrics when psutil is unavailable
(e.g. on serverless platforms like Vercel).
"""

import random

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


def get_metrics(simulate_spike: bool = False) -> dict:
    """
    Collect current system metrics.

    Args:
        simulate_spike: If True, randomly generate high CPU values (85–95%)
                        to demonstrate anomaly detection.

    Returns:
        dict with keys: cpu, memory, disk (all as float percentages)
    """
    if _HAS_PSUTIL:
        # CPU usage over a short interval for accuracy
        cpu = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory().percent
        disk = psutil.disk_usage("/").percent
    else:
        # Fallback for environments without psutil (e.g. Vercel serverless)
        cpu = random.uniform(30, 70)
        memory = random.uniform(40, 75)
        disk = random.uniform(40, 60)

    if simulate_spike:
        # Override CPU with a simulated spike value
        cpu = random.uniform(85, 95)

    return {
        "cpu": round(cpu, 1),
        "memory": round(memory, 1),
        "disk": round(disk, 1),
    }
