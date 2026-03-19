"""
action_log.py - Thread-safe in-memory action log

A single shared deque that both fixer.py (writer) and dashboard.py (reader)
use.  Importing this module from multiple files is safe because Python caches
modules — every import gets the same object.
"""

from collections import deque
import threading
from datetime import datetime

# Max entries kept in memory
MAX_ENTRIES = 50

_lock = threading.Lock()
_log: deque = deque(maxlen=MAX_ENTRIES)


def push(action: str, trigger: str, cpu: float, memory: float, result: str) -> None:
    """
    Record one corrective action execution.

    Args:
        action:  Action constant string (e.g. 'restart_service').
        trigger: Human-readable reason (e.g. 'CPU 92.1% > 90%').
        cpu:     CPU value at time of action.
        memory:  Memory value at time of action.
        result:  Outcome message (e.g. 'Service restarted successfully.').
    """
    entry = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "action": action,
        "trigger": trigger,
        "cpu": cpu,
        "memory": memory,
        "result": result,
    }
    with _lock:
        _log.appendleft(entry)   # newest first


def get_all() -> list:
    """Return a snapshot list of all logged entries (newest first)."""
    with _lock:
        return list(_log)


def get_latest() -> dict | None:
    """Return the most recent entry, or None if the log is empty."""
    with _lock:
        return _log[0] if _log else None
