"""
fixer.py - Self-Healing / Corrective Actions Module

Executes the action chosen by the AI decision agent.
In demo mode all actions are simulated with console messages
so the program is safe to run on any machine.

Every executed action is also recorded in the shared action_log
so the dashboard can display it in real time.
"""

import time
from agent import ACTION_RESTART_SERVICE, ACTION_CLEAR_CACHE, ACTION_DO_NOTHING
import action_log


def restart_service(cpu: float = 0.0, memory: float = 0.0) -> None:
    """
    Simulate restarting a system service to relieve high CPU load.
    In a real deployment this might call systemctl, supervisorctl,
    or a Kubernetes rollout restart.
    """
    print("[ACTION] Restarting service to reduce system load...")
    time.sleep(0.5)  # Simulate work
    result = "Service restarted successfully."
    print(f"[ACTION] {result}")

    trigger = f"CPU {cpu}% > 90% (critical threshold)"
    action_log.push(ACTION_RESTART_SERVICE, trigger, cpu, memory, result)


def clear_cache(cpu: float = 0.0, memory: float = 0.0) -> None:
    """
    Simulate clearing application or OS cache to free up resources.
    In a real deployment this might flush Redis, drop Linux page cache,
    or invoke an application-level cache-clear endpoint.
    """
    print("[ACTION] Clearing cache to free up memory and CPU...")
    time.sleep(0.3)  # Simulate work
    result = "Cache cleared successfully."
    print(f"[ACTION] {result}")

    if cpu > 80:
        trigger = f"CPU {cpu}% > 80% (elevated threshold)"
    else:
        trigger = f"Memory {memory}% > 85% (memory threshold)"
    action_log.push(ACTION_CLEAR_CACHE, trigger, cpu, memory, result)


def execute_action(action: str, cpu: float = 0.0, memory: float = 0.0) -> None:
    """
    Dispatch to the correct corrective-action function.

    Args:
        action: One of the ACTION_* constants from agent.py.
        cpu:    CPU value at decision time (forwarded to the log).
        memory: Memory value at decision time (forwarded to the log).
    """
    if action == ACTION_RESTART_SERVICE:
        restart_service(cpu=cpu, memory=memory)
    elif action == ACTION_CLEAR_CACHE:
        clear_cache(cpu=cpu, memory=memory)
    elif action == ACTION_DO_NOTHING:
        print("[ACTION] System is healthy — no action required.")
    else:
        print(f"[ACTION] Unknown action '{action}' — skipping.")
