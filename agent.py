"""
agent.py - AI Decision Agent Module

A rule-based decision agent that selects the most appropriate
corrective action based on current CPU and memory metrics.

In a production AIOps system this could be replaced with a
trained reinforcement-learning or ML model.
"""

# Available actions
ACTION_RESTART_SERVICE = "restart_service"
ACTION_CLEAR_CACHE = "clear_cache"
ACTION_DO_NOTHING = "do_nothing"


def decide_action(cpu: float, memory: float) -> str:
    """
    Choose a corrective action based on current system metrics.

    Decision logic:
        CPU > 90% → restart_service  (most severe intervention)
        CPU > 80% → clear_cache      (moderate intervention)
        Memory > 85% → clear_cache   (free up memory)
        Otherwise  → do_nothing      (system is healthy)

    Args:
        cpu:    CPU usage percentage (0–100).
        memory: Memory usage percentage (0–100).

    Returns:
        One of the ACTION_* string constants defined above.
    """
    if cpu > 90:
        return ACTION_RESTART_SERVICE
    if cpu > 80:
        return ACTION_CLEAR_CACHE
    if memory > 85:
        return ACTION_CLEAR_CACHE
    return ACTION_DO_NOTHING
