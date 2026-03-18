"""
AIOps RL Environment (Week 2)
==============================
Custom Gymnasium environment that simulates an IT infrastructure
monitoring scenario.  The RL agent observes system metrics and an
anomaly flag, then chooses a corrective action.  The environment
applies the action, updates the metrics, and returns a shaped reward.

State space  (5 continuous values, normalised 0–1):
    [cpu, memory, latency, disk_io, anomaly_flag]

Action space (5 discrete actions):
    0 → Do nothing
    1 → Restart service
    2 → Scale up resources
    3 → Clear cache
    4 → Send alert
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ── We import Week-1 modules to reuse the simulator & detector ────────────
import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.simulator.metrics_simulator import generate_metrics
from src.anomaly_detection.detector import AnomalyDetector


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ACTION_NAMES = {
    0: "Do Nothing",
    1: "Restart Service",
    2: "Scale Up Resources",
    3: "Clear Cache",
    4: "Send Alert",
}

# Normalisation limits (used to scale raw values to 0-1)
METRIC_MAX = {
    "cpu":     100.0,
    "memory":  100.0,
    "latency": 500.0,
    "disk_io": 100.0,
}

MAX_STEPS_PER_EPISODE = 200          # Episode ends after this many steps

# Thresholds that define "system failure" (extreme metrics)
FAILURE_THRESHOLDS = {
    "cpu":     98.0,
    "memory":  98.0,
    "latency": 480.0,
    "disk_io": 95.0,
}


class AIOpsEnv(gym.Env):
    """Gymnasium environment for AIOps autonomous remediation."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        super().__init__()

        # ── Spaces ────────────────────────────────────────────────────
        #  5 continuous observations normalised between 0 and 1
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(5,), dtype=np.float32,
        )
        #  5 discrete actions
        self.action_space = spaces.Discrete(5)

        # ── Internal state ────────────────────────────────────────────
        self.detector = AnomalyDetector()
        self.detector.train(n_samples=1000)

        self.current_metrics: dict = {}
        self.anomaly_flag: int = 0       # 0 = Normal, 1 = Anomaly
        self.steps: int = 0
        self.render_mode = render_mode

    # ------------------------------------------------------------------
    # Helper — convert raw metrics dict → numpy observation
    # ------------------------------------------------------------------
    def _get_observation(self) -> np.ndarray:
        """Return normalised observation vector."""
        obs = np.array([
            self.current_metrics["cpu"]     / METRIC_MAX["cpu"],
            self.current_metrics["memory"]  / METRIC_MAX["memory"],
            self.current_metrics["latency"] / METRIC_MAX["latency"],
            self.current_metrics["disk_io"] / METRIC_MAX["disk_io"],
            float(self.anomaly_flag),
        ], dtype=np.float32)
        return np.clip(obs, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Helper — detect anomaly on current metrics
    # ------------------------------------------------------------------
    def _run_anomaly_detection(self) -> int:
        """Return 1 if anomaly, 0 if normal."""
        status = self.detector.detect(self.current_metrics)
        return 1 if status == "Anomaly Detected" else 0

    # ------------------------------------------------------------------
    # Helper — check for system failure
    # ------------------------------------------------------------------
    def _is_system_failure(self) -> bool:
        """Return True when metrics are critically extreme."""
        return (
            self.current_metrics["cpu"]     >= FAILURE_THRESHOLDS["cpu"]
            or self.current_metrics["memory"]  >= FAILURE_THRESHOLDS["memory"]
            or self.current_metrics["latency"] >= FAILURE_THRESHOLDS["latency"]
            or self.current_metrics["disk_io"] >= FAILURE_THRESHOLDS["disk_io"]
        )

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        """Reset environment to a fresh starting state."""
        super().reset(seed=seed)

        self.steps = 0
        self.current_metrics = generate_metrics()
        self.anomaly_flag = self._run_anomaly_detection()

        return self._get_observation(), {}

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------
    def step(self, action: int):
        """
        Execute one time-step.

        Parameters
        ----------
        action : int   (0-4)

        Returns
        -------
        observation, reward, terminated, truncated, info
        """
        self.steps += 1
        prev_anomaly = self.anomaly_flag

        # ── Apply action effects on current metrics ───────────────────
        self._apply_action(action)

        # ── Generate next sample & detect anomaly ─────────────────────
        next_sample = generate_metrics()

        # Blend: 70 % new sample + 30 % post-action residual
        # This makes the action influence persist into the next state.
        for key in next_sample:
            self.current_metrics[key] = round(
                0.7 * next_sample[key] + 0.3 * self.current_metrics[key], 1
            )

        self.anomaly_flag = self._run_anomaly_detection()

        # ── Compute reward ────────────────────────────────────────────
        reward = self._compute_reward(action, prev_anomaly)

        # ── Termination conditions ────────────────────────────────────
        terminated = self._is_system_failure()
        truncated  = self.steps >= MAX_STEPS_PER_EPISODE

        if terminated:
            reward = -20.0       # System failure penalty

        info = {
            "metrics":   dict(self.current_metrics),
            "anomaly":   self.anomaly_flag,
            "action":    ACTION_NAMES[action],
            "step":      self.steps,
        }

        if self.render_mode == "human":
            self.render()

        return self._get_observation(), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # _apply_action — simulate the effect of each remediation action
    # ------------------------------------------------------------------
    def _apply_action(self, action: int) -> None:
        """Adjust current metrics to reflect the chosen action."""
        m = self.current_metrics

        if action == 0:
            # Do nothing
            pass

        elif action == 1:
            # Restart service → big drop in CPU & latency
            m["cpu"]     = max(5.0,  m["cpu"]     * 0.5)
            m["latency"] = max(5.0,  m["latency"] * 0.4)

        elif action == 2:
            # Scale up resources → reduces memory & disk pressure
            m["memory"]  = max(10.0, m["memory"]  * 0.5)
            m["disk_io"] = max(5.0,  m["disk_io"] * 0.6)

        elif action == 3:
            # Clear cache → reduces latency & moderate CPU drop
            m["latency"] = max(5.0,  m["latency"] * 0.4)
            m["cpu"]     = max(5.0,  m["cpu"]     * 0.8)

        elif action == 4:
            # Send alert → small passive improvement (ops team notified)
            m["cpu"]     = max(5.0,  m["cpu"]     * 0.9)
            m["memory"]  = max(10.0, m["memory"]  * 0.9)

        # Round for neatness
        for key in m:
            m[key] = round(m[key], 1)

    # ------------------------------------------------------------------
    # _compute_reward
    # ------------------------------------------------------------------
    def _compute_reward(self, action: int, prev_anomaly: int) -> float:
        """
        Reward shaping:
            +10  → system stable (no anomaly before or after)
            +5   → anomaly reduced (was 1, now 0)
            -5   → unnecessary action on healthy system
            -10  → anomaly persists despite action
        """
        curr = self.anomaly_flag

        if prev_anomaly == 0 and curr == 0:
            # System was healthy
            if action == 0:
                return 10.0    # Correctly did nothing
            else:
                return -5.0    # Unnecessary action

        elif prev_anomaly == 1 and curr == 0:
            # Anomaly resolved
            return 5.0

        elif prev_anomaly == 1 and curr == 1:
            # Anomaly persists
            if action == 0:
                return -10.0   # Did nothing during anomaly
            else:
                return -10.0   # Action didn't help

        else:
            # prev_anomaly == 0 and curr == 1 → new anomaly appeared
            return -10.0

    # ------------------------------------------------------------------
    # render (human-readable console output)
    # ------------------------------------------------------------------
    def render(self):
        m = self.current_metrics
        flag = "ANOMALY" if self.anomaly_flag else "NORMAL"
        print(
            f"  Step {self.steps:>3} | "
            f"CPU {m['cpu']:>5} | MEM {m['memory']:>5} | "
            f"LAT {m['latency']:>6} | DISK {m['disk_io']:>5} | "
            f"{flag}"
        )


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    env = AIOpsEnv(render_mode="human")
    obs, _ = env.reset()
    print(f"Initial observation: {obs}")
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  Action: {ACTION_NAMES[action]:<20} Reward: {reward:>6.1f}")
        if terminated or truncated:
            break
    env.close()
