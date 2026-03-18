"""
AIOps RL Agent — Main Entry Point (Week 2)
===========================================
Full pipeline:
    1. Train the anomaly detector on synthetic normal data.
    2. Load the trained RL model (or fall back to random actions).
    3. Run the loop:  Simulate → Detect → RL Agent picks action → Show results.
    4. Print a summary at the end.
"""

import sys
import os

# ---------------------------------------------------------------------------
# Make sure the project root is on the Python path
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.simulator.metrics_simulator import generate_metrics
from src.anomaly_detection.detector import AnomalyDetector
from src.utils.logger import log_metrics
from src.rl_agent.environment import AIOpsEnv, ACTION_NAMES

# Try importing Stable-Baselines3 (might not be installed yet)
try:
    from stable_baselines3 import DQN
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NUM_ITERATIONS = 30
LOG_FILE       = os.path.join(PROJECT_ROOT, "data", "metrics_log.csv")
MODEL_PATH     = os.path.join(PROJECT_ROOT, "models", "dqn_aiops_model.zip")


def main() -> None:
    """Run the AIOps monitoring + RL remediation pipeline."""

    print("=" * 65)
    print("  AIOps RL Agent — Week 2 Demo")
    print("=" * 65)
    print()

    # ── Step 1: Create the RL environment (also trains detector) ──────
    env = AIOpsEnv()
    print()

    # ── Step 2: Load trained RL model (if available) ──────────────────
    model = None
    if SB3_AVAILABLE and os.path.isfile(MODEL_PATH):
        model = DQN.load(MODEL_PATH)
        print(f"[INFO] Loaded trained DQN model from {MODEL_PATH}")
    else:
        if not SB3_AVAILABLE:
            print("[WARN] stable-baselines3 not installed — using random actions.")
        else:
            print("[WARN] No trained model found — using random actions.")
        print("       Run  python src/rl_agent/train_agent.py  to train first.")
    print()

    # ── Step 3: Run the pipeline loop ─────────────────────────────────
    header = (
        f"{'Step':>4} | {'CPU':>5} | {'MEM':>5} | {'LAT':>6} | "
        f"{'DISK':>5} | {'Status':<17} | {'Action':<20} | {'Reward':>7}"
    )
    print(header)
    print("-" * len(header))

    obs, _ = env.reset()
    total_reward    = 0.0
    anomalies_seen  = 0
    actions_counter = {i: 0 for i in range(5)}

    for step in range(1, NUM_ITERATIONS + 1):
        # Choose action
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        actions_counter[action] += 1
        m = info["metrics"]
        anomaly_str = "[!] Anomaly" if info["anomaly"] else "Normal"
        if info["anomaly"]:
            anomalies_seen += 1

        # Log to CSV (keeping Week 1 logging running)
        log_metrics(m, anomaly_str, filepath=LOG_FILE)

        # Print per-step summary
        print(
            f"{step:>4} | "
            f"{m['cpu']:>5} | {m['memory']:>5} | {m['latency']:>6} | "
            f"{m['disk_io']:>5} | {anomaly_str:<17} | "
            f"{ACTION_NAMES[action]:<20} | {reward:>+7.1f}"
        )

        if terminated:
            print("  *** System failure — episode ended ***")
            break

    # ── Step 4: Summary ───────────────────────────────────────────────
    print()
    print("=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print(f"  Steps completed     : {step}")
    print(f"  Total reward        : {total_reward:>+.1f}")
    print(f"  Anomalies detected  : {anomalies_seen}")
    print()
    for a_id, count in actions_counter.items():
        print(f"    {ACTION_NAMES[a_id]:<20} : {count}")
    print()
    print(f"[INFO] Metrics logged to {LOG_FILE}")
    print("[INFO] Done.")

    env.close()


if __name__ == "__main__":
    main()
