"""
RL Agent Evaluation Script (Week 2)
====================================
Loads the trained DQN model and compares its performance against a
random agent over a fixed number of evaluation steps.

Usage
-----
    python src/rl_agent/evaluate.py

Output
------
    Per-step:  Metrics → Action → Reward
    Summary :  Total reward, anomalies resolved, avg. metrics
               for both trained and random agents.
"""

import sys
import os

# ── Project path setup ────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from stable_baselines3 import DQN
from src.rl_agent.environment import AIOpsEnv, ACTION_NAMES


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH  = os.path.join(PROJECT_ROOT, "models", "dqn_aiops_model.zip")
EVAL_STEPS  = 100            # Number of steps per evaluation run


# ---------------------------------------------------------------------------
# Run one evaluation episode
# ---------------------------------------------------------------------------
def run_episode(env: AIOpsEnv, model=None, label: str = "Agent",
                verbose: bool = True) -> dict:
    """
    Run `EVAL_STEPS` in the environment.

    Parameters
    ----------
    env   : AIOpsEnv instance
    model : trained SB3 model (None → random actions)
    label : display label
    verbose : print per-step info

    Returns
    -------
    dict with summary statistics
    """
    obs, _ = env.reset()
    total_reward = 0.0
    anomalies_seen = 0
    anomalies_resolved = 0
    actions_taken = {i: 0 for i in range(5)}
    cpu_sum = mem_sum = lat_sum = disk_sum = 0.0

    if verbose:
        print(f"\n{'-' * 70}")
        print(f"  {label} -- {EVAL_STEPS}-step evaluation")
        print(f"{'-' * 70}")

    for step in range(1, EVAL_STEPS + 1):
        # Choose action
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        actions_taken[action] += 1

        m = info["metrics"]
        cpu_sum  += m["cpu"]
        mem_sum  += m["memory"]
        lat_sum  += m["latency"]
        disk_sum += m["disk_io"]

        if info["anomaly"] == 1:
            anomalies_seen += 1
        # If previous anomaly resolved (reward == +5)
        if reward == 5.0:
            anomalies_resolved += 1

        if verbose:
            flag = "[!] ANOMALY" if info["anomaly"] else "    NORMAL "
            print(
                f"  Step {step:>3} | "
                f"CPU {m['cpu']:>5} | MEM {m['memory']:>5} | "
                f"LAT {m['latency']:>6} | DISK {m['disk_io']:>5} | "
                f"{flag} | Action: {ACTION_NAMES[action]:<20} | "
                f"Reward: {reward:>+6.1f}"
            )

        if terminated:
            if verbose:
                print("  *** System failure — episode terminated ***")
            break

    summary = {
        "label":              label,
        "steps":              step,
        "total_reward":       total_reward,
        "anomalies_seen":     anomalies_seen,
        "anomalies_resolved": anomalies_resolved,
        "avg_cpu":            round(cpu_sum  / step, 1),
        "avg_memory":         round(mem_sum  / step, 1),
        "avg_latency":        round(lat_sum  / step, 1),
        "avg_disk_io":        round(disk_sum / step, 1),
        "actions":            actions_taken,
    }
    return summary


# ---------------------------------------------------------------------------
# Print comparison table
# ---------------------------------------------------------------------------
def print_comparison(trained: dict, random: dict) -> None:
    """Pretty-print a side-by-side comparison."""

    print("\n" + "=" * 70)
    print("  EVALUATION SUMMARY — Trained DQN vs Random Agent")
    print("=" * 70)

    row = "{:<25} {:>18} {:>18}"
    print(row.format("Metric", "Trained DQN", "Random Agent"))
    print("-" * 62)
    print(row.format("Total Reward",
                     f"{trained['total_reward']:>+.1f}",
                     f"{random['total_reward']:>+.1f}"))
    print(row.format("Steps Completed",
                     str(trained["steps"]),
                     str(random["steps"])))
    print(row.format("Anomalies Seen",
                     str(trained["anomalies_seen"]),
                     str(random["anomalies_seen"])))
    print(row.format("Anomalies Resolved",
                     str(trained["anomalies_resolved"]),
                     str(random["anomalies_resolved"])))
    print(row.format("Avg CPU (%)",
                     str(trained["avg_cpu"]),
                     str(random["avg_cpu"])))
    print(row.format("Avg Memory (%)",
                     str(trained["avg_memory"]),
                     str(random["avg_memory"])))
    print(row.format("Avg Latency (ms)",
                     str(trained["avg_latency"]),
                     str(random["avg_latency"])))
    print(row.format("Avg Disk I/O (MB/s)",
                     str(trained["avg_disk_io"]),
                     str(random["avg_disk_io"])))

    print()
    for s in (trained, random):
        acts = ", ".join(
            f"{ACTION_NAMES[k]}: {v}" for k, v in s["actions"].items()
        )
        print(f"  {s['label']} actions -> {acts}")

    diff = trained["total_reward"] - random["total_reward"]
    print()
    if diff > 0:
        print(f"  [OK] Trained agent outperforms random by {diff:+.1f} reward points.")
    else:
        print(f"  [!!] Trained agent did not outperform random ({diff:+.1f})."
              "  Consider more training steps.")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("  AIOps RL Agent — Evaluation (Week 2)")
    print("=" * 60)

    # ── Load trained model ────────────────────────────────────────────
    if not os.path.isfile(MODEL_PATH):
        print(f"[ERROR] Model not found at {MODEL_PATH}")
        print("        Run  python src/rl_agent/train_agent.py  first.")
        sys.exit(1)

    model = DQN.load(MODEL_PATH)
    print(f"[INFO] Loaded model from {MODEL_PATH}")

    # ── Evaluate trained agent ────────────────────────────────────────
    env_trained = AIOpsEnv()
    trained_summary = run_episode(env_trained, model=model,
                                  label="Trained DQN", verbose=True)
    env_trained.close()

    # ── Evaluate random baseline ──────────────────────────────────────
    env_random = AIOpsEnv()
    random_summary = run_episode(env_random, model=None,
                                 label="Random Agent", verbose=True)
    env_random.close()

    # ── Comparison ────────────────────────────────────────────────────
    print_comparison(trained_summary, random_summary)


if __name__ == "__main__":
    main()
