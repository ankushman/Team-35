"""
DQN Training Script (Week 2)
==============================
Trains a Deep Q-Network agent on the AIOps custom environment
using Stable-Baselines3.

Usage
-----
    python src/rl_agent/train_agent.py

The trained model is saved to:
    models/dqn_aiops_model.zip
"""

import sys
import os

# ── Project path setup ────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from src.rl_agent.environment import AIOpsEnv


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TOTAL_TIMESTEPS = 20_000                         # Training budget
MODEL_SAVE_DIR  = os.path.join(PROJECT_ROOT, "models")
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "dqn_aiops_model")


# ---------------------------------------------------------------------------
# Custom callback — prints training progress every N steps
# ---------------------------------------------------------------------------
class ProgressCallback(BaseCallback):
    """Print a one-liner every `print_freq` timesteps."""

    def __init__(self, print_freq: int = 2000, verbose: int = 0):
        super().__init__(verbose)
        self.print_freq = print_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.print_freq == 0:
            # Retrieve the most recent episode reward from the Monitor
            infos = self.locals.get("infos", [{}])
            ep_rew = infos[0].get("episode", {}).get("r", "N/A")
            print(
                f"  [Training] Step {self.num_timesteps:>6} / "
                f"{TOTAL_TIMESTEPS}  |  Last episode reward: {ep_rew}"
            )
        return True


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------
def train() -> None:
    """Instantiate environment, train DQN agent, save model."""

    print("=" * 60)
    print("  AIOps RL Agent — DQN Training (Week 2)")
    print("=" * 60)
    print()

    # ── 1. Create & wrap environment ──────────────────────────────────
    env = AIOpsEnv()
    env = Monitor(env)          # Wraps env to log episode stats
    print("[INFO] Environment created.")

    # ── 2. Initialise DQN agent ───────────────────────────────────────
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-3,
        buffer_size=50_000,
        learning_starts=1000,
        batch_size=64,
        gamma=0.99,
        target_update_interval=500,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        verbose=0,
        seed=42,
    )
    print("[INFO] DQN agent initialised.")
    print(f"[INFO] Training for {TOTAL_TIMESTEPS:,} timesteps …")
    print()

    # ── 3. Train ──────────────────────────────────────────────────────
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=ProgressCallback(print_freq=2000),
    )
    print()

    # ── 4. Save model ─────────────────────────────────────────────────
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    print(f"[INFO] Model saved -> {MODEL_SAVE_PATH}.zip")
    print("[INFO] Training complete.")

    env.close()


if __name__ == "__main__":
    train()
