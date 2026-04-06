"""Train baseline or LLM-enhanced PPO and save unified metrics.jsonl.

This script is Python 3.11+ oriented. It supports two modes:
1) Baseline RL: no LLM reward (`--use_default_reward`)
2) LLM-enhanced RL: generate reward code from natural language

Outputs are written to:
    experiments/<experiment_name>/metrics.jsonl
    experiments/<experiment_name>/summary.json
"""

import argparse
import os
import sys

# ── 確保 project root 和 AutoCkt 在 path 上 ───────────────────────────────────
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
_AUTOCKT_DIR = os.path.join(_PROJECT_ROOT, "autockt", "AutoCkt")

for p in [_PROJECT_ROOT, _AUTOCKT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

os.chdir(_AUTOCKT_DIR)
# ─────────────────────────────────────────────────────────────────────────────

import json
import time

import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig

from src.llmrl.modified_env import LLMEnhancedTwoStageAmp
from src.llmrl.reward_generator import RewardGenerator

# ── CLI Arguments ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train AutoCkt RL with or without LLM-generated reward.")
parser.add_argument(
    "--reward_description", "-rd",
    type=str,
    default="Reward proportionally to how close each spec is to the goal, penalize specs that are far from target more aggressively than the original.",
    help="Natural language description of the desired reward function.",
)
parser.add_argument(
    "--max_iters", "-mi",
    type=int,
    default=100,
    help="Maximum training iterations (default: 100).",
)
parser.add_argument(
    "--use_default_reward",
    action="store_true",
    help="Skip LLM and use the original AutoCkt reward function.",
)
parser.add_argument(
    "--reward_code_path",
    type=str,
    default=None,
    help="Path to an existing reward .py file to use (skips LLM generation).",
)
parser.add_argument(
    "--experiment_name", "-en",
    type=str,
    default="train_45nm_ngspice_llm",
    help="Experiment name (saved under experiments/<name>).",
)
parser.add_argument(
    "--checkpoint_dir", "-cpd",
    type=str,
    default=None,
    help="Path to checkpoint to restore from.",
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=2,
    help="Number of RLlib workers (default: 2 for local stability).",
)
args = parser.parse_args()

# ── Step 1: Generate or load reward function ──────────────────────────────────
reward_code_path = None

if not args.use_default_reward:
    if args.reward_code_path and os.path.isfile(args.reward_code_path):
        reward_code_path = args.reward_code_path
        print(f"[train_with_llm] Using existing reward file: {reward_code_path}")
    else:
        print(f"[train_with_llm] Generating reward function from description:")
        print(f"  \"{args.reward_description}\"")

        gen = RewardGenerator()
        code = gen.generate(args.reward_description)

        # 儲存生成的 reward 代碼，方便審查與 Ray worker 讀取
        reward_save_dir = os.path.join(_PROJECT_ROOT, "ray_tmp")
        os.makedirs(reward_save_dir, exist_ok=True)
        reward_code_path = os.path.join(reward_save_dir, "llm_reward.py")

        with open(reward_code_path, "w") as f:
            f.write(code)

        print(f"[train_with_llm] Reward code saved to: {reward_code_path}")
        print("[train_with_llm] Generated reward code:")
        print("-" * 60)
        print(code)
        print("-" * 60)

# ── Step 2: Register environment ──────────────────────────────────────────────
env_config = {
    "generalize": True,
    "run_valid": False,
    "max_steps": 30,
}
if reward_code_path:
    env_config["reward_code_path"] = reward_code_path

register_env("llm-opamp-v0", lambda cfg: LLMEnhancedTwoStageAmp(cfg))

# ── Step 3: Configure and run training ───────────────────────────────────────
ray.init(ignore_reinit_error=True)

run_dir = os.path.join(_PROJECT_ROOT, "experiments", args.experiment_name)
os.makedirs(run_dir, exist_ok=True)
metrics_path = os.path.join(run_dir, "metrics.jsonl")

ppo_config = (
    PPOConfig()
    .environment("llm-opamp-v0", env_config=env_config, disable_env_checking=True)
    .framework("torch")
    .env_runners(num_env_runners=args.num_workers, batch_mode="complete_episodes")
    .training(train_batch_size=1200, model={"fcnet_hiddens": [64, 64]})
    .resources(num_gpus=0)
)

algo = ppo_config.build()

if args.checkpoint_dir:
    print(f"[train_with_llm] Restoring from checkpoint: {args.checkpoint_dir}")
    algo.restore(args.checkpoint_dir)

best_reward = -1e9
best_ckpt = None
start_t = time.time()

with open(metrics_path, "w") as mf:
    for i in range(1, args.max_iters + 1):
        result = algo.train()

        if i == 1:
            with open(os.path.join(run_dir, "first_result_debug.json"), "w") as df:
                json.dump(result, df, indent=2, default=str)

        env_metrics = result.get("env_runners", {})
        reward_mean = result.get("episode_reward_mean")
        if reward_mean is None:
            reward_mean = env_metrics.get("episode_return_mean", env_metrics.get("episode_reward_mean"))

        reward_max = result.get("episode_reward_max")
        if reward_max is None:
            reward_max = env_metrics.get("episode_return_max", env_metrics.get("episode_reward_max"))

        timesteps_total = result.get("timesteps_total", result.get("num_env_steps_sampled_lifetime", 0))
        time_total_s = result.get("time_total_s", time.time() - start_t)

        row = {
            "training_iteration": i,
            "episode_reward_mean": reward_mean,
            "episode_reward_max": reward_max,
            "timesteps_total": timesteps_total,
            "time_total_s": time_total_s,
        }
        mf.write(json.dumps(row) + "\n")
        mf.flush()

        print(
            f"[train_with_llm] iter={i:03d} reward_mean={reward_mean} "
            f"timesteps={timesteps_total}"
        )

        if reward_mean is not None and reward_mean > best_reward:
            best_reward = reward_mean
            best_ckpt = str(algo.save(run_dir))

        if reward_mean is not None and reward_mean >= -0.02:
            print("[train_with_llm] Early stop: reached reward threshold -0.02")
            break

summary = {
    "experiment_name": args.experiment_name,
    "use_default_reward": args.use_default_reward,
    "reward_code_path": reward_code_path,
    "max_iters": args.max_iters,
    "best_reward_mean": best_reward,
    "best_checkpoint": best_ckpt,
    "metrics_path": metrics_path,
}

with open(os.path.join(run_dir, "summary.json"), "w") as sf:
    json.dump(summary, sf, indent=2)

print(f"[train_with_llm] Training complete. Summary saved in: {run_dir}")

algo.stop()
ray.shutdown()
