"""
Automated Comparison Script: baseline RL (no LLM) vs LLM-enhanced RL.

Workflow:
1. Optionally run baseline and LLM-enhanced training.
2. Parse unified metrics.jsonl files produced by train_with_llm.py.
3. Generate comparison chart and summary report.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
_AUTOCKT_DIR = os.path.join(_PROJECT_ROOT, "autockt", "AutoCkt")
_RESULTS_DIR = os.path.join(_PROJECT_ROOT, "comparison_results")

parser = argparse.ArgumentParser(description="Compare baseline RL vs LLM-enhanced RL.")
parser.add_argument("--max_iters", "-mi", type=int, default=20, help="Max iterations for each run.")
parser.add_argument(
    "--reward_description", "-rd", type=str,
    default="Reward proportionally to spec satisfaction and penalize large deviations more strongly.",
    help="Natural language description for LLM reward generation.",
)
parser.add_argument("--skip_training", action="store_true", help="Skip training and only compare existing logs.")
parser.add_argument(
    "--original_log",
    type=str,
    default=os.path.join(_PROJECT_ROOT, "experiments", "train_45nm_ngspice_baseline", "metrics.jsonl"),
    help="Path to baseline metrics.jsonl (or its parent directory).",
)
parser.add_argument(
    "--llm_log",
    type=str,
    default=os.path.join(_PROJECT_ROOT, "experiments", "train_45nm_ngspice_llm", "metrics.jsonl"),
    help="Path to LLM metrics.jsonl (or its parent directory).",
)
parser.add_argument("--output_dir", type=str, default=None, help="Output directory for report/chart.")
args = parser.parse_args()


def resolve_metrics_path(path: str):
    p = os.path.expanduser(path)
    if os.path.isfile(p):
        return p
    cand = os.path.join(p, "metrics.jsonl")
    if os.path.isfile(cand):
        return cand
    return None


def parse_metrics_jsonl(path: str):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def extract_series(rows):
    return {
        "iterations": [r.get("training_iteration", i + 1) for i, r in enumerate(rows)],
        "reward_mean": [r.get("episode_reward_mean") for r in rows],
        "timesteps": [r.get("timesteps_total") for r in rows],
        "time_s": [r.get("time_total_s") for r in rows],
    }


def compute_summary(rows, label):
    rewards = [r.get("episode_reward_mean") for r in rows if r.get("episode_reward_mean") is not None]
    if not rewards:
        return {"label": label, "error": "No reward data found"}

    converged_at = None
    for r in rows:
        if r.get("episode_reward_mean", -999) >= -0.02:
            converged_at = r.get("training_iteration")
            break

    return {
        "label": label,
        "total_iterations": len(rows),
        "total_timesteps": rows[-1].get("timesteps_total", 0),
        "total_time_s": rows[-1].get("time_total_s", 0),
        "reward_initial": rewards[0],
        "reward_final": rewards[-1],
        "reward_best": max(rewards),
        "reward_improvement": rewards[-1] - rewards[0],
        "converged": rewards[-1] >= -0.02,
        "converged_at_iter": converged_at,
    }


def run_training(script_args, label):
    cmd = [sys.executable] + script_args
    print("\n" + "=" * 60)
    print(f"[compare] Starting: {label}")
    print(f"[compare] Command: {' '.join(cmd)}")
    print("=" * 60 + "\n")

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{_AUTOCKT_DIR}:{_PROJECT_ROOT}:{env.get('PYTHONPATH', '')}"

    result = subprocess.run(cmd, cwd=_PROJECT_ROOT, env=env)
    return result.returncode


def generate_charts(base_series, llm_series, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("AutoCkt RL: Baseline vs LLM-Enhanced", fontsize=14, fontweight="bold")

    ax1 = axes[0]
    ax1.plot(base_series["iterations"], base_series["reward_mean"], label="Baseline (No LLM)", color="steelblue", linewidth=2)
    ax1.plot(llm_series["iterations"], llm_series["reward_mean"], label="LLM-Enhanced", color="darkorange", linewidth=2, linestyle="--")
    ax1.axhline(y=-0.02, color="green", linestyle=":", linewidth=1.5, label="Threshold")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Episode Reward Mean")
    ax1.set_title("Reward vs Iteration")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(base_series["timesteps"], base_series["reward_mean"], label="Baseline (No LLM)", color="steelblue", linewidth=2)
    ax2.plot(llm_series["timesteps"], llm_series["reward_mean"], label="LLM-Enhanced", color="darkorange", linewidth=2, linestyle="--")
    ax2.axhline(y=-0.02, color="green", linestyle=":", linewidth=1.5, label="Threshold")
    ax2.set_xlabel("Timesteps")
    ax2.set_ylabel("Episode Reward Mean")
    ax2.set_title("Reward vs Timesteps")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "comparison_chart.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    return chart_path


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join(_RESULTS_DIR, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    trainer_script = os.path.join(_PROJECT_ROOT, "src", "llmrl", "train_with_llm.py")

    if not args.skip_training:
        rc = run_training([
            trainer_script,
            "--experiment_name", "train_45nm_ngspice_baseline",
            "--max_iters", str(args.max_iters),
            "--use_default_reward",
        ], "Baseline (No LLM)")
        if rc != 0:
            print(f"[compare] Warning: baseline run exited with code {rc}")

        rc = run_training([
            trainer_script,
            "--experiment_name", "train_45nm_ngspice_llm",
            "--max_iters", str(args.max_iters),
            "--reward_description", args.reward_description,
        ], "LLM-Enhanced")
        if rc != 0:
            print(f"[compare] Warning: LLM run exited with code {rc}")

    base_path = resolve_metrics_path(args.original_log)
    llm_path = resolve_metrics_path(args.llm_log)

    if not base_path:
        print(f"[compare] ERROR: metrics.jsonl not found: {args.original_log}")
        sys.exit(1)
    if not llm_path:
        print(f"[compare] ERROR: metrics.jsonl not found: {args.llm_log}")
        sys.exit(1)

    base_rows = parse_metrics_jsonl(base_path)
    llm_rows = parse_metrics_jsonl(llm_path)

    base_series = extract_series(base_rows)
    llm_series = extract_series(llm_rows)

    base_summary = compute_summary(base_rows, "Baseline (No LLM)")
    llm_summary = compute_summary(llm_rows, "LLM-Enhanced")

    report = {
        "timestamp": timestamp,
        "config": {
            "max_iters": args.max_iters,
            "reward_description": args.reward_description,
            "python": sys.version,
        },
        "baseline": base_summary,
        "llm_enhanced": llm_summary,
        "comparison": {
            "final_reward_delta": llm_summary.get("reward_final", 0) - base_summary.get("reward_final", 0),
            "llm_converges_faster": (
                llm_summary.get("converged_at_iter") is not None
                and base_summary.get("converged_at_iter") is not None
                and llm_summary["converged_at_iter"] < base_summary["converged_at_iter"]
            ),
        },
    }

    report_path = os.path.join(output_dir, "comparison_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    chart_path = generate_charts(base_series, llm_series, output_dir)

    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"Baseline final reward: {base_summary.get('reward_final')}")
    print(f"LLM final reward:      {llm_summary.get('reward_final')}")
    print(f"Final reward delta:    {report['comparison']['final_reward_delta']}")
    print(f"LLM converges faster:  {report['comparison']['llm_converges_faster']}")
    print(f"Report: {report_path}")
    print(f"Chart:  {chart_path}")


if __name__ == "__main__":
    main()
