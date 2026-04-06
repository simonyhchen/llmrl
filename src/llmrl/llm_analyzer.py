"""
LLM Analyzer: 讀取 Ray 訓練日誌，使用 LLM 分析並建議 reward 調整。

使用方式:
    from src.llmrl.llm_analyzer import LLMAnalyzer

    analyzer = LLMAnalyzer()
    suggestion = analyzer.analyze("~/ray_results/train_45nm_ngspice")
    print(suggestion)
"""

import json
import glob
import os
from src.llmrl.llm_client import get_llm_client
import src.llmrl.config as config

_ANALYSIS_PROMPT = """\
You are an expert in reinforcement learning for analog circuit design optimization.
Analyze the following RL training metrics from AutoCkt (PPO training an opamp circuit):

{metrics_summary}

Based on these metrics, identify any problems with the current reward function and suggest specific, actionable improvements.
Focus on:
1. Convergence speed (is it converging fast enough?)
2. Reward magnitude (is the reward scale appropriate?)
3. Exploration vs exploitation (is the agent exploring enough?)
4. Spec satisfaction rate (are specs being met?)

Provide concrete suggestions for modifying the reward function to improve performance.
Keep your response concise and actionable (3-5 bullet points).
"""


class LLMAnalyzer:
    """
    讀取 Ray 的訓練結果 JSON 日誌，用 LLM 分析並建議 reward 調整。
    """

    def __init__(self, provider: str = None):
        self.client = get_llm_client(provider or config.LLM_PROVIDER)

    def parse_ray_logs(self, log_dir: str) -> list:
        """
        從 Ray 結果目錄中讀取所有 result.json 並解析。

        Args:
            log_dir: Ray 結果目錄，例如 ~/ray_results/train_45nm_ngspice

        Returns:
            list of dict，每個 dict 是一個訓練 iteration 的指標。
        """
        log_dir = os.path.expanduser(log_dir)
        result_files = glob.glob(os.path.join(log_dir, "**/result.json"), recursive=True)

        if not result_files:
            print(f"[LLMAnalyzer] No result.json found in: {log_dir}")
            return []

        # 使用最新的 result.json
        result_file = sorted(result_files)[-1]
        print(f"[LLMAnalyzer] Parsing: {result_file}")

        records = []
        with open(result_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        return records

    def summarize_metrics(self, records: list) -> dict:
        """
        從訓練記錄中提取關鍵指標摘要。

        Returns:
            dict with summary statistics.
        """
        if not records:
            return {}

        rewards = [r.get("episode_reward_mean", None) for r in records if r.get("episode_reward_mean") is not None]
        iters = [r.get("training_iteration", 0) for r in records]
        times = [r.get("time_total_s", 0) for r in records]
        timesteps = [r.get("timesteps_total", 0) for r in records]

        return {
            "total_iterations": max(iters) if iters else 0,
            "total_timesteps": max(timesteps) if timesteps else 0,
            "total_time_s": max(times) if times else 0,
            "reward_initial": rewards[0] if rewards else None,
            "reward_final": rewards[-1] if rewards else None,
            "reward_best": max(rewards) if rewards else None,
            "reward_worst": min(rewards) if rewards else None,
            "reward_improvement": (rewards[-1] - rewards[0]) if len(rewards) >= 2 else None,
            "converged": rewards[-1] >= 10 if rewards else False,
        }

    def analyze(self, log_dir: str) -> str:
        """
        分析訓練日誌並用 LLM 生成改進建議。

        Args:
            log_dir: Ray 結果目錄路徑。

        Returns:
            LLM 的分析建議字串。
        """
        records = self.parse_ray_logs(log_dir)
        if not records:
            return "[LLMAnalyzer] No training data found to analyze."

        summary = self.summarize_metrics(records)

        metrics_summary = f"""
Training Summary:
- Total iterations: {summary['total_iterations']}
- Total timesteps: {summary['total_timesteps']}
- Total training time: {summary['total_time_s']:.1f}s
- Initial reward: {summary['reward_initial']:.4f}
- Final reward: {summary['reward_final']:.4f}
- Best reward: {summary['reward_best']:.4f}
- Reward improvement: {summary['reward_improvement']:.4f}
- Converged (reward >= 10): {summary['converged']}

Last 5 iterations reward trend:
{[r.get('episode_reward_mean') for r in records[-5:]]}
"""

        prompt = _ANALYSIS_PROMPT.format(metrics_summary=metrics_summary)
        print(f"[LLMAnalyzer] Sending to LLM ({config.LLM_PROVIDER}) for analysis...")

        try:
            suggestion = self.client.generate_response(prompt)
            return suggestion
        except Exception as e:
            return f"[LLMAnalyzer] LLM analysis failed: {e}"

    def analyze_and_suggest_description(self, log_dir: str) -> str:
        """
        分析訓練日誌後，回傳可直接傳給 RewardGenerator 的自然語言描述。

        Returns:
            str: 可傳給 RewardGenerator.generate() 的描述字串。
        """
        analysis = self.analyze(log_dir)

        follow_up_prompt = f"""
Based on this analysis of an RL training run:
{analysis}

Write a single concise sentence describing what the new reward function should do differently.
This description will be used to generate a new Python reward function.
Output only the description sentence, nothing else.
"""
        try:
            description = self.client.generate_response(follow_up_prompt)
            return description.strip()
        except Exception as e:
            return "Reward all specs satisfaction equally, penalize proportionally to spec deviation magnitude."
