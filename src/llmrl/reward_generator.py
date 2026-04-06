"""
Reward Generator: 基於自然語言描述，用LLM生成 reward function 代碼。

使用方式:
    from src.llmrl.reward_generator import RewardGenerator

    gen = RewardGenerator()

    # 1. 從自然語言生成 reward function 代碼字串
    code = gen.generate("獎勵所有規格滿足，並輕微懲罰功耗超過10%")

    # 2. 動態載入成可呼叫的 Python function
    reward_fn = gen.load(code)

    # 3. 在 RL 環境中使用
    reward = reward_fn(spec, goal_spec, specs_id)
"""

import types
from src.llmrl.llm_client import get_llm_client, validate_generated_code
import src.llmrl.config as config

# ─────────────────────────────────────────────────────────────────────────────
# 系統提示：告訴LLM要生成甚麼格式
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an expert in reinforcement learning for analog circuit design.
Your task is to generate a Python reward function for the AutoCkt RL framework.

The function MUST follow this exact signature:
    def reward(spec, goal_spec, specs_id):

Arguments:
- spec      : numpy array of current simulated circuit specs
- goal_spec : numpy array of ideal/target circuit specs
- specs_id  : list of spec names (e.g. ["gain_min", "ibias_max", "pm_min", "ugbw_min"])

Return value:
- A float reward. Return 10.0 if ALL specs are satisfied (done condition).
  Otherwise return a negative float (the more unsatisfied, the more negative).

Rules:
1. Do NOT import anything. All needed values are passed as arguments.
2. Use only basic Python and numpy-compatible operations on arrays.
3. The function name MUST be "reward".
4. Output ONLY the Python function code, no explanation, no markdown fences.

Background - original AutoCkt reward (for reference):
    def reward(spec, goal_spec, specs_id):
        goal_spec = [float(e) for e in goal_spec]
        norm_spec = (spec - goal_spec) / (goal_spec + spec)
        reward = 0.0
        for i, rel_spec in enumerate(norm_spec):
            if specs_id[i] == 'ibias_max':
                rel_spec = rel_spec * -1.0
            if rel_spec < 0:
                reward += rel_spec
        return reward if reward < -0.02 else 10.0
"""

_USER_PROMPT_TEMPLATE = """\
Generate a new reward function based on the following description:

\"{description}\"

Output only the Python function code.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Default fallback: 原版 AutoCkt reward（不需要LLM也能用）
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_REWARD_CODE = """\
def reward(spec, goal_spec, specs_id):
    goal_spec = [float(e) for e in goal_spec]
    norm_spec = (spec - goal_spec) / (goal_spec + spec)
    total_reward = 0.0
    for i, rel_spec in enumerate(norm_spec):
        if specs_id[i] == 'ibias_max':
            rel_spec = rel_spec * -1.0
        if rel_spec < 0:
            total_reward += rel_spec
    return total_reward if total_reward < -0.02 else 10.0
"""


class RewardGenerator:
    """
    使用LLM根據自然語言描述生成 reward function。
    如果LLM失敗或生成無效代碼，自動回退到原版 AutoCkt reward。
    """

    def __init__(self, provider: str = None):
        """
        Args:
            provider: LLM provider，預設使用 config.LLM_PROVIDER。
        """
        self.client = get_llm_client(provider or config.LLM_PROVIDER)

    def generate(self, description: str) -> str:
        """
        根據自然語言描述生成 reward function 的 Python 代碼字串。

        Args:
            description: 自然語言描述，例如「獎勵所有規格滿足，並輕微懲罰功耗超過10%」。

        Returns:
            合法的 Python 代碼字串（包含 def reward(...)）。
            如果生成失敗，回傳原版 AutoCkt reward 代碼。
        """
        prompt = _SYSTEM_PROMPT + "\n\n" + _USER_PROMPT_TEMPLATE.format(description=description)

        print(f"[RewardGenerator] Requesting LLM ({config.LLM_PROVIDER}) to generate reward function...")
        try:
            raw_code = self.client.generate_response(prompt)
            code = self._extract_code(raw_code)

            if validate_generated_code(code):
                # 確認函數簽名正確
                if "def reward(" in code:
                    print("[RewardGenerator] Successfully generated valid reward function.")
                    return code
                else:
                    print("[RewardGenerator] Warning: generated code missing 'def reward('. Using default.")
            else:
                print("[RewardGenerator] Warning: generated code has syntax errors. Using default.")

        except Exception as e:
            print(f"[RewardGenerator] LLM call failed: {e}. Using default reward function.")

        return _DEFAULT_REWARD_CODE

    def load(self, code: str):
        """
        將 Python 代碼字串動態載入為可呼叫的 function。

        Args:
            code: 包含 def reward(...) 的 Python 代碼字串。

        Returns:
            可呼叫的 reward function。

        Raises:
            ValueError: 如果代碼無法編譯或不含 reward function。
        """
        if not validate_generated_code(code):
            raise ValueError("Cannot load reward function: code has syntax errors.")

        namespace = {}
        exec(compile(code, "<llm_generated_reward>", "exec"), namespace)

        if "reward" not in namespace:
            raise ValueError("Cannot load reward function: 'reward' function not found in code.")

        fn = namespace["reward"]
        if not callable(fn):
            raise ValueError("'reward' is not callable.")

        return fn

    def generate_and_load(self, description: str):
        """
        一步完成：生成並載入 reward function。

        Args:
            description: 自然語言描述。

        Returns:
            (reward_fn, code) tuple:
                - reward_fn: 可呼叫的 reward function
                - code: 生成的 Python 代碼字串（方便紀錄和審查）
        """
        code = self.generate(description)
        reward_fn = self.load(code)
        return reward_fn, code

    def default_reward_fn(self):
        """
        回傳原版 AutoCkt reward function（不需要LLM）。

        Returns:
            可呼叫的原版 reward function。
        """
        return self.load(_DEFAULT_REWARD_CODE)

    @staticmethod
    def _extract_code(raw: str) -> str:
        """
        從LLM輸出中提取純 Python 代碼（去除 markdown fences 等）。
        """
        lines = raw.strip().splitlines()
        result = []
        in_block = False

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```python"):
                in_block = True
                continue
            if stripped.startswith("```") and in_block:
                in_block = False
                continue
            if in_block or not stripped.startswith("```"):
                result.append(line)

        return "\n".join(result).strip()
