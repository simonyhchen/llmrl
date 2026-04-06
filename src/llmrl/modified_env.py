"""
Modified RL Environment: 繼承 AutoCkt 的 TwoStageAmp，注入 LLM 生成的 reward function。

不修改 AutoCkt 原始碼，透過繼承方式替換 reward。
reward function 以代碼檔案的形式傳遞（而非 function 物件），
確保 Ray 多 worker 分散執行時可正確載入。

使用方式:
    env_config = {
        "generalize": True,
        "run_valid": False,
        "reward_code_path": "/path/to/llm_reward.py",  # 可選，不設則用原版
    }
    env = LLMEnhancedTwoStageAmp(env_config)
"""

import os
import sys
import gymnasium as gym
import numpy as np

# Compatibility shim for legacy AutoCkt yaml.load() calls under PyYAML>=6.
import yaml

if hasattr(yaml, "FullLoader"):
    _orig_yaml_load = yaml.load

    def _compat_yaml_load(stream, Loader=None, *args, **kwargs):
        if Loader is None:
            Loader = yaml.FullLoader
        return _orig_yaml_load(stream, Loader=Loader, *args, **kwargs)

    yaml.load = _compat_yaml_load

# ── 設定 AutoCkt 路徑並確保在正確目錄下 import ──────────────────────────────────
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
_AUTOCKT_DIR = os.path.join(_PROJECT_ROOT, "autockt", "AutoCkt")

if not os.path.isdir(_AUTOCKT_DIR):
    raise FileNotFoundError(
        f"AutoCkt not found at: {_AUTOCKT_DIR}\n"
        "Please run: git clone https://github.com/ksettaluri6/AutoCkt autockt/AutoCkt"
    )

if _AUTOCKT_DIR not in sys.path:
    sys.path.insert(0, _AUTOCKT_DIR)

# TwoStageAmp 在 class 層級使用 os.getcwd() 設定路徑，
# 必須在 import 前切換到 AutoCkt 目錄。
_original_cwd = os.getcwd()
os.chdir(_AUTOCKT_DIR)

from autockt.envs.ngspice_vanilla_opamp import TwoStageAmp  # noqa: E402

os.chdir(_original_cwd)
# ─────────────────────────────────────────────────────────────────────────────


class LLMEnhancedTwoStageAmp(TwoStageAmp, gym.Env):
    """
    TwoStageAmp with a pluggable LLM-generated reward function.

    env_config 額外支援的 key:
        reward_code_path (str): 指向 LLM 生成的 reward .py 檔案路徑。
                                 如果不提供，使用原版 AutoCkt reward。
    """

    def __init__(self, env_config):
        # 取出我們自定義的設定，避免傳給父類
        reward_code_path = env_config.pop("reward_code_path", None)
        self._llm_reward_fn = None

        if reward_code_path and os.path.isfile(reward_code_path):
            self._llm_reward_fn = self._load_reward_from_file(reward_code_path)
            print(f"[LLMEnhancedTwoStageAmp] Loaded LLM reward from: {reward_code_path}")
        else:
            print("[LLMEnhancedTwoStageAmp] No reward_code_path provided. Using original AutoCkt reward.")

        self.max_steps = int(env_config.get("max_steps", 30))

        # AutoCkt 需要在其目錄下執行
        os.chdir(_AUTOCKT_DIR)
        super().__init__(env_config)
        os.chdir(_original_cwd)

        # Convert legacy gym spaces from AutoCkt to gymnasium spaces.
        self.action_space = self._to_gymnasium_space(self.action_space)
        self.observation_space = self._to_gymnasium_space(self.observation_space)

    def reward(self, spec, goal_spec):
        """Override: 如果有 LLM reward 則使用，否則回退到原版。"""
        if self._llm_reward_fn is not None:
            try:
                return self._llm_reward_fn(spec, goal_spec, self.specs_id)
            except Exception as e:
                print(f"[LLMEnhancedTwoStageAmp] LLM reward error: {e}. Falling back to original.")
                self._llm_reward_fn = None

        return super().reward(spec, goal_spec)

    def reset(self, *, seed=None, options=None):
        """Gymnasium-compatible reset wrapper."""
        if seed is not None:
            try:
                import random
                random.seed(seed)
            except Exception:
                pass
        obs = super().reset()
        try:
            # Parent env tracks this counter as well; keep explicit reset for safety.
            self.env_steps = 0
        except Exception:
            pass
        return obs, {}

    def step(self, action):
        """Gymnasium-compatible step wrapper."""
        obs, reward, done, info = super().step(action)
        terminated = bool(done)
        truncated = bool(getattr(self, "env_steps", 0) >= self.max_steps)
        return obs, reward, terminated, truncated, info

    @staticmethod
    def _load_reward_from_file(path: str):
        """
        從檔案中載入 reward function。
        使用隔離的 namespace 執行，確保安全。
        """
        with open(path, "r") as f:
            code = f.read()

        namespace = {}
        exec(compile(code, path, "exec"), namespace)

        if "reward" not in namespace or not callable(namespace["reward"]):
            raise ValueError(f"'reward' function not found in: {path}")

        return namespace["reward"]

    @staticmethod
    def _to_gymnasium_space(space):
        """Convert common gym spaces to gymnasium spaces."""
        gsp = gym.spaces

        if isinstance(space, gsp.Space):
            return space

        # Handle legacy gym space classes by name to avoid direct dependency coupling.
        cls_name = space.__class__.__name__
        if cls_name == "Discrete":
            return gsp.Discrete(space.n)
        if cls_name == "Box":
            return gsp.Box(low=np.array(space.low), high=np.array(space.high), dtype=space.dtype)
        if cls_name == "Tuple":
            return gsp.Tuple(tuple(LLMEnhancedTwoStageAmp._to_gymnasium_space(s) for s in space.spaces))

        # Fall back to original object; RLlib will raise a clear error if unsupported.
        return space
