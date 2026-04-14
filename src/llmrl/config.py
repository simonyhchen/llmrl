# pylint: disable=invalid-name
import json
import os
from pathlib import Path


def _load_settings() -> dict:
	"""Load provider settings from optional JSON file."""
	settings_path = os.getenv("LLMRL_SETTINGS_FILE", "")
	if settings_path:
		candidate = Path(settings_path)
	else:
		candidate = Path(__file__).with_name("llm_settings.json")

	if not candidate.exists():
		return {}

	try:
		with candidate.open("r", encoding="utf-8") as f:
			data = json.load(f)
			return data if isinstance(data, dict) else {}
	except Exception:
		return {}


_SETTINGS = _load_settings()


def _pick(key: str, env_key: str, default=None):
	"""Resolve config value in order: ENV > settings file > default."""
	env_val = os.getenv(env_key)
	if env_val is not None and env_val != "":
		return env_val
	if key in _SETTINGS and _SETTINGS[key] not in (None, ""):
		return _SETTINGS[key]
	return default


# Unified LLM Provider Configuration
# Priority: env vars > llm_settings.json > defaults
# Options: "ollama", "openai", "claude", "huggingface"
LLM_PROVIDER = _pick("provider", "LLMRL_PROVIDER", "ollama")

# Ollama (free local models)
OLLAMA_HOST = _pick("ollama_host", "OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = _pick("ollama_model", "OLLAMA_MODEL", "qwen3:8b")

# OpenAI (paid)
OPENAI_API_KEY = _pick("openai_api_key", "OPENAI_API_KEY", None)
OPENAI_MODEL = _pick("openai_model", "OPENAI_MODEL", "gpt-4o-mini")

# Anthropic Claude (paid)
CLAUDE_API_KEY = _pick("claude_api_key", "CLAUDE_API_KEY", None)
CLAUDE_MODEL = _pick("claude_model", "CLAUDE_MODEL", "claude-3-5-sonnet-20241022")

# Hugging Face (free-tier API fallback)
HF_API_KEY = _pick("hf_api_key", "HF_API_KEY", "")
HF_MODEL = _pick("hf_model", "HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.1")
