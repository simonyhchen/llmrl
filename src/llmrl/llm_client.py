"""
LLM Client abstraction layer.

Supports multiple LLM backends with a unified interface.
Currently implemented:
  - HuggingFaceClient: free Inference API
  - OllamaClient: free local model

Future support (placeholder):
  - OpenAIClient: GPT-4 etc.
  - ClaudeClient: Claude 3.5 Sonnet etc.

Usage:
    from src.llmrl.llm_client import get_llm_client
    import src.llmrl.config as config

    client = get_llm_client(config.LLM_PROVIDER)
    response = client.generate_response("Your prompt here")
"""

import ast
from abc import ABC, abstractmethod

import src.llmrl.config as config


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        """
        Send a prompt and return the model's text response.

        Args:
            prompt: The input prompt string.

        Returns:
            The model's response as a string.
        """
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Free backends
# ─────────────────────────────────────────────────────────────────────────────

class HuggingFaceClient(LLMClient):
    """
    Hugging Face Inference API client (free tier).
    Requires: pip install huggingface_hub
    Get your free token at: https://huggingface.co/settings/tokens
    """

    def __init__(self, api_key: str = None, model: str = None):
        try:
            from huggingface_hub import InferenceClient
        except ImportError:
            raise ImportError("huggingface_hub is required. Run: pip install huggingface_hub")

        self.model = model or config.HF_MODEL
        self.client = InferenceClient(
            model=self.model,
            token=api_key or config.HF_API_KEY or None,
        )

    def generate_response(self, prompt: str) -> str:
        response = self.client.text_generation(
            prompt,
            max_new_tokens=512,
            temperature=0.3,
        )
        return response.strip()


class OllamaClient(LLMClient):
    """
    Ollama client for local models (completely free, no API key needed).
    Requires: pip install ollama
    Install Ollama: https://ollama.com/
    Pull a model: ollama pull llama3.1
    """

    def __init__(self, host: str = None, model: str = None):
        try:
            import ollama
        except ImportError:
            raise ImportError("ollama is required. Run: pip install ollama")

        self.host = host or config.OLLAMA_HOST
        self.model = model or config.OLLAMA_MODEL
        self._ollama = ollama

    def generate_response(self, prompt: str) -> str:
        response = self._ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            host=self.host,
        )
        return response["message"]["content"].strip()


# ─────────────────────────────────────────────────────────────────────────────
# Paid backends (future)
# ─────────────────────────────────────────────────────────────────────────────

class OpenAIClient(LLMClient):
    """
    OpenAI API client (paid).
    Requires: pip install openai
    Get your API key at: https://platform.openai.com/
    """

    def __init__(self, api_key: str = None, model: str = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai is required. Run: pip install openai")

        key = api_key or config.OPENAI_API_KEY
        if not key:
            raise ValueError("OPENAI_API_KEY is not set in config.py")

        self.model = model or config.OPENAI_MODEL
        self.client = OpenAI(api_key=key)

    def generate_response(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()


class ClaudeClient(LLMClient):
    """
    Anthropic Claude API client (paid).
    Requires: pip install anthropic
    Get your API key at: https://www.anthropic.com/
    """

    def __init__(self, api_key: str = None, model: str = None):
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic is required. Run: pip install anthropic")

        key = api_key or config.CLAUDE_API_KEY
        if not key:
            raise ValueError("CLAUDE_API_KEY is not set in config.py")

        self.model = model or config.CLAUDE_MODEL
        self.client = anthropic.Anthropic(api_key=key)

    def generate_response(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def get_llm_client(provider: str = None) -> LLMClient:
    """
    Factory function to instantiate the correct LLM client.

    Args:
        provider: One of "huggingface", "ollama", "openai", "claude".
                  Defaults to config.LLM_PROVIDER.

    Returns:
        An LLMClient instance for the specified provider.
    """
    provider = (provider or config.LLM_PROVIDER).lower()

    if provider == "huggingface":
        return HuggingFaceClient()
    elif provider == "ollama":
        return OllamaClient()
    elif provider == "openai":
        return OpenAIClient()
    elif provider == "claude":
        return ClaudeClient()
    else:
        raise ValueError(
            f"Unknown LLM provider: '{provider}'. "
            "Choose from: huggingface, ollama, openai, claude"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Utility: safe code validator
# ─────────────────────────────────────────────────────────────────────────────

def validate_generated_code(code: str) -> bool:
    """
    Validate that LLM-generated Python code is syntactically correct.
    Uses AST parsing to avoid executing arbitrary code.

    Args:
        code: Python code string to validate.

    Returns:
        True if valid Python syntax, False otherwise.
    """
    try:
        ast.parse(code)
        return True
    except SyntaxError as e:
        print(f"[validate_generated_code] SyntaxError: {e}")
        return False
