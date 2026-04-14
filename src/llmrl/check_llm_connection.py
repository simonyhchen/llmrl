"""Simple connectivity test for configured LLM provider."""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.llmrl.llm_client import get_llm_client
import src.llmrl.config as config


def main():
    provider = config.LLM_PROVIDER
    print(f"[check] provider={provider}")

    client = get_llm_client(provider)
    prompt = "Reply with exactly: LLM connection OK"
    resp = client.generate_response(prompt)

    print("[check] response:")
    print(resp)


if __name__ == "__main__":
    main()
