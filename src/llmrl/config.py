# LLM Provider Configuration
# Change LLM_PROVIDER to switch between different LLM backends

# Options: "huggingface", "ollama", "openai", "claude"
LLM_PROVIDER = "huggingface"

# Hugging Face (free)
# Get your free API key at: https://huggingface.co/settings/tokens
HF_API_KEY = ""
HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"

# Ollama (free, local)
# Install Ollama: https://ollama.com/ and run `ollama pull llama3.1`
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "llama3.1"

# OpenAI (future, paid)
# Get your API key at: https://platform.openai.com/
OPENAI_API_KEY = None
OPENAI_MODEL = "gpt-4"

# Anthropic Claude (future, paid)
# Get your API key at: https://www.anthropic.com/
CLAUDE_API_KEY = None
CLAUDE_MODEL = "claude-3-5-sonnet-20241022"
