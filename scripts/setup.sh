#!/bin/bash
set -e

# Setup script for LLMRL (all Python 3.11.1)

echo "[setup] Project root: $(pwd)"

if [ ! -d "autockt/AutoCkt" ]; then
  echo "[setup] Cloning AutoCkt..."
  git clone https://github.com/ksettaluri6/AutoCkt autockt/AutoCkt
else
  echo "[setup] AutoCkt already exists, skip clone."
fi

if ! conda env list | grep -q "llmrl311"; then
  echo "[setup] Creating conda env llmrl311 (python=3.11.1)..."
  conda create -y -n llmrl311 -c conda-forge python=3.11.1
else
  echo "[setup] Conda env llmrl311 already exists."
fi

echo "[setup] Installing dependencies into llmrl311..."
conda run -n llmrl311 pip install -r requirements.txt

echo "[setup] Done."
echo "[setup] Activate with: conda activate llmrl311"
