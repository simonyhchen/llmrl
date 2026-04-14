# 項目交接清單 (Handoff Checklist)

## 📦 當前版本信息
- **Latest Commit**: `d1c3e1f` - docs: add git repo URL and SSH setup to handoff guide
- **Date**: April 14, 2026
- **State**: LLM reward generation 可正常運作，已驗證 20 iterations 訓練
- **Dev Machine**: WSL2 (Linux) + Windows 11, i7-14700HX, 16GB RAM, RTX 5070 Laptop 8GB VRAM

---

## 🚀 快速上手 (< 15 minutes)

### Step 0: 取得代碼
```bash
# 首次 clone（選 SSH 或 HTTPS 擇一）
git clone git@github.com:simonyhchen/llmrl.git      # SSH（需要設 SSH key）
git clone https://github.com/simonyhchen/llmrl.git  # HTTPS

# clone 到與 dev machine 相同位置
mkdir -p ~/project && cd ~/project
git clone git@github.com:simonyhchen/llmrl.git
cd llmrl   # 完整路徑: ~/project/llmrl
```

**設置 SSH key（如果還沒設過）**:
```bash
# 產生 SSH key
ssh-keygen -t ed25519 -C "your.email@example.com"

# 將公鑰加到 GitHub: Settings → SSH and GPG keys → New SSH key
cat ~/.ssh/id_ed25519.pub  # 複製這個內容貼到 GitHub

# 測試連線
ssh -T git@github.com  # 看到 "Hi simonyhchen!" 代表成功
```

### Step 1: 拉下最新代碼（已有 repo 者）
```bash
cd ~/project/llmrl
git pull origin master
```

### Step 2: 安裝 Miniconda（如果尚未安裝）
```bash
# 安裝 Miniconda3 到 ~/lib/miniconda3（與 dev machine 一致）
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/lib/miniconda3
echo 'export PATH="$HOME/lib/miniconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### Step 3: 安裝 Python 3.11 訓練環境
```bash
# 建立 conda 環境（名稱與 dev machine 完全一致）
conda create -n llmrl311 python=3.11 -y
conda activate llmrl311

# 安裝訓練依賴
pip install ray[rllib]>=2.9 torch==2.1.1 gymnasium numpy matplotlib pandas

# 驗證
python -c "import ray; import gymnasium; print('OK')"
# 預期: OK
```

### Step 4: 安裝 Ollama 並下載模型
```bash
# Linux 安裝
curl -fsSL https://ollama.com/install.sh | sh

# 下載輕量模型（~4.7GB，適合 8GB VRAM）
ollama pull qwen2.5-coder:7b

# 啟動 server（如果沒有自動啟動）
ollama serve   # 另開一個 terminal 執行

# 確認 server 正常
curl http://localhost:11434/api/tags
```

### Step 5: 設置 LLM 配置
```bash
cd ~/project/llmrl
cp src/llmrl/llm_settings.example.json src/llmrl/llm_settings.json
```

編輯 `src/llmrl/llm_settings.json`，確認內容如下（本機 Linux 用 localhost）：
```json
{
  "provider": "ollama",
  "ollama_host": "http://localhost:11434",
  "ollama_model": "qwen2.5-coder:7b",
  "openai_api_key": "",
  "openai_model": "gpt-4o-mini",
  "claude_api_key": "",
  "claude_model": "claude-3-5-sonnet-20241022",
  "hf_api_key": "",
  "hf_model": "mistralai/Mistral-7B-Instruct-v0.1"
}
```

> **如果你跑 WSL2（Windows 上的 Linux）且 Ollama 裝在 Windows**:
> 需要把 `ollama_host` 改成 Windows 主機 IP：
> ```bash
> # 找出 Windows 主機 IP
> ip route show default | awk '{print $3}'
> # 把結果填入 ollama_host，例如: "http://172.30.224.1:11434"
> ```
> 並在 Windows 設環境變數 `OLLAMA_HOST=0.0.0.0` 後重啟 Ollama Desktop App。

### Step 6: 驗證 LLM 連線
```bash
cd ~/project/llmrl
conda activate llmrl311
python src/llmrl/check_llm_connection.py
# 預期輸出: [check] response: LLM connection OK
```

### Step 7: 跑測試訓練
```bash
conda activate llmrl311
cd ~/project/llmrl

# LLM reward 自動生成 + 訓練（5 iterations, ~3 分鐘）
LLMRL_PYTHON=/home/$(whoami)/lib/miniconda3/envs/llmrl311/bin/python \
  /home/$(whoami)/lib/miniconda3/envs/llmrl311/bin/python \
  src/llmrl/train_with_llm.py --max_iters 5 --num_workers 2

# 或自動比較 baseline vs LLM（20 iterations, ~20 分鐘）
LLMRL_PYTHON=/home/$(whoami)/lib/miniconda3/envs/llmrl311/bin/python \
  /home/$(whoami)/lib/miniconda3/envs/llmrl311/bin/python \
  src/llmrl/compare.py --max_iters 20
```

> **注意**: `LLMRL_PYTHON` 必須指向 `llmrl311` 的 Python，確保 `ray` 和 `gymnasium` 都在同一個環境。

---

## 📂 代碼結構與文件說明

```
src/llmrl/
├── config.py                    # LLM provider 配置讀取（支持環境變數覆蓋）
├── llm_client.py               # 多 provider 客戶端（Ollama/OpenAI/Claude/HF）
├── reward_generator.py          # 核心：LLM 生成 reward function 的邏輯
├── check_llm_connection.py      # 診斷工具：測試 LLM 連線
├── train_with_llm.py           # 主訓練腳本（改進版，支持 LLM reward）
├── compare.py                  # 自動化比較：baseline vs LLM
├── modified_env.py             # 包裝 AutoCkt env 以支持 LLM reward 注入
├── llm_settings.json           # 當前 LLM 配置（本地範本，不含敏感信息）
└── llm_settings.example.json   # 配置範本

experiments/                    # RL 訓練輸出（用 .gitignore 排除）
├── train_45nm_ngspice_baseline/  # Baseline 執行
├── train_45nm_ngspice_llm/       # LLM-enhanced 執行
└── */metrics.jsonl              # 每次訓練的 reward 等指標

comparison_results/            # 自動比較的圖表與報告（用 .gitignore 排除）
```

---

## 🔧 核心功能速覽

### 1. 自動生成 Reward Function
```python
from src.llmrl.reward_generator import RewardGenerator

gen = RewardGenerator(provider="ollama")
code = gen.generate("獎勵規格滿足，懲罰功耗超過 10%")
# 返回完整的 Python reward function 字符串
```

### 2. 多 Provider 支持
```python
from src.llmrl.llm_client import get_llm_client

# Ollama（推薦開發用）
client = get_llm_client("ollama")

# 或 OpenAI/Claude（設置 API key）
client = get_llm_client("openai")
```

### 3. 自動化對比訓練
```bash
# Baseline: 純 AutoCkt reward（不用 LLM）
# LLM: 用自動生成的 reward function

LLMRL_PYTHON=~/lib/miniconda3/envs/llmrl311/bin/python \
  ~/lib/miniconda3/envs/llmrl311/bin/python \
  src/llmrl/compare.py --max_iters 20
# 輸出: ~/project/llmrl/comparison_results/{timestamp}/comparison_chart.png
```

---

## 📊 當前驗證結果

### 20 iterations 運行結果
| Metric | Baseline | LLM-Enhanced |
|--------|----------|--------------|
| 初始 Reward | ~-31 | ~-175 |
| 最終 Reward | -23.6 | -115.5 |
| 改善量 | +7.4 | +59.5 |
| 學習穩定度 | 穩定，增速平緩 | 波動較大，持續改善 |

**解釋**:
- LLM 選擇了更激進的懲罰策略（二次方損失），導致 reward 尺度更負
- 兩者改善趨勢相似，LLM 的波動反映了激進的梯度信號
- 建議嘗試 50+ iterations 看長期收斂

### 性能指標
- **Ollama qwen2.5-coder:7b** 推理: ~1 秒/prompt（GPU 加速）
- **訓練時間**: 5 iterations ~3 分鐘；20 iterations ~20 分鐘（取決於硬體）

---

## 🔬 建議的研究方向

### 短期（1-2 週）
- [ ] **Prompt 工程**: 修改 `reward_generator.py` 中的 `_SYSTEM_PROMPT`，測試不同懲罰策略
  - 嘗試: "使用線性加總懲罰（不用平方）"
  - 觀察效果: reward 尺度與學習速度
  
- [ ] **超參優化**: 在 LLM reward 下測試不同 batch size / learning rate
  - 檔案: `src/llmrl/train_with_llm.py` 的 `ppo_config`
  
- [ ] **更多 iterations**: 跑 50-100 iterations 看最終收斂效果
  - 命令: `python src/llmrl/compare.py --max_iters 50`

### 中期（1 個月）
- [ ] **多目標 Reward**: LLM 同時優化功耗、面積、性能
  - 修改: `reward_description` 加入多參數要求
  
- [ ] **Reward Shaping**: 嘗試不同的 reward 縮放與 clip 策略
  - 檔案: `src/llmrl/modified_env.py` 的 `reward()` 方法
  
- [ ] **模型切換**: 測試其他 Ollama 模型
  - 備選: `mistral:7b`, `neural-chat:7b`

### 長期（持續研究）
- [ ] **遷移學習**: 用訓練好的 agent 初始化新的電路設計任務
- [ ] **因果推理**: 分析 LLM 生成的 reward 與訓練軌跡的相關性
- [ ] **對標研究**: 與其他 RL+ LLM 方案比較（如 reward modeling）

---

## ⚙️ 常見問題

### Q1: Ollama 連線失敗
**A**: 確保 Ollama server 在運行
```bash
# Linux: 另開 terminal 執行
ollama serve

# 測試連線
curl http://localhost:11434/api/tags

# 或用我們的檢測工具
cd ~/project/llmrl && conda activate llmrl311
python src/llmrl/check_llm_connection.py
```

### Q2: 訓練很慢
**A**: 檢查
1. **Ollama 推理**: `cd ~/project/llmrl && python src/llmrl/check_llm_connection.py`（應 < 2 s）
2. **GPU 使用**: `nvidia-smi`（應顯示 Ollama process 使用 GPU memory）
3. **num_workers**: 減少 workers 避免 OOM
   ```bash
   # 在 train_with_llm.py 的 ppo_config 調整，或用參數
   python src/llmrl/train_with_llm.py --max_iters 5 --num_workers 1
   ```

### Q3: 生成的 reward 代碼看起來很奇怪
**A**: LLM 可能理解不同。檢查：
1. 查看 `~/project/llmrl/ray_tmp/llm_reward.py` 的實際生成代碼
2. 修改 `~/project/llmrl/src/llmrl/reward_generator.py` 中的 `_SYSTEM_PROMPT`
3. 測試另一個 Ollama 模型：`ollama pull mistral:7b`，再改 `llm_settings.json`

### Q4: 我想用 OpenAI/Claude 的 LLM
**A**: 
1. 在 `llm_settings.json` 改 `"provider": "openai"` 或 `"claude"`
2. 設置 API key: `"openai_api_key": "sk-..."`
3. 確認網路連線，其他邏輯不變

---

## 🔗 相關資源

- **Ollama**: https://ollama.com/
- **Ray RLlib**: https://docs.ray.io/en/latest/rllib/index.html
- **Gymnasium**: https://gymnasium.farama.org/
- **AutoCkt**: https://github.com/ksettaluri6/AutoCkt

---

## 📝 開發環境規格參考（Dev Machine）

| 項目 | 規格 |
|------|------|
| OS | Windows 11 + WSL2 (Ubuntu) |
| CPU | Intel i7-14700HX (14 cores / 28 threads) |
| RAM | 16GB |
| GPU | NVIDIA RTX 5070 Laptop 8GB VRAM |
| 磁碟 | 1TB |
| Miniconda 位置 | `~/lib/miniconda3` |
| 主環境 | `llmrl311`（Python 3.11.1）|
| 項目位置 | `~/project/llmrl` |
| Ollama 位置 | Linux 本機（`localhost:11434`）|
| 模型 | `qwen2.5-coder:7b`（~4.7GB）|

## 📝 交接默認假設

1. ✅ 你有 Python 3.11+ 環境（推薦 `conda create -n llmrl311 python=3.11`）
2. ✅ 你有 ~8GB+ VRAM 的 NVIDIA GPU（或接受跑 CPU，速度慢 5-10x）
3. ✅ Ollama 裝在同一台機器上（本機 Linux 或 WSL2 + Windows Ollama）
4. ✅ 你熟悉基本 RL 概念與 PyTorch

---

## 🙋 有問題？

檢查 `README.md` 的「詳細設置」部分或查看各檔案的 docstring。

祝研究順利！🚀
