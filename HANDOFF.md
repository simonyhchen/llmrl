# 項目交接清單 (Handoff Checklist)

## 📦 當前版本信息
- **Latest Commit**: `340f1e5` - feat: integrate Ollama + LLM reward generation
- **Date**: April 14, 2026
- **State**: LLM reward generation 可正常運作，已驗證 20 iterations 訓練

---

## 🚀 快速上手 (< 15 minutes)

### Step 0: 取得代碼
```bash
# 首次 clone（選 SSH 或 HTTPS 擇一）
git clone git@github.com:simonyhchen/llmrl.git      # SSH （需要設 SSH key）
git clone https://github.com/simonyhchen/llmrl.git  # HTTPS

cd llmrl
```

**設置 SSH key（如果還沒設過）**:
```bash
# 產生 SSH key
ssh-keygen -t ed25519 -C "your.email@example.com"

# 將公鑰加到 GitHub: Settings → SSH and GPG keys → New SSH key
cat ~/.ssh/id_ed25519.pub  # 複製這個內容貼到 GitHub
```

### Step 1: 拉下最新代碼（已有 repo 者）
```bash
cd llmrl
git pull origin master
```

### Step 2: 安裝環境（如果尚未）
```bash
# Python 3.11 訓練環境
conda create -n llmrl311 python=3.11 -y
conda activate llmrl311
pip install ray[rllib]>=2.9 torch==2.1.1 gymnasium numpy matplotlib pandas

# Ollama（本地 LLM，無需 API key）
# Windows: https://ollama.com/download
# Linux: curl -fsSL https://ollama.com/install.sh | sh

# 下載模型
ollama pull qwen2.5-coder:7b
```

### Step 3: 設置 LLM 配置
```bash
cp src/llmrl/llm_settings.example.json src/llmrl/llm_settings.json
# 編輯 llm_settings.json，確認:
# - provider: "ollama"
# - ollama_host: "http://localhost:11434"
# - ollama_model: "qwen2.5-coder:7b"
```

### Step 4: 驗證連線
```bash
python src/llmrl/check_llm_connection.py
# 預期輸出: [check] response: LLM connection OK
```

### Step 5: 跑一個測試訓練
```bash
conda activate llmrl311
# LLM reward 自動生成 + 訓練（5 iterations, ~3 分鐘）
python src/llmrl/train_with_llm.py --max_iters 5 --num_workers 2

# 或自動比較 baseline vs LLM（20 iterations, ~20 分鐘）
LLMRL_PYTHON=$(which python) python src/llmrl/compare.py --max_iters 20
```

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

LLMRL_PYTHON=<path-to-python> python src/llmrl/compare.py --max_iters 20
# 輸出: comparison_results/{timestamp}/comparison_chart.png
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
# Windows: 打開 Ollama Desktop App
# Linux: ollama serve

# 測試連線
curl http://localhost:11434/api/tags
```

### Q2: 訓練很慢
**A**: 檢查
1. **Ollama 推理**: `python src/llmrl/check_llm_connection.py`（應 < 2 s）
2. **GPU 使用**: `nvidia-smi`（應顯示 GPU memory 被占用）
3. **num_workers**: 減少 workers 避免 OOM（在 `train_with_llm.py` 調整 `--num_workers`）

### Q3: 生成的 reward 代碼看起來很奇怪
**A**: LLM 可能理解不同。檢查：
1. 查看 `ray_tmp/llm_reward.py` 的實際生成代碼
2. 修改 prompt 在 `reward_generator.py` 加入更多結構化指引
3. 嘗試另一個 Ollama 模型

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

## 📝 交接默認假設

1. ✅ 你有 Python 3.11+ 環境（conda 或 venv）
2. ✅ 你有 ~8GB+ 的 GPU memory（NVIDIA GPU）或願意跑 CPU
3. ✅ 你能訪問本地 Ollama（如果跑 WSL，知道如何配置 host IP）
4. ✅ 你熟悉基本 RL 概念與 PyTorch

---

## 🙋 有問題？

檢查 `README.md` 的「詳細設置」部分或查看各檔案的 docstring。

祝研究順利！🚀
