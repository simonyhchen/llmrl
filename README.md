Project: LLMRL - LLM-Enhanced Reinforcement Learning for Circuit Design

**目標**: 研究 LLM 是否可以通過生成自適應的 reward function，幫助 RL 更有效地優化電路設計（應用於 SPICE simulation）。

## 快速開始 (Quick Start)

### 環境設置

#### 1. 主訓練環境 (Python 3.11 + Ray RLlib)
```bash
# 建立 conda 環境
conda create -n llmrl311 python=3.11 -y
conda activate llmrl311

# 安裝依賴
pip install ray[rllib]>=2.9 torch==2.1.1 gymnasium==1.2.2 numpy matplotlib pandas

# （如果使用 OpenAI/Claude API）
pip install openai anthropic

# 驗證 Ray 安裝
python -c "import ray; print(f'Ray {ray.__version__}')"
```

#### 2. Ollama + 本地 LLM 設置（推薦用於開發）

**a) 安裝 Ollama**
- **Windows**: 下載 [Ollama Desktop App](https://ollama.com/download)
- **Linux/Mac**: `curl -fsSL https://ollama.com/install.sh | sh`

**b) 下載輕量模型（推薦）**
```bash
ollama pull qwen2.5-coder:7b   # ~4.7GB，適合 8GB VRAM 的 GPU
# 或
ollama pull mistral:7b          # ~4.1GB，另一個快速選項
```

**c) 啟動 Ollama server**
```bash
# Windows: Ollama Desktop App 會自動啟動
# Linux: 
ollama serve

# 驗證連線
curl http://localhost:11434/api/tags
```

**d) 配置 LLM settings**
```bash
# 複製範本（如果尚未存在）
cp src/llmrl/llm_settings.example.json src/llmrl/llm_settings.json

# 編輯 src/llmrl/llm_settings.json，確認：
# - "provider": "ollama"
# - "ollama_host": "http://localhost:11434"（本機）或 Windows IP（WSL）
# - "ollama_model": "qwen2.5-coder:7b"
```

**e) 測試 LLM 連線**
```bash
cd /path/to/llmrl
source .venv/bin/activate  # 如果有 venv
python src/llmrl/check_llm_connection.py
```

### 運行 LLM 增強的 RL 訓練

```bash
# 啟動 Python 3.11 環境
conda activate llmrl311

# 單次訓練（5 iterations，用 LLM 生成 reward）
python src/llmrl/train_with_llm.py \
  --experiment_name my_experiment \
  --max_iters 5 \
  --num_workers 2 \
  --reward_description "獎勵所有規格滿足，懲罰規格偏離"

# 自動比較 Baseline vs LLM（20 iterations）
LLMRL_PYTHON=$(which python) python src/llmrl/compare.py --max_iters 20
```

### 輸出位置

- **RL 指標**: `experiments/{experiment_name}/metrics.jsonl`
- **比較報告**: `comparison_results/{timestamp}/comparison_report.json` 與圖表
- **生成的 Reward Code**: `ray_tmp/llm_reward.py`

---

## 詳細設置

Project : LLMRL
想要研究LLM是否可以將RL控制得更好,應用在spice simulation

#Dir:
##src : 未來要放一些llm控制rl的成品
##autockt: 
###這個是參考一個呼叫ngspice做電路最佳化的reforcement learning experimental platform
###原始套件的git: git clone https://github.com/ksettaluri6/AutoCkt.git (只下載唯讀來讓我們參考)
###執行RL 最佳化電路步驟如下:
**註記：** 這些步驟基於官方AutoCkt (https://github.com/ksettaluri6/AutoCkt) 的README.md，以確保可重現性。未進行任何自訂修改，適合合作開發和重現結果。
1. 安裝NGspice（如果尚未安裝）：參考 https://sourceforge.net/projects/ngspice/files/ng-spice-rework/old-releases/27/ 的安裝說明。注意可能需要調整標誌以適應您的機器。

2. 進入AutoCkt資料夾，並使用conda安裝需要的套件：
   ```
   $ cd AutoCkt
   $ conda env create -f environment.yml
   ```

3. 啟動環境：
   ```
   $ conda activate autockt
   ```

4. （可選）如果需要，安裝額外的套件使用pip。檢查environment.yml以確保版本正確。

5. 連接library files：
   ```
   $ python eval_engines/ngspice/ngspice_inputs/correct_inputs.py
   ```

6. 生成spec：
   ```
   $ python autockt/gen_specs.py --num_specs ###
   ```
   - spec檔為pickle檔，會生在autockt/gen_specs/裡面
   - ### 為生成的spec組數，如果要再現github中的results，要設定350

7. （可選）設定環境變數（如果需要）：
   ```
   $ export PATH=$HOME/ngspice-27/opt/bin:$PATH  # 如果NGspice安裝在自訂路徑
   $ export PYTHONPATH=$PWD
   $ mkdir -p ray_tmp
   $ export RAY_TMPDIR=$PWD/ray_tmp
   ```

8. 開啟ipython介面並開始訓練：
   ```
   $ ipython
   %run autockt/val_autobag_ray.py
   ```
   - 若開始正常跑，應該過不到一分鐘會出現訓練訊息
   - 整個訓練過程根據設定的num_specs越大，需要的時間越久
   - RL訓練過程中每個iteration產生的電路在ckt_da/designs_two_stage_opamp/裡面
   - RL訓練過程中的checkpoint會在ray_results/train_45nm_ngspice/PPO_opamp-v0_xxx/裡面

9. 驗證 AutoCKT：
   - 生成新的驗證spec：
     ```
     $ python autockt/gen_specs.py --num_specs ###
     ```
   - 進入ipython，輸入驗證指令：
     ```
     %run autockt/rollout.py /abs/path/to/checkpoint --run PPO --env opamp-v0 --num_val_specs ### --traj_len ## --no-render
     ```
     - /abs/path/to/checkpoint：RL訓練的checkpoint路徑，例如 ~/ray_results/train_45nm_ngspice/PPO_opamp-v0_xxx/checkpoint_%%%/checkpoint-%%%
     - --num_val_specs ###：驗證spec組數，例如 50
     - --traj_len ##：每組規格的最大步數，例如 30
   - 結果會顯示達標的spec數量，例如 Specs reached: 44/50

## GitHub Commit 規劃
根據項目的目標（讓合作夥伴能重現和開發），以下是哪些文件需要commit的建議。基於項目的結構和reproducibility（可重現性）來分類，確保倉庫輕量、清晰，且易於合作。

### 總體原則
- **Commit重點**：只commit你自己的代碼、文檔和配置。避免commit大型外部依賴（如整個autockt/），因為它會讓倉庫膨脹，且可以通過腳本自動獲取。
- **Reproducibility**：在README.md中添加清晰的設置說明，讓合作夥伴能從頭重現環境。
- **版本控制最佳實踐**：使用`.gitignore`排除不必要的文件（如日誌、快取、環境文件）。
- **初始commit**：先commit核心文件，然後逐步添加新功能。

### 需要Commit的文件/目錄
1. **README.md**（根目錄）
   - 原因：這是項目的主要文檔，已包含修正的execute autockt章節和註記。合作夥伴需要它來理解項目和設置。

2. **src/** 目錄（及其子文件）
   - 原因：這是你的核心開發區域，包括未來實現的LLMRL模塊（如llm_client.py, reward_generator.py等）。一旦你實現這些，它們是項目的新價值。

3. **requirements.txt**（新建）
   - 原因：列出Python依賴，讓合作夥伴能輕鬆安裝環境。包括autockt的依賴和你的新依賴（如openai, huggingface_hub）。

4. **.gitignore**（新建）
   - 原因：排除不必要的文件，如環境文件、快取、日誌等。示例內容：
     ```
     # Python
     __pycache__/
     *.pyc
     *.pyo
     .env

     # Ray and training logs
     ray_results/
     ray_tmp/
     *.log

     # IDE
     .vscode/
     .idea/

     # OS
     .DS_Store
     Thumbs.db

     # Exclude autockt/ as it's cloned externally
     autockt/
     ```

5. **scripts/** 或 **setup.py**（新建）
   - 原因：添加一個簡單的腳本（如`setup.sh`或`clone_autockt.sh`），自動克隆autockt並設置環境。這確保reproducibility，而不需commit整個autockt/。

### 不需要Commit的文件/目錄
1. **autockt/** 目錄
   - 原因：它是外部克隆的倉庫，大小龐大，且可以通過腳本或README說明自動獲取。

2. **環境文件**（如conda環境本身）和訓練日誌。

### Commit步驟建議
1. 初始化倉庫：`git init`，添加`.gitignore`。
2. 初始Commit：`git add README.md src/ requirements.txt .gitignore`，`git commit -m "Initial commit"`。
3. 推送：`git remote add origin <URL>`，`git push -u origin main`。
4. 後續：實現功能後逐步commit。

## Todo List
我們一個一個逐步實現，以下是分解的任務清單：

- [ ] 添加GitHub Commit規劃到README.md（已完成）
- [ ] 創建`.gitignore`文件
- [ ] 創建`requirements.txt`，基於autockt/environment.yml和新增依賴
- [ ] 創建`scripts/setup.sh`腳本，用於自動克隆autockt和安裝依賴
- [ ] 初始化Git倉庫並進行初始commit
- [ ] 在src/llmrl/中實現`llm_client.py`（抽象LLM客戶端，支持Hugging Face）
- [ ] 實現`reward_generator.py`（基於自然語言生成reward代碼）
- [ ] 實現`llm_analyzer.py`（分析訓練日誌，建議reward調整）
- [ ] 創建`modified_env.py`（修改RL環境，整合動態reward）
- [ ] 創建`train_with_llm.py`（修改訓練腳本，使用新環境）
- [ ] 測試小規模整合（生成reward並運行訓練）
- [ ] 更新README.md文檔並commit所有變更

---

## 項目狀態 (April 14, 2026)

### ✅ 已完成
- **Python 3.11 遷移**: 全套流程從舊版 Python → Python 3.11.1（Ray RLlib 2.54.1, gymnasium 1.2.2）
- **Ollama 多模型支持**: 實裝可切換的 LLM provider（Ollama, OpenAI, Claude, HuggingFace）
- **RewardGenerator**: LLM 根據自然語言自動生成 reward function
- **自動化比較管道**: `compare.py` 自動跑 baseline vs LLM 訓練並產生對比圖表
- **連線測試工具**: `check_llm_connection.py` 驗證 LLM 可用性

### ✅ 已驗證的結果
- **20 iterations 比較運行**（當前最佳結果）:
  - Baseline final reward: **-23.6**
  - LLM-Enhanced final reward: **-115.5**（更激進的懲罰策略）
  - 兩者改善幅度相近，LLM 學習波動較大但持續改善
- **Ollama qwen2.5-coder:7b** 推理時間 ~1 秒/prompt（GPU 加速）

### 🔧 已知問題與改進空間
1. **Reward Scale**: LLM 選擇的懲罰策略（如二次方損失）會改變 reward 尺度，難以直接比較
   - 建議: 在 prompt 加入更多結構化指引，要求線性加總懲罰
2. **Learning Stability**: LLM reward 導致的梯度變化較大，收斂需要更多 iterations
   - 建議: 嘗試 50-100 iterations 看收斂行為
3. **模型選擇**: qwen3-coder:30b 太大（超過 8GB VRAM），實際使用 qwen2.5-coder:7b
   - 備選: mistral:7b

### 📋 交接要點

#### 對 @合作夥伴
1. **環境配置只需 5 分鐘**: 按照上方「快速開始」節點的步驟即可
2. **Ollama 對新手友善**: 無需 API key，支持本地 GPU 加速，推理費用為 0
3. **修改 reward prompt 很簡單**: 在 `src/llmrl/reward_generator.py` 的 `_SYSTEM_PROMPT` 修改要求即可
4. **主要代碼位置**:
   - `src/llmrl/train_with_llm.py`: 主訓練邏輯（新增 LLM reward 參數化）
   - `src/llmrl/reward_generator.py`: LLM 生成 reward 的核心
   - `src/llmrl/compare.py`: 自動化比較工具
   - `src/llmrl/llm_client.py`: 多 provider LLM 客戶端

#### 建議的後續研究方向
- **Prompt 工程**: 優化 `reward_description` 和系統 prompt，測試不同懲罰策略
- **超參數調優**: batch size, learning rate 在 LLM reward 下的最優配置
- **多目標 reward**: LLM 同時生成多個 sub-reward（功耗、面積、性能）
- **遷移學習**: 用訓練好的 agent 初始化新的電路設計任務

---