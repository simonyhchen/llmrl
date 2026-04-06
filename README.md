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