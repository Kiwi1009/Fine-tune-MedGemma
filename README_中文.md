# MedGemma 27B 中文微調指南

## 📋 專案概述

這個專案展示了如何使用 LoRA (Low-Rank Adaptation) 技術在醫療問答數據上微調 MedGemma 27B 多模態模型。專案針對 GPU 訓練進行了優化，特別適合 RTX 4090 等高端顯卡。

## 🎯 主要功能

- **GPU 加速訓練**: 支援 CUDA GPU 加速，優化記憶體使用
- **LoRA 微調**: 高效參數適應，無需完整模型重訓練
- **醫療專用**: 基於 MedQuAD 醫療問答數據集訓練
- **記憶體監控**: 內建 GPU 記憶體使用追蹤和優化建議
- **中文支援**: 完整的中文說明和錯誤處理

## 🖥️ 系統需求

### 硬體需求
- **GPU**: NVIDIA GPU，至少 24GB VRAM (推薦 RTX 4090 或 A100)
- **記憶體**: 至少 32GB 系統記憶體
- **儲存**: 至少 50GB 可用空間

### 軟體需求
- **作業系統**: Windows 10/11, Linux, 或 macOS
- **Python**: 3.8 或更高版本
- **CUDA**: 11.8 或更高版本 (與 PyTorch 相容)

## 🚀 安裝指南

### 1. 環境準備

```bash
# 創建虛擬環境 (推薦)
conda create -n medgemma python=3.10
conda activate medgemma

# 或使用 venv
python -m venv medgemma_env
source medgemma_env/bin/activate  # Linux/Mac
# 或
medgemma_env\Scripts\activate  # Windows
```

### 2. 安裝必要套件

```bash
# 安裝 PyTorch (根據您的 CUDA 版本選擇)
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安裝其他必要套件
pip install transformers==4.44.0
pip install peft
pip install bitsandbytes
pip install accelerate
pip install datasets
pip install pandas
pip install huggingface_hub
pip install jupyter
pip install tensorboard
```

### 3. 驗證安裝

```python
import torch
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU 型號: {torch.cuda.get_device_name(0)}")
    print(f"GPU 記憶體: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
```

## 🔐 模型存取設置

### 1. HuggingFace 帳號設置

1. 前往 [HuggingFace](https://huggingface.co/) 註冊帳號
2. 前往 [MedGemma 模型頁面](https://huggingface.co/google/medgemma-4b-multimodal)
3. 點擊 "Request Access" 申請存取權限
4. 接受授權協議
5. 等待審核通過 (通常需要幾分鐘到幾小時)

### 2. 取得 API Token

1. 前往 [Token 設置頁面](https://huggingface.co/settings/tokens)
2. 點擊 "New token"
3. 選擇 "Read" 權限
4. 複製生成的 token

### 3. 驗證存取

```python
from huggingface_hub import login
login(token="your_token_here")

# 測試模型存取
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("google/medgemma-4b-multimodal")
print("✅ 模型存取成功!")
```

## 📊 數據集準備

### 1. MedQuAD 數據集

專案使用 MedQuAD 醫療問答數據集，包含：
- **問題**: 醫療相關問題
- **答案**: 專業醫療回答
- **來源**: 可靠的醫療資源
- **領域**: 涵蓋多個醫療專業領域

### 2. 數據格式

數據應為 CSV 格式，包含以下欄位：
```csv
question,answer,source,focus_area
"什麼是糖尿病?","糖尿病是一種代謝疾病...",medical_textbook,endocrinology
```

### 3. 數據預處理

```python
import pandas as pd

# 載入數據
df = pd.read_csv('medquad.csv')

# 格式化為指令-回應格式
def format_instruction(row):
    return f"Instruction: {row['question']}\nResponse: {row['answer']}"

df['text'] = df.apply(format_instruction, axis=1)
```

## 🎛️ 模型配置

### 1. 量化設置 (4-bit)

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
```

### 2. LoRA 配置

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=16,                    # LoRA 秩
    lora_alpha=32,           # 縮放參數
    lora_dropout=0.05,       # Dropout 率
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "v_proj"],  # 目標模組
)
```

### 3. 訓練參數

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    weight_decay=0.001,
    bf16=True,               # 使用 BF16 精度
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    save_steps=100,
    logging_steps=25,
    report_to="tensorboard",
)
```

## 🚀 開始訓練

### 1. 執行訓練腳本

```bash
# 使用 Jupyter Notebook
jupyter notebook finetune_medgemma_27b_中文.ipynb

# 或使用 Python 腳本
python train_medgemma_27b.py
```

### 2. 監控訓練過程

```bash
# 監控 GPU 使用情況
nvidia-smi -l 1

# 啟動 TensorBoard
tensorboard --logdir ./results
```

### 3. 訓練檢查點

訓練過程中會自動保存：
- **模型權重**: LoRA 適配器
- **配置檔案**: 訓練參數和模型配置
- **日誌檔案**: 訓練過程記錄

## 📈 性能優化

### 1. 記憶體優化

- **批次大小**: 根據 GPU 記憶體調整
- **序列長度**: 控制輸入長度以節省記憶體
- **梯度累積**: 維持有效批次大小

### 2. 速度優化

- **混合精度**: 使用 BF16 加速訓練
- **梯度檢查點**: 節省記憶體
- **數據載入**: 優化數據管道

### 3. 常見問題解決

#### 記憶體不足 (OOM)
```python
# 減少批次大小
per_device_train_batch_size=2

# 減少序列長度
max_length=256

# 增加梯度累積
gradient_accumulation_steps=8
```

#### 訓練速度慢
```python
# 增加批次大小 (如果記憶體允許)
per_device_train_batch_size=8

# 使用更快的優化器
optim="adamw_torch"

# 減少日誌頻率
logging_steps=50
```

## 🧪 模型測試

### 1. 載入微調模型

```python
from peft import PeftModel

# 載入基礎模型
base_model = AutoModelForCausalLM.from_pretrained(
    "google/medgemma-4b-multimodal",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# 載入 LoRA 適配器
finetuned_model = PeftModel.from_pretrained(base_model, "./finetuned_medgemma_4b")
```

### 2. 生成醫療回答

```python
def generate_medical_response(question, max_length=256):
    test_input = f"Instruction: {question}\nResponse:"
    inputs = tokenizer(test_input, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = finetuned_model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("Response:")[-1].strip()

# 測試問題
questions = [
    "什麼是糖尿病的症狀?",
    "高血壓如何治療?",
    "心臟病的原因是什麼?",
]

for question in questions:
    answer = generate_medical_response(question)
    print(f"問題: {question}")
    print(f"回答: {answer}\n")
```

## 📁 專案結構

```
Fine-tune-MedGemma/
├── finetune_medgemma_27b_中文.ipynb    # 主要訓練腳本
├── train_medgemma_27b.py              # Python 訓練腳本
├── medquad.csv                        # 醫療問答數據集
├── README_中文.md                     # 中文說明文件
├── requirements.txt                    # 依賴套件清單
├── results/                           # 訓練結果目錄
│   ├── checkpoints/                   # 模型檢查點
│   └── logs/                         # 訓練日誌
└── finetuned_medgemma_4b/            # 微調模型輸出
    ├── adapter_config.json            # LoRA 配置
    ├── adapter_model.bin              # LoRA 權重
    └── training_config.json           # 訓練配置
```

## 🔧 故障排除

### 常見錯誤及解決方案

#### 1. CUDA 記憶體不足
```
RuntimeError: CUDA out of memory
```
**解決方案**:
- 減少 `per_device_train_batch_size`
- 減少 `max_length`
- 增加 `gradient_accumulation_steps`

#### 2. 模型存取被拒絕
```
401 Client Error: Unauthorized
```
**解決方案**:
- 確認已申請 MedGemma 模型存取權限
- 檢查 HuggingFace token 是否正確
- 重新執行 `login()` 函數

#### 3. 套件版本衝突
```
ImportError: cannot import name 'xxx'
```
**解決方案**:
- 更新到指定版本: `pip install transformers==4.44.0`
- 重新安裝衝突套件
- 使用虛擬環境隔離依賴

#### 4. 數據格式錯誤
```
KeyError: 'text'
```
**解決方案**:
- 檢查 CSV 檔案格式
- 確認欄位名稱正確
- 檢查數據預處理步驟

## 📊 性能基準

### 硬體配置建議

| GPU 型號 | VRAM | 批次大小 | 序列長度 | 訓練時間 |
|----------|------|----------|----------|----------|
| RTX 4090 | 24GB | 4 | 512 | ~2小時 |
| RTX 3090 | 24GB | 2 | 512 | ~4小時 |
| RTX 3080 | 10GB | 1 | 256 | ~8小時 |

### 記憶體使用情況

- **模型載入**: ~8GB
- **訓練過程**: ~18-22GB
- **緩衝區**: ~2-4GB

## ⚠️ 重要注意事項

### 1. 醫療免責聲明
- 此模型僅供教育和研究用途
- 醫療建議應由合格醫療專業人員驗證
- 不應直接用於臨床診斷

### 2. 數據隱私
- 確保使用適當的醫療數據
- 遵守相關隱私法規
- 注意數據去識別化

### 3. 模型限制
- 可能存在偏見和不準確性
- 需要持續評估和改進
- 建議加入安全檢查機制

## 🤝 貢獻指南

歡迎提交 Issue 和 Pull Request 來改進這個專案！

### 如何貢獻
1. Fork 這個專案
2. 創建功能分支
3. 提交變更
4. 發起 Pull Request

## 📄 授權

本專案僅供教育和研究用途。請尊重原始模型和數據集的授權條款。

## 📞 支援

如果您遇到問題或有疑問，請：
1. 查看本文件的故障排除部分
2. 搜尋現有的 Issue
3. 創建新的 Issue 並提供詳細資訊

---

**祝您訓練順利！** 🚀 