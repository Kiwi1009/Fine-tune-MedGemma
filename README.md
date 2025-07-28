# MedGemma 27B 中文微調專案

<div align="center">

![MedGemma Logo](https://img.shields.io/badge/MedGemma-27B-blue?style=for-the-badge&logo=google)
![Python](https://img.shields.io/badge/Python-3.8+-green?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?style=for-the-badge&logo=pytorch)
![GPU](https://img.shields.io/badge/GPU-RTX%204090-orange?style=for-the-badge&logo=nvidia)

**基於 MedGemma 27B 多模態模型的醫療問答微調專案**

[快速開始](#-快速開始) • [專案架構](#-專案架構) • [程式碼說明](#-程式碼說明) • [使用指南](#-使用指南) • [故障排除](#-故障排除)

</div>

---

## 📋 專案概述

本專案是一個完整的 MedGemma 27B 中文微調解決方案，專門針對醫療問答任務進行優化。專案採用 LoRA (Low-Rank Adaptation) 技術，在保持模型原有能力的同時，高效地適應醫療領域的特定需求。

### 🎯 主要特色

- **🖥️ GPU 優化**: 針對 RTX 4090 等高端顯卡進行記憶體優化
- **🔧 LoRA 微調**: 使用低秩適應技術，高效更新模型參數
- **🏥 醫療專用**: 基於 MedQuAD 醫療問答數據集訓練
- **📊 記憶體監控**: 內建 GPU 記憶體使用追蹤和優化建議
- **🌏 中文支援**: 完整的中文說明和錯誤處理
- **📈 性能基準**: 不同硬體配置的訓練時間預估

### ⚡ 技術亮點

- **4-bit 量化**: 大幅減少記憶體使用，支援更大模型
- **混合精度訓練**: 使用 BF16 精度加速訓練過程
- **梯度檢查點**: 優化記憶體使用，支援更長序列
- **動態批次處理**: 智能調整批次大小以適應記憶體限制

---

## 🏗️ 專案架構

```
Fine-tune-MedGemma/
├── 📁 核心訓練文件
│   ├── finetune_medgemma_27b_中文.ipynb    # Jupyter Notebook 版本
│   ├── train_medgemma_27b.py              # Python 腳本版本
│   └── config.json                        # 訓練配置文件
│
├── 📁 文檔說明
│   ├── README_中文.md                      # 詳細中文說明
│   ├── 快速開始指南.md                     # 簡潔快速指南
│   └── upload_to_github.md                # GitHub 上傳指南
│
├── 📁 依賴配置
│   ├── requirements.txt                    # Python 依賴套件
│   └── medquad.csv                        # 醫療問答數據集
│
└── 📁 輸出目錄 (訓練時生成)
    ├── ./results/                         # 訓練結果和日誌
    └── ./finetuned_medgemma_4b/          # 微調模型輸出
```

### 🔧 核心組件說明

#### 1. 訓練引擎 (`train_medgemma_27b.py`)
```python
class MedGemmaTrainer:
    """MedGemma 訓練器類 - 核心訓練邏輯"""
    
    def __init__(self, config):
        # 初始化訓練器
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def check_gpu(self):
        # GPU 可用性檢查和記憶體評估
        
    def authenticate_huggingface(self):
        # HuggingFace 認證和模型存取驗證
        
    def load_dataset(self):
        # 數據集載入和預處理
        
    def setup_model(self):
        # 模型載入和量化配置
        
    def setup_lora(self):
        # LoRA 配置和適配器設置
        
    def prepare_dataset(self):
        # 數據集 tokenization 和格式化
        
    def setup_training(self):
        # 訓練參數配置和 Trainer 初始化
        
    def train(self):
        # 執行訓練過程
        
    def save_model(self):
        # 保存微調模型和配置
        
    def test_model(self):
        # 模型測試和推理驗證
```

#### 2. 配置管理 (`config.json`)
```json
{
  "model_id": "google/medgemma-4b-multimodal",
  "data_path": "medquad.csv",
  "output_dir": "./results",
  "model_output_dir": "./finetuned_medgemma_4b",
  "sample_size": 2000,
  "batch_size": 4,
  "max_length": 512,
  "num_epochs": 1,
  "learning_rate": 2e-5,
  "lora_r": 16,
  "lora_alpha": 32,
  "lora_dropout": 0.05,
  "target_modules": ["q_proj", "v_proj"],
  "gradient_accumulation_steps": 4
}
```

#### 3. 數據處理流程
```python
# 數據預處理流程
def format_instruction(row):
    """將醫療問答格式化為指令-回應格式"""
    return f"Instruction: {row['question']}\nResponse: {row['answer']}"

# Tokenization 流程
def tokenize_function(examples):
    """對文本進行 tokenization，適用於因果語言建模"""
    texts = [text + tokenizer.eos_token for text in examples['text']]
    model_inputs = tokenizer(
        texts,
        truncation=True,
        padding=False,
        max_length=512,
        return_tensors=None
    )
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs
```

---

## 💻 程式碼說明

### 🔧 核心技術實現

#### 1. 模型量化配置
```python
# 4-bit 量化配置，大幅減少記憶體使用
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                    # 啟用 4-bit 量化
    bnb_4bit_use_double_quant=True,       # 雙重量化
    bnb_4bit_quant_type="nf4",           # 使用 NF4 量化類型
    bnb_4bit_compute_dtype=torch.bfloat16 # 計算精度
)
```

#### 2. LoRA 適配器配置
```python
# LoRA 配置，高效參數適應
lora_config = LoraConfig(
    r=16,                    # LoRA 秩，控制適配器大小
    lora_alpha=32,           # 縮放參數，影響學習率
    lora_dropout=0.05,       # Dropout 率，防止過擬合
    bias="none",             # 不更新偏置項
    task_type=TaskType.CAUSAL_LM,  # 因果語言建模任務
    target_modules=["q_proj", "v_proj"],  # 目標注意力模組
)
```

#### 3. 訓練參數優化
```python
# 針對 GPU 記憶體優化的訓練參數
training_args = TrainingArguments(
    output_dir="./results",                    # 輸出目錄
    num_train_epochs=1,                       # 訓練輪數
    per_device_train_batch_size=4,            # 批次大小
    gradient_accumulation_steps=4,            # 梯度累積步數
    learning_rate=2e-5,                       # 學習率
    weight_decay=0.001,                       # 權重衰減
    bf16=True,                                # 混合精度訓練
    max_grad_norm=0.3,                        # 梯度裁剪
    warmup_ratio=0.03,                        # 熱身比例
    group_by_length=True,                      # 按長度分組
    lr_scheduler_type="constant",             # 學習率調度器
    report_to="tensorboard",                  # 報告到 TensorBoard
)
```

#### 4. 記憶體優化策略
```python
# GPU 記憶體監控和優化
def check_gpu(self):
    """檢查 GPU 可用性和記憶體"""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory < 20:
            print("⚠️  警告: GPU 記憶體不足 20GB，建議調整批次大小")
        elif gpu_memory >= 24:
            print("✅ GPU 記憶體充足，可以使用預設設置")

# 梯度檢查點啟用
model.gradient_checkpointing_enable()

# 動態記憶體清理
torch.cuda.empty_cache()
```

### 📊 數據處理架構

#### 1. 數據載入和預處理
```python
def load_dataset(self):
    """載入和預處理 MedQuAD 數據集"""
    df = pd.read_csv(self.config['data_path'])
    
    # 格式化為指令-回應格式
    def format_instruction(row):
        return f"Instruction: {row['question']}\nResponse: {row['answer']}"
    
    df['text'] = df.apply(format_instruction, axis=1)
    
    # 取樣控制訓練時間
    sample_size = min(self.config.get('sample_size', 2000), len(df))
    df_sample = df.sample(n=sample_size, random_state=42)
    
    # 轉換為 HuggingFace Dataset
    self.dataset = Dataset.from_pandas(df_sample[['text']])
```

#### 2. Tokenization 和批次處理
```python
def prepare_dataset(self):
    """準備訓練數據集"""
    def tokenize_function(examples):
        texts = [text + self.tokenizer.eos_token for text in examples['text']]
        
        model_inputs = self.tokenizer(
            texts,
            truncation=True,
            padding=False,
            max_length=self.config.get('max_length', 512),
            return_tensors=None
        )
        
        # 設置標籤進行因果語言建模
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        return model_inputs
    
    self.tokenized_dataset = self.dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=self.dataset.column_names,
        desc="Tokenizing 數據集"
    )
```

### 🧪 模型測試和推理

#### 1. 模型載入和推理
```python
def test_model(self):
    """測試微調模型"""
    # 重新載入模型進行推理
    base_model = AutoModelForCausalLM.from_pretrained(
        self.config['model_id'],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    
    # 載入 LoRA 適配器
    finetuned_model = PeftModel.from_pretrained(base_model, output_dir)
    finetuned_model.eval()
    
    # 測試推理
    test_questions = [
        "糖尿病的症狀是什麼?",
        "高血壓如何治療?",
        "心臟病的原因是什麼?",
    ]
    
    for question in test_questions:
        test_input = f"Instruction: {question}\nResponse:"
        inputs = self.tokenizer(test_input, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = finetuned_model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_response = response.split("Response:")[-1].strip()
```

---

## 🚀 快速開始

### 📋 系統需求

#### 硬體需求
- **GPU**: NVIDIA GPU，至少 24GB VRAM (推薦 RTX 4090 或 A100)
- **記憶體**: 至少 32GB 系統記憶體
- **儲存**: 至少 50GB 可用空間

#### 軟體需求
- **作業系統**: Windows 10/11, Linux, 或 macOS
- **Python**: 3.8 或更高版本
- **CUDA**: 11.8 或更高版本 (與 PyTorch 相容)

### ⚡ 快速安裝

#### 1. 環境準備
```bash
# 創建虛擬環境
conda create -n medgemma python=3.10
conda activate medgemma

# 或使用 venv
python -m venv medgemma_env
source medgemma_env/bin/activate  # Linux/Mac
medgemma_env\Scripts\activate     # Windows
```

#### 2. 安裝依賴
```bash
# 安裝 PyTorch (根據您的 CUDA 版本)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安裝其他依賴
pip install -r requirements.txt
```

#### 3. 驗證安裝
```python
import torch
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU 型號: {torch.cuda.get_device_name(0)}")
    print(f"GPU 記憶體: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
```

### 🔐 模型存取設置

#### 1. 申請 MedGemma 存取權限
1. 前往 [MedGemma 模型頁面](https://huggingface.co/google/medgemma-4b-multimodal)
2. 點擊 "Request Access" 申請存取權限
3. 接受授權協議
4. 等待審核通過 (通常需要幾分鐘到幾小時)

#### 2. 取得 HuggingFace Token
1. 前往 [Token 設置頁面](https://huggingface.co/settings/tokens)
2. 點擊 "New token"
3. 選擇 "Read" 權限
4. 複製生成的 token

#### 3. 設置環境變數
```bash
# Linux/Mac
export HUGGINGFACE_HUB_TOKEN="your_token_here"

# Windows
set HUGGINGFACE_HUB_TOKEN=your_token_here
```

### 🚀 開始訓練

#### 方法 1: 使用 Jupyter Notebook
```bash
jupyter notebook finetune_medgemma_27b_中文.ipynb
```

#### 方法 2: 使用 Python 腳本
```bash
# 使用默認配置
python train_medgemma_27b.py

# 使用自定義配置
python train_medgemma_27b.py --config config.json

# 調整參數
python train_medgemma_27b.py --batch_size 2 --sample_size 1000 --num_epochs 2
```

---

## 📊 使用指南

### 🎛️ 配置參數說明

#### 模型配置
| 參數 | 說明 | 預設值 | 建議範圍 |
|------|------|--------|----------|
| `model_id` | 基礎模型 ID | `google/medgemma-4b-multimodal` | - |
| `sample_size` | 訓練樣本數量 | 2000 | 1000-5000 |
| `batch_size` | 批次大小 | 4 | 1-8 |
| `max_length` | 最大序列長度 | 512 | 256-1024 |

#### LoRA 配置
| 參數 | 說明 | 預設值 | 建議範圍 |
|------|------|--------|----------|
| `lora_r` | LoRA 秩 | 16 | 8-32 |
| `lora_alpha` | 縮放參數 | 32 | 16-64 |
| `lora_dropout` | Dropout 率 | 0.05 | 0.01-0.1 |

#### 訓練配置
| 參數 | 說明 | 預設值 | 建議範圍 |
|------|------|--------|----------|
| `learning_rate` | 學習率 | 2e-5 | 1e-5 - 5e-5 |
| `num_epochs` | 訓練輪數 | 1 | 1-5 |
| `gradient_accumulation_steps` | 梯度累積步數 | 4 | 2-8 |

### 📈 性能基準

#### 硬體配置建議
| GPU 型號 | VRAM | 批次大小 | 序列長度 | 訓練時間 | 記憶體使用 |
|----------|------|----------|----------|----------|------------|
| RTX 4090 | 24GB | 4 | 512 | ~2小時 | 18-22GB |
| RTX 3090 | 24GB | 2 | 512 | ~4小時 | 16-20GB |
| RTX 3080 | 10GB | 1 | 256 | ~8小時 | 8-12GB |

#### 記憶體使用分析
- **模型載入**: ~8GB (4-bit 量化)
- **訓練過程**: ~18-22GB (包含梯度、優化器狀態)
- **緩衝區**: ~2-4GB (數據載入和緩存)

### 🧪 模型測試

#### 1. 載入微調模型
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# 載入基礎模型
base_model = AutoModelForCausalLM.from_pretrained("google/medgemma-4b-multimodal")
tokenizer = AutoTokenizer.from_pretrained("google/medgemma-4b-multimodal")

# 載入 LoRA 適配器
finetuned_model = PeftModel.from_pretrained(base_model, "./finetuned_medgemma_4b")
```

#### 2. 生成醫療回答
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

# 測試
questions = [
    "糖尿病的症狀是什麼?",
    "高血壓如何治療?",
    "心臟病的原因是什麼?",
]

for question in questions:
    answer = generate_medical_response(question)
    print(f"問題: {question}")
    print(f"回答: {answer}\n")
```

---

## 🔧 故障排除

### 常見錯誤及解決方案

#### 1. CUDA 記憶體不足
```
RuntimeError: CUDA out of memory
```

**解決方案**:
```python
# 減少批次大小
per_device_train_batch_size=2

# 減少序列長度
max_length=256

# 增加梯度累積
gradient_accumulation_steps=8

# 啟用梯度檢查點
model.gradient_checkpointing_enable()
```

#### 2. 模型存取被拒絕
```
401 Client Error: Unauthorized
```

**解決方案**:
1. 確認已申請 MedGemma 模型存取權限
2. 檢查 HuggingFace token 是否正確
3. 重新執行認證流程

#### 3. 套件版本衝突
```
ImportError: cannot import name 'xxx'
```

**解決方案**:
```bash
# 更新到指定版本
pip install transformers==4.44.0

# 重新安裝衝突套件
pip install --force-reinstall peft

# 使用虛擬環境
conda create -n medgemma python=3.10
```

#### 4. 數據格式錯誤
```
KeyError: 'text'
```

**解決方案**:
1. 檢查 CSV 檔案格式
2. 確認欄位名稱正確
3. 檢查數據預處理步驟

### 🔍 調試技巧

#### 1. GPU 記憶體監控
```bash
# 即時監控 GPU 使用情況
nvidia-smi -l 1

# 在 Python 中檢查記憶體
import torch
print(f"已分配: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"已保留: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
```

#### 2. 訓練過程監控
```bash
# 啟動 TensorBoard
tensorboard --logdir ./results

# 查看訓練日誌
tail -f ./results/trainer_state.json
```

#### 3. 模型驗證
```python
# 檢查模型參數
print(f"可訓練參數: {trainable_params:,}")
print(f"總參數: {all_params:,}")
print(f"參數比例: {100 * trainable_params / all_params:.2f}%")

# 檢查數據集
print(f"數據集大小: {len(dataset)}")
print(f"樣本範例: {dataset[0]}")
```

---

## 📈 性能優化

### 🎯 記憶體優化策略

#### 1. 量化優化
```python
# 使用 4-bit 量化
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
```

#### 2. 梯度優化
```python
# 啟用梯度檢查點
model.gradient_checkpointing_enable()

# 設置梯度累積
gradient_accumulation_steps=4

# 梯度裁剪
max_grad_norm=0.3
```

#### 3. 批次處理優化
```python
# 動態填充
pad_to_multiple_of=8

# 按長度分組
group_by_length=True

# 混合精度訓練
bf16=True
```

### ⚡ 速度優化策略

#### 1. 數據載入優化
```python
# 設置數據載入器
dataloader_pin_memory=False
remove_unused_columns=False

# 使用多進程載入
num_workers=4
```

#### 2. 模型優化
```python
# 使用更快的優化器
optim="paged_adamw_32bit"

# 減少日誌頻率
logging_steps=25
```

#### 3. 硬體優化
```python
# 使用 BF16 精度
bf16=True

# 啟用 CUDA 圖
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

---

## ⚠️ 重要注意事項

### 🏥 醫療免責聲明

- **教育用途**: 此模型僅供教育和研究用途
- **專業驗證**: 醫療建議應由合格醫療專業人員驗證
- **臨床限制**: 不應直接用於臨床診斷

### 🔒 數據隱私

- **適當使用**: 確保使用適當的醫療數據
- **法規遵守**: 遵守相關隱私法規
- **去識別化**: 注意數據去識別化處理

### 🎯 模型限制

- **偏見風險**: 可能存在偏見和不準確性
- **持續評估**: 需要持續評估和改進
- **安全檢查**: 建議加入安全檢查機制

---

## 🤝 貢獻指南

歡迎提交 Issue 和 Pull Request 來改進這個專案！

### 如何貢獻

1. **Fork 專案**: 在 GitHub 上 Fork 此專案
2. **創建分支**: 創建功能分支 (`git checkout -b feature/AmazingFeature`)
3. **提交變更**: 提交您的變更 (`git commit -m 'Add some AmazingFeature'`)
4. **推送分支**: 推送到分支 (`git push origin feature/AmazingFeature`)
5. **發起 PR**: 開啟 Pull Request

### 貢獻指南

- 請確保代碼符合 PEP 8 規範
- 添加適當的註釋和文檔
- 測試您的更改
- 更新相關文檔

---

## 📄 授權

本專案僅供教育和研究用途。請尊重原始模型和數據集的授權條款。

- **MedGemma 模型**: 遵循 Google 的授權條款
- **MedQuAD 數據集**: 遵循原始數據集的授權條款
- **本專案**: MIT License

---

## 📞 支援

如果您遇到問題或有疑問，請：

1. **查看文檔**: 仔細閱讀本 README 和相關文檔
2. **搜尋 Issue**: 在 GitHub Issues 中搜尋類似問題
3. **創建 Issue**: 如果沒有找到解決方案，請創建新的 Issue
4. **提供資訊**: 在 Issue 中提供詳細的錯誤信息和環境配置

### 聯繫方式

- **GitHub Issues**: [創建 Issue](https://github.com/Kiwi1009/Fine-tune-MedGemma/issues)
- **Email**: 通過 GitHub 個人資料聯繫

---

<div align="center">

**⭐ 如果這個專案對您有幫助，請給我們一個 Star！**

**🚀 祝您訓練順利！**

</div> 