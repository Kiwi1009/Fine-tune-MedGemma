#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
創建修正版的 MedGemma 27B 中文微調 Jupyter Notebook
"""

import json

# 創建 notebook 結構
notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# MedGemma 27B 中文微調指南\n\n",
                "## 🎯 專案目標\n\n",
                "本腳本展示如何使用 LoRA (Low-Rank Adaptation) 技術在醫療問答數據上微調 MedGemma 27B 多模態模型。\n",
                "專案針對 GPU 訓練進行了優化，特別適合 RTX 4090 等高端顯卡。\n\n",
                "## 📋 主要功能\n\n",
                "- **GPU 加速訓練**: 支援 CUDA GPU 加速，優化記憶體使用\n",
                "- **LoRA 微調**: 高效參數適應，無需完整模型重訓練\n",
                "- **醫療專用**: 基於 MedQuAD 醫療問答數據集訓練\n",
                "- **記憶體監控**: 內建 GPU 記憶體使用追蹤和優化建議\n",
                "- **中文支援**: 完整的中文說明和錯誤處理\n\n",
                "## ⚠️ 重要提醒\n\n",
                "⚠️ **醫療免責聲明**: 此模型僅供教育和研究用途。醫療建議應由合格醫療專業人員驗證。"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. 環境設置與套件安裝\n\n",
                "首先安裝所有必要的套件並檢查 GPU 可用性。"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 安裝必要套件\n",
                "%pip install -q torch transformers peft bitsandbytes datasets pandas accelerate huggingface_hub\n",
                "%pip install -q -U transformers==4.44.0\n\n",
                "print(\"✅ 套件安裝完成!\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 導入必要套件\n",
                "import torch\n",
                "import pandas as pd\n",
                "import os\n",
                "import warnings\n",
                "import json\n",
                "from transformers import (\n",
                "    AutoModelForCausalLM,\n",
                "    AutoTokenizer,\n",
                "    BitsAndBytesConfig,\n",
                "    TrainingArguments,\n",
                "    Trainer,\n",
                "    DataCollatorForLanguageModeling,\n",
                "    pipeline\n",
                ")\n",
                "from peft import LoraConfig, get_peft_model, TaskType, PeftModel, prepare_model_for_kbit_training\n",
                "from datasets import Dataset\n",
                "from huggingface_hub import login, HfApi\n\n",
                "warnings.filterwarnings('ignore')\n\n",
                "print(\"✅ 套件導入完成!\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. HuggingFace 認證設置\n\n",
                "MedGemma 是一個需要授權的模型，需要先進行 HuggingFace 認證。"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 🔐 MedGemma 認證設置\n",
                "print(\"🔐 MedGemma 認證設置\")\n",
                "print(\"=\" * 50)\n",
                "print(\"MedGemma 是一個需要授權的模型，請按照以下步驟進行認證:\")\n",
                "print(\"1. 前往 https://huggingface.co/google/medgemma-27b-multimodal\")\n",
                "print(\"2. 點擊 'Request Access' 並接受授權協議\")\n",
                "print(\"3. 等待審核通過 (通常需要幾分鐘到幾小時)\")\n",
                "print(\"4. 從 https://huggingface.co/settings/tokens 取得您的 token\")\n",
                "print(\"5. 使用以下方法之一進行認證\")\n",
                "print(\"=\" * 50)\n\n",
                "# 方法 1: 互動式登入 (推薦 - 會提示輸入 token)\n",
                "# from huggingface_hub import login\n",
                "# login()\n\n",
                "# 方法 2: 直接 token 登入 (替換為您的實際 token)\n",
                "# from huggingface_hub import login\n",
                "# login(token=\"hf_your_token_here\")\n\n",
                "# 方法 3: 環境變數 (將 token 設為環境變數)\n",
                "# import os\n",
                "# os.environ[\"HUGGINGFACE_HUB_TOKEN\"] = \"hf_your_token_here\"\n\n",
                "print(\"\\n⚠️  重要: 請取消註解並執行上述認證方法之一!\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# GPU 檢查和設置\n",
                "if torch.cuda.is_available():\n",
                "    print(f\"✅ GPU 可用: {torch.cuda.get_device_name(0)}\")\n",
                "    print(f\"GPU 記憶體: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB\")\n",
                "    print(f\"CUDA 版本: {torch.version.cuda}\")\n",
                "    \n",
                "    # 檢查記憶體是否足夠\n",
                "    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3\n",
                "    if gpu_memory < 20:\n",
                "        print(\"⚠️  警告: GPU 記憶體不足 20GB，可能需要調整批次大小\")\n",
                "    elif gpu_memory >= 24:\n",
                "        print(\"✅ GPU 記憶體充足，可以使用預設設置\")\n",
                "    else:\n",
                "        print(\"⚡ GPU 記憶體適中，建議監控使用情況\")\n",
                "else:\n",
                "    print(\"❌ 警告: 未檢測到 GPU! 此腳本需要 GPU 才能運行。\")\n",
                "    \n",
                "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
                "print(f\"使用設備: {device}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. 載入和預處理數據集\n\n",
                "載入 MedQuAD.csv 數據集並格式化為指令跟隨微調格式。"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 載入數據集\n",
                "print(\"📊 載入 MedQuAD 數據集...\")\n",
                "df = pd.read_csv('medquad.csv')\n\n",
                "print(f\"數據集形狀: {df.shape}\")\n",
                "print(f\"欄位: {df.columns.tolist()}\")\n",
                "print(\"\\n前 3 行數據:\")\n",
                "print(df.head(3))\n\n",
                "print(\"\\n缺失值統計:\")\n",
                "print(df.isnull().sum())\n\n",
                "print(\"\\n數據來源統計:\")\n",
                "print(df['source'].value_counts())\n",
                "print(\"\\n專注領域統計:\")\n",
                "print(df['focus_area'].value_counts())"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 格式化數據為指令-回應格式\n",
                "def format_instruction(row):\n",
                "    \"\"\"將問題和答案格式化為 MedGemma 的指令-回應格式\"\"\"\n",
                "    instruction = f\"Instruction: {row['question']}\\nResponse: {row['answer']}\"\n",
                "    return instruction\n\n",
                "df['text'] = df.apply(format_instruction, axis=1)\n\n",
                "# 取樣數據以控制訓練時間 (可根據 GPU 記憶體調整)\n",
                "df_sample = df.sample(n=min(2000, len(df)), random_state=42)\n",
                "print(f\"使用 {len(df_sample)} 個樣本進行微調\")\n\n",
                "print(\"\\n格式化文本範例:\")\n",
                "print(df_sample['text'].iloc[0][:500] + \"...\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. 配置模型 (4-bit 量化)\n\n",
                "載入 MedGemma 27B 並使用 4-bit 量化以優化 RTX 4090 的記憶體使用。"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 模型 ID\n",
                "model_id = \"google/medgemma-27b-multimodal\"\n\n",
                "# 檢查用戶是否已認證\n",
                "try:\n",
                "    api = HfApi()\n",
                "    # 嘗試獲取模型資訊 - 如果未認證會失敗\n",
                "    model_info = api.model_info(model_id)\n",
                "    print(\"✅ 認證成功! 模型存取已確認。\")\n",
                "except Exception as e:\n",
                "    print(\"❌ 認證失敗!\")\n",
                "    print(f\"錯誤: {e}\")\n",
                "    print(\"\\n🔧 快速修復:\")\n",
                "    print(\"在新單元格中運行:\")\n",
                "    print(\"from huggingface_hub import login\")\n",
                "    print(\"login()  # 這會提示您輸入 token\")\n",
                "    print(\"\\n或直接設置您的 token:\")\n",
                "    print(\"login(token='your_huggingface_token_here')\")\n",
                "    print(\"\\n然後重新運行此單元格。\")\n",
                "    raise Exception(\"請先進行 HuggingFace 認證!\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 配置 4-bit 量化\n",
                "bnb_config = BitsAndBytesConfig(\n",
                "    load_in_4bit=True,\n",
                "    bnb_4bit_use_double_quant=True,\n",
                "    bnb_4bit_quant_type=\"nf4\",\n",
                "    bnb_4bit_compute_dtype=torch.bfloat16\n",
                ")\n\n",
                "print(\"載入 tokenizer...\")\n",
                "try:\n",
                "    tokenizer = AutoTokenizer.from_pretrained(model_id)\n",
                "    tokenizer.pad_token = tokenizer.eos_token\n",
                "    tokenizer.padding_side = \"right\"\n",
                "    print(\"✅ Tokenizer 載入成功!\")\n",
                "except Exception as e:\n",
                "    print(f\"❌ Tokenizer 載入失敗: {e}\")\n",
                "    print(\"這通常表示需要認證。\")\n",
                "    raise e\n\n",
                "print(\"載入模型 (4-bit 量化)...\")\n",
                "try:\n",
                "    model = AutoModelForCausalLM.from_pretrained(\n",
                "        model_id,\n",
                "        quantization_config=bnb_config,\n",
                "        device_map=\"auto\",\n",
                "        trust_remote_code=True,\n",
                "        torch_dtype=torch.bfloat16,\n",
                "    )\n",
                "    print(\"✅ 模型載入成功!\")\n",
                "except Exception as e:\n",
                "    print(f\"❌ 模型載入失敗: {e}\")\n",
                "    print(\"常見解決方案:\")\n",
                "    print(\"1. 確認您已進行 HuggingFace 認證\")\n",
                "    print(\"2. 檢查您是否有 MedGemma 模型的存取權限\")\n",
                "    print(\"3. 驗證您的網路連線\")\n",
                "    raise e\n\n",
                "if torch.cuda.is_available():\n",
                "    print(f\"\\n📊 GPU 記憶體使用情況:\")\n",
                "    print(f\"  已分配: {torch.cuda.memory_allocated() / 1024**3:.2f} GB\")\n",
                "    print(f\"  已保留: {torch.cuda.memory_reserved() / 1024**3:.2f} GB\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. 配置 LoRA\n\n",
                "設置 LoRA (Low-Rank Adaptation) 參數以啟用高效微調。"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 配置 LoRA\n",
                "lora_config = LoraConfig(\n",
                "    r=16,                    # LoRA 秩\n",
                "    lora_alpha=32,           # Alpha 參數\n",
                "    lora_dropout=0.05,       # Dropout 率\n",
                "    bias=\"none\",\n",
                "    task_type=TaskType.CAUSAL_LM,\n",
                "    target_modules=[\"q_proj\", \"v_proj\"],  # 目標注意力模組\n",
                ")\n\n",
                "# 啟用梯度檢查點以節省記憶體\n",
                "model.gradient_checkpointing_enable()\n\n",
                "# 準備模型進行 k-bit 訓練\n",
                "model = prepare_model_for_kbit_training(model)\n\n",
                "# 獲取 PEFT 模型\n",
                "model = get_peft_model(model, lora_config)\n\n",
                "# 打印可訓練參數\n",
                "trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)\n",
                "all_params = sum(p.numel() for p in model.parameters())\n",
                "print(f\"可訓練參數: {trainable_params:,} ({100 * trainable_params / all_params:.2f}%)\")\n",
                "print(f\"總參數: {all_params:,}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. 準備訓練數據集\n\n",
                "手動對數據集進行 tokenization 並準備用於標準 Trainer 的訓練。"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 轉換為 HuggingFace Dataset\n",
                "dataset = Dataset.from_pandas(df_sample[['text']])\n\n",
                "def tokenize_function(examples):\n",
                "    \"\"\"對文本進行 tokenization，適用於因果語言建模\"\"\"\n",
                "    texts = [text + tokenizer.eos_token for text in examples['text']]\n",
                "    \n",
                "    model_inputs = tokenizer(\n",
                "        texts,\n",
                "        truncation=True,\n",
                "        padding=False,\n",
                "        max_length=512,  # 可根據 GPU 記憶體調整\n",
                "        return_tensors=None\n",
                "    )\n",
                "    \n",
                "    model_inputs[\"labels\"] = model_inputs[\"input_ids\"].copy()\n",
                "    \n",
                "    return model_inputs\n\n",
                "tokenized_dataset = dataset.map(\n",
                "    tokenize_function,\n",
                "    batched=True,\n",
                "    remove_columns=dataset.column_names,\n",
                "    desc=\"Tokenizing 數據集\"\n",
                ")\n\n",
                "print(f\"Tokenized 數據集: {tokenized_dataset}\")\n",
                "print(f\"樣本 tokenized 範例長度: {len(tokenized_dataset[0]['input_ids'])}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. 配置訓練參數\n\n",
                "設置針對 RTX 4090 24GB VRAM 優化的訓練參數。"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 訓練參數\n",
                "training_args = TrainingArguments(\n",
                "    output_dir=\"./results\",\n",
                "    num_train_epochs=1,\n",
                "    per_device_train_batch_size=4,  # 可根據 GPU 記憶體調整\n",
                "    gradient_accumulation_steps=4,\n",
                "    optim=\"paged_adamw_32bit\",\n",
                "    save_steps=100,\n",
                "    logging_steps=25,\n",
                "    learning_rate=2e-5,\n",
                "    weight_decay=0.001,\n",
                "    bf16=True,  # 使用 BF16 精度\n",
                "    max_grad_norm=0.3,\n",
                "    max_steps=-1,\n",
                "    warmup_ratio=0.03,\n",
                "    group_by_length=True,\n",
                "    lr_scheduler_type=\"constant\",\n",
                "    report_to=\"tensorboard\",\n",
                "    save_total_limit=2,\n",
                "    dataloader_pin_memory=False,\n",
                "    remove_unused_columns=False,\n",
                ")\n\n",
                "print(\"訓練參數配置:\")\n",
                "print(f\"批次大小: {training_args.per_device_train_batch_size}\")\n",
                "print(f\"梯度累積步數: {training_args.gradient_accumulation_steps}\")\n",
                "print(f\"有效批次大小: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}\")\n",
                "print(f\"學習率: {training_args.learning_rate}\")\n",
                "print(f\"訓練輪數: {training_args.num_train_epochs}\")\n",
                "print(f\"混合精度: BF16 = {training_args.bf16}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 8. 初始化數據收集器和標準 Trainer\n\n",
                "使用標準 Trainer 和 DataCollatorForLanguageModeling 進行因果語言建模。"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 初始化數據收集器\n",
                "data_collator = DataCollatorForLanguageModeling(\n",
                "    tokenizer=tokenizer,\n",
                "    mlm=False,  # 確保我們進行因果語言建模，而不是掩碼語言建模\n",
                "    pad_to_multiple_of=8  # 優化現代 GPU 的張量操作\n",
                ")\n\n",
                "# 初始化 Trainer\n",
                "trainer = Trainer(\n",
                "    model=model,\n",
                "    args=training_args,\n",
                "    train_dataset=tokenized_dataset,\n",
                "    tokenizer=tokenizer,\n",
                "    data_collator=data_collator,\n",
                ")\n\n",
                "if torch.cuda.is_available():\n",
                "    print(f\"訓練前 GPU 記憶體:\")\n",
                "    print(f\"  已分配: {torch.cuda.memory_allocated() / 1024**3:.2f} GB\")\n",
                "    print(f\"  已保留: {torch.cuda.memory_reserved() / 1024**3:.2f} GB\")\n",
                "    print(f\"  可用: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1024**3:.2f} GB\")\n\n",
                "print(\"\\nTrainer 已使用標準 Trainer 類初始化\")\n",
                "print(\"注意: 訓練需要一些時間。可在另一個終端中使用 'nvidia-smi' 監控 GPU 記憶體使用情況。\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 9. 開始訓練\n\n",
                "現在開始使用標準 Trainer 進行訓練過程。"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 開始訓練\n",
                "try:\n",
                "    print(\"🚀 開始訓練...\")\n",
                "    trainer.train()\n",
                "    print(\"✅ 訓練成功完成!\")\n",
                "except Exception as e:\n",
                "    print(f\"❌ 訓練失敗，錯誤: {e}\")\n",
                "    print(\"這可能是由於:\")\n",
                "    print(\"1. GPU 記憶體不足 - 嘗試減少 batch_size 或 max_length\")\n",
                "    print(\"2. 模型存取問題 - 確保您有 MedGemma 模型的存取權限\")\n",
                "    print(\"3. CUDA 相容性問題 - 檢查您的 PyTorch 和 CUDA 版本\")\n",
                "    \n",
                "if torch.cuda.is_available():\n",
                "    print(f\"\\n訓練後 GPU 記憶體:\")\n",
                "    print(f\"  已分配: {torch.cuda.memory_allocated() / 1024**3:.2f} GB\")\n",
                "    print(f\"  已保留: {torch.cuda.memory_reserved() / 1024**3:.2f} GB\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 10. 保存微調模型\n\n",
                "保存 LoRA 適配器和完整的微調模型。"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 保存微調模型\n",
                "output_dir = \"./finetuned_medgemma_27b\"\n\n",
                "model.save_pretrained(output_dir)\n",
                "tokenizer.save_pretrained(output_dir)\n\n",
                "print(f\"微調模型已保存到: {output_dir}\")\n",
                "print(\"內容:\")\n",
                "for file in os.listdir(output_dir):\n",
                "    print(f\"  - {file}\")\n\n",
                "# 保存訓練配置\n",
                "training_config = {\n",
                "    \"model_id\": model_id,\n",
                "    \"trainer_type\": \"normal_trainer\",\n",
                "    \"lora_config\": {\n",
                "        \"r\": lora_config.r,\n",
                "        \"lora_alpha\": lora_config.lora_alpha,\n",
                "        \"lora_dropout\": lora_config.lora_dropout,\n",
                "        \"target_modules\": lora_config.target_modules,\n",
                "    },\n",
                "    \"training_args\": {\n",
                "        \"learning_rate\": training_args.learning_rate,\n",
                "        \"num_train_epochs\": training_args.num_train_epochs,\n",
                "        \"per_device_train_batch_size\": training_args.per_device_train_batch_size,\n",
                "        \"gradient_accumulation_steps\": training_args.gradient_accumulation_steps,\n",
                "    },\n",
                "    \"dataset_size\": len(tokenized_dataset),\n",
                "}\n\n",
                "with open(os.path.join(output_dir, \"training_config.json\"), \"w\") as f:\n",
                "    json.dump(training_config, f, indent=2)\n\n",
                "print(\"\\n訓練配置已保存到 training_config.json\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 11. 測試微調模型\n\n",
                "通過在數據集的樣本問題上運行推理來測試微調模型。"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 清理 GPU 記憶體\n",
                "torch.cuda.empty_cache()\n\n",
                "print(\"載入微調模型進行推理...\")\n\n",
                "# 載入基礎模型\n",
                "base_model = AutoModelForCausalLM.from_pretrained(\n",
                "    model_id,\n",
                "    quantization_config=bnb_config,\n",
                "    device_map=\"auto\",\n",
                "    trust_remote_code=True,\n",
                "    torch_dtype=torch.bfloat16,\n",
                ")\n\n",
                "# 載入微調的 LoRA 適配器\n",
                "finetuned_model = PeftModel.from_pretrained(base_model, output_dir)\n\n",
                "# 設置為評估模式\n",
                "finetuned_model.eval()\n\n",
                "print(\"微調模型載入成功!\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 測試樣本問題\n",
                "sample_question = df_sample['question'].iloc[0]\n",
                "print(f\"樣本問題: {sample_question}\")\n",
                "print(\"\\n\" + \"=\"*50)\n\n",
                "test_input = f\"Instruction: {sample_question}\\nResponse:\"\n\n",
                "inputs = tokenizer(test_input, return_tensors=\"pt\").to(device)\n\n",
                "print(\"生成回應...\")\n",
                "with torch.no_grad():\n",
                "    outputs = finetuned_model.generate(\n",
                "        **inputs,\n",
                "        max_new_tokens=256,\n",
                "        do_sample=True,\n",
                "        temperature=0.7,\n",
                "        top_p=0.9,\n",
                "        pad_token_id=tokenizer.eos_token_id,\n",
                "        eos_token_id=tokenizer.eos_token_id,\n",
                "    )\n\n",
                "response = tokenizer.decode(outputs[0], skip_special_tokens=True)\n\n",
                "if \"Response:\" in response:\n",
                "    generated_response = response.split(\"Response:\")[-1].strip()\n",
                "else:\n",
                "    generated_response = response\n\n",
                "print(f\"生成回應: {generated_response}\")\n",
                "print(\"\\n\" + \"=\"*50)\n\n",
                "original_answer = df_sample[df_sample['question'] == sample_question]['answer'].iloc[0]\n",
                "print(f\"原始答案: {original_answer}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 12. 記憶體使用監控\n\n",
                "檢查最終的 GPU 記憶體使用情況並提供優化建議。"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 檢查最終 GPU 記憶體使用情況\n",
                "if torch.cuda.is_available():\n",
                "    allocated = torch.cuda.memory_allocated() / 1024**3\n",
                "    reserved = torch.cuda.memory_reserved() / 1024**3\n",
                "    total = torch.cuda.get_device_properties(0).total_memory / 1024**3\n",
                "    \n",
                "    print(\"最終 GPU 記憶體使用情況:\")\n",
                "    print(f\"  已分配: {allocated:.2f} GB\")\n",
                "    print(f\"  已保留: {reserved:.2f} GB\")\n",
                "    print(f\"  總 GPU 記憶體: {total:.2f} GB\")\n",
                "    print(f\"  記憶體利用率: {(reserved/total)*100:.1f}%\")\n",
                "    \n",
                "    print(\"\\n記憶體優化建議:\")\n",
                "    if reserved > 20:\n",
                "        print(\"⚠️  檢測到高記憶體使用 (>20GB)。建議:\")\n",
                "        print(\"   - 將 batch_size 從 4 減少到 2\")\n",
                "        print(\"   - 將 max_length 從 512 減少到 256\")\n",
                "        print(\"   - 使用 gradient_accumulation_steps=8 以維持有效批次大小\")\n",
                "    elif reserved > 15:\n",
                "        print(\"⚡ 良好的記憶體使用 (15-20GB)。當前設置是最佳的。\")\n",
                "    else:\n",
                "        print(\"✅ 低記憶體使用 (<15GB)。您可以:\")\n",
                "        print(\"   - 將 batch_size 增加到 8 以加快訓練\")\n",
                "        print(\"   - 將 max_length 增加到 1024 以處理更長序列\")\n",
                "        print(\"   - 使用更大的數據集樣本\")\n",
                "        \n",
                "    print(\"\\n對於即時監控，在終端中運行此命令:\")\n",
                "    print(\"nvidia-smi -l 1\")\n",
                "    \n",
                "else:\n",
                "    print(\"未檢測到 GPU。此腳本需要 CUDA 相容的 GPU。\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 13. 額外測試功能\n\n",
                "這裡有一些用於使用自定義問題測試模型的額外功能。"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def generate_medical_response(question, model=finetuned_model, tokenizer=tokenizer, max_length=256):\n",
                "    \"\"\"\n",
                "    使用微調模型為給定問題生成醫療回應。\n",
                "    \n",
                "    參數:\n",
                "        question (str): 要回答的醫療問題\n",
                "        model: 微調模型\n",
                "        tokenizer: tokenizer\n",
                "        max_length (int): 回應的最大長度\n",
                "    \n",
                "    返回:\n",
                "        str: 生成的醫療回應\n",
                "    \"\"\"\n",
                "    test_input = f\"Instruction: {question}\\nResponse:\"\n",
                "    \n",
                "    inputs = tokenizer(test_input, return_tensors=\"pt\").to(device)\n",
                "    \n",
                "    with torch.no_grad():\n",
                "        outputs = model.generate(\n",
                "            **inputs,\n",
                "            max_new_tokens=max_length,\n",
                "            do_sample=True,\n",
                "            temperature=0.7,\n",
                "            top_p=0.9,\n",
                "            pad_token_id=tokenizer.eos_token_id,\n",
                "            eos_token_id=tokenizer.eos_token_id,\n",
                "        )\n",
                "    \n",
                "    response = tokenizer.decode(outputs[0], skip_special_tokens=True)\n",
                "    if \"Response:\" in response:\n",
                "        generated_response = response.split(\"Response:\")[-1].strip()\n",
                "    else:\n",
                "        generated_response = response\n",
                "    \n",
                "    return generated_response\n\n",
                "# 測試自定義問題\n",
                "test_questions = [\n",
                "    \"糖尿病的症狀是什麼?\",\n",
                "    \"高血壓如何治療?\",\n",
                "    \"心臟病的原因是什麼?\",\n",
                "]\n\n",
                "print(\"使用自定義問題進行測試:\")\n",
                "print(\"=\"*60)\n\n",
                "for i, question in enumerate(test_questions, 1):\n",
                "    print(f\"\\n測試 {i}:\")\n",
                "    print(f\"問題: {question}\")\n",
                "    response = generate_medical_response(question)\n",
                "    print(f\"回應: {response}\")\n",
                "    print(\"-\" * 40)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 🎉 標準 TRAINER 完成!\n\n",
                "print(\"✅ 標準 Trainer 實現完成!\")\n",
                "print(\"=\" * 50)\n",
                "print(\"📋 您已完成的工作:\")\n",
                "print(\"1. ✅ 環境設置和認證\")\n",
                "print(\"2. ✅ 數據集載入和預處理\")\n",
                "print(\"3. ✅ 具有 4-bit 量化的模型配置\")\n",
                "print(\"4. ✅ LoRA 配置\")\n",
                "print(\"5. ✅ 標準 Trainer 的手動 tokenization\")\n",
                "print(\"6. ✅ 訓練參數優化\")\n",
                "print(\"7. ✅ 數據收集器和 trainer 初始化\")\n",
                "print(\"8. ✅ 訓練執行\")\n",
                "print(\"9. ✅ 模型保存\")\n",
                "print(\"10. ✅ 推理測試\")\n",
                "print(\"=\" * 50)\n\n",
                "print(\"\\n🔧 如果您遇到 'unauthorized' 錯誤:\")\n",
                "print(\"1. 運行認證單元格 (單元格 2)\")\n",
                "print(\"2. 按照認證步驟操作\")\n",
                "print(\"3. 重新運行模型載入單元格 (單元格 4)\")\n",
                "print(\"4. 繼續其餘的訓練\")\n\n",
                "print(\"\\n💡 標準 Trainer 方法提供:\")\n",
                "print(\"- 對 tokenization 的完整控制\")\n",
                "print(\"- 手動數據集預處理\")\n",
                "print(\"- 明確的標籤處理\")\n",
                "print(\"- 更好的調試能力\")\n",
                "print(\"- 更多自定義訓練的靈活性\")\n\n",
                "print(\"\\n🚀 準備好訓練您的 MedGemma 27B 模型!\")\n",
                "print(\"所有 SFTTrainer 單元格已被移除以保持清晰。\")"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.5"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# 寫入文件
with open('finetune_medgemma_27b_中文.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=2)

print('✅ 修正版 Jupyter Notebook 創建成功!')
print('文件: finetune_medgemma_27b_中文.ipynb')
print('所有 4B 標註已修正為 27B') 