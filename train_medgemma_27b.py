#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MedGemma 27B 中文微調訓練腳本

這個腳本展示如何使用 LoRA (Low-Rank Adaptation) 技術在醫療問答數據上微調 MedGemma 27B 多模態模型。
專案針對 GPU 訓練進行了優化，特別適合 RTX 4090 等高端顯卡。

作者: AI Assistant
版本: 1.0
日期: 2024
"""

import os
import sys
import json
import warnings
import argparse
from pathlib import Path

import torch
import pandas as pd
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType, 
    PeftModel, 
    prepare_model_for_kbit_training
)
from datasets import Dataset
from huggingface_hub import login, HfApi

warnings.filterwarnings('ignore')

class MedGemmaTrainer:
    """MedGemma 訓練器類"""
    
    def __init__(self, config):
        """
        初始化訓練器
        
        Args:
            config (dict): 配置字典
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        print("🚀 MedGemma 27B 中文微調訓練器初始化")
        print("=" * 60)
        
    def check_gpu(self):
        """檢查 GPU 可用性和記憶體"""
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ GPU 可用: {gpu_name}")
            print(f"GPU 記憶體: {gpu_memory:.2f} GB")
            
            if gpu_memory < 20:
                print("⚠️  警告: GPU 記憶體不足 20GB，可能需要調整批次大小")
            elif gpu_memory >= 24:
                print("✅ GPU 記憶體充足，可以使用預設設置")
            else:
                print("⚡ GPU 記憶體適中，建議監控使用情況")
        else:
            print("❌ 錯誤: 未檢測到 GPU! 此腳本需要 CUDA 相容的 GPU。")
            sys.exit(1)
            
    def authenticate_huggingface(self):
        """HuggingFace 認證"""
        print("\n🔐 進行 HuggingFace 認證...")
        
        # 檢查是否已設置環境變數
        if os.getenv("HUGGINGFACE_HUB_TOKEN"):
            print("✅ 發現環境變數中的 HuggingFace token")
            return True
            
        # 嘗試互動式登入
        try:
            login()
            print("✅ HuggingFace 認證成功!")
            return True
        except Exception as e:
            print(f"❌ 認證失敗: {e}")
            print("\n🔧 請按照以下步驟進行認證:")
            print("1. 前往 https://huggingface.co/google/medgemma-4b-multimodal")
            print("2. 點擊 'Request Access' 並接受授權協議")
            print("3. 等待審核通過")
            print("4. 設置環境變數: export HUGGINGFACE_HUB_TOKEN='your_token'")
            print("5. 重新運行此腳本")
            return False
            
    def load_dataset(self):
        """載入和預處理數據集"""
        print("\n📊 載入 MedQuAD 數據集...")
        
        try:
            df = pd.read_csv(self.config['data_path'])
            print(f"數據集形狀: {df.shape}")
            print(f"欄位: {df.columns.tolist()}")
            
            # 檢查必要欄位
            required_columns = ['question', 'answer']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"數據集缺少必要欄位: {missing_columns}")
                
            # 格式化為指令-回應格式
            def format_instruction(row):
                return f"Instruction: {row['question']}\nResponse: {row['answer']}"
                
            df['text'] = df.apply(format_instruction, axis=1)
            
            # 取樣數據以控制訓練時間
            sample_size = min(self.config.get('sample_size', 2000), len(df))
            df_sample = df.sample(n=sample_size, random_state=42)
            print(f"使用 {len(df_sample)} 個樣本進行微調")
            
            # 轉換為 HuggingFace Dataset
            self.dataset = Dataset.from_pandas(df_sample[['text']])
            print("✅ 數據集載入成功!")
            
        except Exception as e:
            print(f"❌ 數據集載入失敗: {e}")
            raise
            
    def setup_model(self):
        """設置模型和 tokenizer"""
        print("\n🤖 設置 MedGemma 模型...")
        
        model_id = self.config['model_id']
        
        # 檢查模型存取權限
        try:
            api = HfApi()
            model_info = api.model_info(model_id)
            print("✅ 模型存取權限確認")
        except Exception as e:
            print(f"❌ 模型存取失敗: {e}")
            print("請確保您已申請 MedGemma 模型存取權限")
            raise
            
        # 配置 4-bit 量化
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        # 載入 tokenizer
        print("載入 tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        # 載入模型
        print("載入模型 (4-bit 量化)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        
        print("✅ 模型載入成功!")
        
        # 檢查記憶體使用
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"GPU 記憶體使用: 已分配 {allocated:.2f} GB, 已保留 {reserved:.2f} GB")
            
    def setup_lora(self):
        """設置 LoRA 配置"""
        print("\n🎛️  設置 LoRA 配置...")
        
        lora_config = LoraConfig(
            r=self.config.get('lora_r', 16),
            lora_alpha=self.config.get('lora_alpha', 32),
            lora_dropout=self.config.get('lora_dropout', 0.05),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=self.config.get('target_modules', ["q_proj", "v_proj"]),
        )
        
        # 啟用梯度檢查點以節省記憶體
        self.model.gradient_checkpointing_enable()
        
        # 準備模型進行 k-bit 訓練
        self.model = prepare_model_for_kbit_training(self.model)
        
        # 獲取 PEFT 模型
        self.model = get_peft_model(self.model, lora_config)
        
        # 打印可訓練參數
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.model.parameters())
        print(f"可訓練參數: {trainable_params:,} ({100 * trainable_params / all_params:.2f}%)")
        print(f"總參數: {all_params:,}")
        
    def prepare_dataset(self):
        """準備訓練數據集"""
        print("\n📝 準備訓練數據集...")
        
        def tokenize_function(examples):
            """對文本進行 tokenization"""
            texts = [text + self.tokenizer.eos_token for text in examples['text']]
            
            model_inputs = self.tokenizer(
                texts,
                truncation=True,
                padding=False,
                max_length=self.config.get('max_length', 512),
                return_tensors=None
            )
            
            model_inputs["labels"] = model_inputs["input_ids"].copy()
            return model_inputs
            
        self.tokenized_dataset = self.dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=self.dataset.column_names,
            desc="Tokenizing 數據集"
        )
        
        print(f"Tokenized 數據集大小: {len(self.tokenized_dataset)}")
        
    def setup_training(self):
        """設置訓練參數和 trainer"""
        print("\n⚙️  設置訓練參數...")
        
        training_args = TrainingArguments(
            output_dir=self.config.get('output_dir', "./results"),
            num_train_epochs=self.config.get('num_epochs', 1),
            per_device_train_batch_size=self.config.get('batch_size', 4),
            gradient_accumulation_steps=self.config.get('gradient_accumulation_steps', 4),
            optim="paged_adamw_32bit",
            save_steps=self.config.get('save_steps', 100),
            logging_steps=self.config.get('logging_steps', 25),
            learning_rate=self.config.get('learning_rate', 2e-5),
            weight_decay=self.config.get('weight_decay', 0.001),
            bf16=True,
            max_grad_norm=self.config.get('max_grad_norm', 0.3),
            max_steps=-1,
            warmup_ratio=self.config.get('warmup_ratio', 0.03),
            group_by_length=True,
            lr_scheduler_type="constant",
            report_to="tensorboard",
            save_total_limit=2,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
        )
        
        # 設置數據收集器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
        # 初始化 Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        print("✅ 訓練設置完成!")
        print(f"批次大小: {training_args.per_device_train_batch_size}")
        print(f"梯度累積步數: {training_args.gradient_accumulation_steps}")
        print(f"有效批次大小: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
        print(f"學習率: {training_args.learning_rate}")
        print(f"訓練輪數: {training_args.num_train_epochs}")
        
    def train(self):
        """開始訓練"""
        print("\n🚀 開始訓練...")
        
        try:
            self.trainer.train()
            print("✅ 訓練成功完成!")
        except Exception as e:
            print(f"❌ 訓練失敗: {e}")
            print("常見解決方案:")
            print("1. 減少 batch_size 或 max_length")
            print("2. 檢查模型存取權限")
            print("3. 檢查 CUDA 相容性")
            raise
            
    def save_model(self):
        """保存微調模型"""
        print("\n💾 保存微調模型...")
        
        output_dir = self.config.get('model_output_dir', "./finetuned_medgemma_4b")
        
        # 保存模型和 tokenizer
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # 保存訓練配置
        training_config = {
            "model_id": self.config['model_id'],
            "trainer_type": "normal_trainer",
            "lora_config": {
                "r": self.config.get('lora_r', 16),
                "lora_alpha": self.config.get('lora_alpha', 32),
                "lora_dropout": self.config.get('lora_dropout', 0.05),
                "target_modules": self.config.get('target_modules', ["q_proj", "v_proj"]),
            },
            "training_args": {
                "learning_rate": self.config.get('learning_rate', 2e-5),
                "num_train_epochs": self.config.get('num_epochs', 1),
                "per_device_train_batch_size": self.config.get('batch_size', 4),
                "gradient_accumulation_steps": self.config.get('gradient_accumulation_steps', 4),
            },
            "dataset_size": len(self.tokenized_dataset),
        }
        
        with open(os.path.join(output_dir, "training_config.json"), "w", encoding='utf-8') as f:
            json.dump(training_config, f, indent=2, ensure_ascii=False)
            
        print(f"✅ 模型已保存到: {output_dir}")
        
    def test_model(self):
        """測試微調模型"""
        print("\n🧪 測試微調模型...")
        
        # 清理 GPU 記憶體
        torch.cuda.empty_cache()
        
        # 重新載入模型進行推理
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config['model_id'],
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        
        output_dir = self.config.get('model_output_dir', "./finetuned_medgemma_4b")
        finetuned_model = PeftModel.from_pretrained(base_model, output_dir)
        finetuned_model.eval()
        
        # 測試問題
        test_questions = [
            "糖尿病的症狀是什麼?",
            "高血壓如何治療?",
            "心臟病的原因是什麼?",
        ]
        
        print("測試自定義問題:")
        print("=" * 60)
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n測試 {i}:")
            print(f"問題: {question}")
            
            test_input = f"Instruction: {question}\nResponse:"
            inputs = self.tokenizer(test_input, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = finetuned_model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "Response:" in response:
                generated_response = response.split("Response:")[-1].strip()
            else:
                generated_response = response
                
            print(f"回應: {generated_response}")
            print("-" * 40)
            
    def run(self):
        """運行完整的訓練流程"""
        print("🎯 開始 MedGemma 27B 中文微調流程")
        print("=" * 60)
        
        # 檢查 GPU
        self.check_gpu()
        
        # 認證
        if not self.authenticate_huggingface():
            return False
            
        # 載入數據集
        self.load_dataset()
        
        # 設置模型
        self.setup_model()
        
        # 設置 LoRA
        self.setup_lora()
        
        # 準備數據集
        self.prepare_dataset()
        
        # 設置訓練
        self.setup_training()
        
        # 開始訓練
        self.train()
        
        # 保存模型
        self.save_model()
        
        # 測試模型
        self.test_model()
        
        print("\n🎉 訓練流程完成!")
        print("=" * 60)
        return True

def main():
    """主函數"""
    parser = argparse.ArgumentParser(description="MedGemma 27B 中文微調訓練腳本")
    parser.add_argument("--config", type=str, default="config.json", help="配置文件路徑")
    parser.add_argument("--data_path", type=str, default="medquad.csv", help="數據集路徑")
    parser.add_argument("--output_dir", type=str, default="./results", help="輸出目錄")
    parser.add_argument("--model_output_dir", type=str, default="./finetuned_medgemma_4b", help="模型輸出目錄")
    parser.add_argument("--sample_size", type=int, default=2000, help="數據集樣本大小")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--max_length", type=int, default=512, help="最大序列長度")
    parser.add_argument("--num_epochs", type=int, default=1, help="訓練輪數")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="學習率")
    
    args = parser.parse_args()
    
    # 默認配置
    config = {
        'model_id': 'google/medgemma-4b-multimodal',
        'data_path': args.data_path,
        'output_dir': args.output_dir,
        'model_output_dir': args.model_output_dir,
        'sample_size': args.sample_size,
        'batch_size': args.batch_size,
        'max_length': args.max_length,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'lora_r': 16,
        'lora_alpha': 32,
        'lora_dropout': 0.05,
        'target_modules': ["q_proj", "v_proj"],
        'gradient_accumulation_steps': 4,
        'save_steps': 100,
        'logging_steps': 25,
        'weight_decay': 0.001,
        'max_grad_norm': 0.3,
        'warmup_ratio': 0.03,
    }
    
    # 如果提供了配置文件，則載入
    if os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            file_config = json.load(f)
            config.update(file_config)
    
    # 創建輸出目錄
    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(config['model_output_dir'], exist_ok=True)
    
    # 運行訓練
    trainer = MedGemmaTrainer(config)
    success = trainer.run()
    
    if success:
        print("\n✅ 訓練成功完成!")
        print(f"模型已保存到: {config['model_output_dir']}")
        print("您可以使用保存的模型進行醫療問答!")
    else:
        print("\n❌ 訓練失敗，請檢查錯誤信息")
        sys.exit(1)

if __name__ == "__main__":
    main() 