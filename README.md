# MedGemma 4B Fine-tuning with LoRA

This project demonstrates how to fine-tune the MedGemma 4B Multimodal model using LoRA (Low-Rank Adaptation) on medical question-answering data.

## Requirements

- NVIDIA GPU with at least 16GB VRAM (optimized for RTX 4090 24GB)
- Python 3.8+
- CUDA-compatible PyTorch installation

## Setup

1. Clone this repository
2. Install required packages:
```bash
pip install torch transformers peft bitsandbytes trl datasets pandas accelerate
```

3. Ensure you have the MedQuAD dataset (`medquad.csv`) in the project directory

## Usage

1. Open the Jupyter notebook:
```bash
jupyter notebook finetune_medgemma_4b.ipynb
```

2. Run all cells sequentially to:
   - Load and preprocess the MedQuAD dataset
   - Configure the model with 4-bit quantization
   - Set up LoRA fine-tuning
   - Train the model
   - Save the fine-tuned model
   - Test with sample questions

## Key Features

- **Memory Efficient**: Uses 4-bit quantization to fit on 24GB VRAM
- **LoRA Fine-tuning**: Efficient adaptation without full model retraining
- **Medical Focus**: Trained on medical Q&A data from MedQuAD
- **GPU Monitoring**: Built-in memory usage tracking and optimization suggestions

## Model Configuration

- **Base Model**: google/medgemma-4b-multimodal
- **LoRA Parameters**: r=16, alpha=32, dropout=0.05
- **Training**: Batch size=4, Gradient accumulation=4, BF16 precision
- **Dataset**: 1000 samples from MedQuAD (configurable)

## Output

The fine-tuned model will be saved to `./finetuned_medgemma_4b/` containing:
- LoRA adapter weights
- Tokenizer configuration
- Training configuration JSON

## Important Notes

⚠️ **Disclaimer**: This model is for educational and research purposes only. Medical advice should always be verified by qualified healthcare professionals.

## Troubleshooting

- **Out of Memory**: Reduce batch size or sequence length
- **Model Access**: Ensure HuggingFace authentication for restricted models
- **CUDA Issues**: Verify PyTorch CUDA compatibility

## License

This project is for educational use. Please respect the original model and dataset licenses. 