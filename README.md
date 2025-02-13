# Fine-Tuning Qwen1.5-0.5B with LoRA & 4-bit Quantization

This project fine-tunes the `Qwen1.5-0.5B` model using 4-bit quantization and Low-Rank Adaptation (LoRA) to efficiently adapt it for legal text generation.

## Features
- Uses 4-bit quantization for reduced memory footprint.
- Implements LoRA for efficient fine-tuning.
- Loads and processes a legal dataset in JSON format.
- Splits dataset into training and validation sets.
- Fine-tunes with `Trainer` from Hugging Face.
- Saves and downloads the fine-tuned model.

## Installation
Ensure the required libraries are installed:
```bash
pip install torch torchvision torchaudio
pip install --upgrade fsspec gcsfs
pip install transformers datasets accelerate peft bitsandbytes
```

## Running the Fine-Tuning Script
1. Clone the repository:
```bash
git clone https://github.com/yourusername/qwen-fine-tune.git
cd qwen-fine-tune
```
2. Run the fine-tuning script:
```python
python fine_tune_qwen.py
```

## File Structure
```
/qwen-fine-tune
│── fine_tune_qwen.py   # Main script to fine-tune the model
│── legal_finetune_data.json # Legal dataset for training
│── qwen_fine_tuned/    # Directory for the fine-tuned model
```

## Dataset Format
The dataset should be a JSON file structured as follows:
```json
[
  {
    "prompt": "Provide details on legal section 1",
    "completion": "Section 1 covers the introduction to legal principles..."
  }
]
```

## How It Works
1. Loads the `Qwen1.5-0.5B` model with 4-bit quantization.
2. Tokenizes the dataset for efficient processing.
3. Applies LoRA fine-tuning to optimize memory usage.
4. Uses Hugging Face’s `Trainer` to train the model.
5. Saves and downloads the fine-tuned model.

## Training Configuration
Key training arguments:
- 3 epochs with `fp16` for performance.
- Batch size: `1` with gradient accumulation (`8` steps).
- Saves model every epoch.
- Logging every `10` steps.

## Saving & Downloading the Model
Once training is complete, the model is saved and compressed:
```bash
zip -r qwen_fine_tuned.zip ./qwen_fine_tuned
```
You can download the model directly from Colab.

## Contribution
Feel free to fork and contribute! Open issues for any questions or improvements.

## License
MIT License

## Acknowledgments
- [Transformers](https://huggingface.co) for pre-trained models
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) for quantization
- [PEFT](https://huggingface.co/docs/peft) for efficient parameter fine-tuning
- This project utilizes datasets from another GitHub repository for model training. Special thanks to the original dataset creator. 🎖 [https://github.com/civictech-India/Indian-Law-Penal-Code-Json/tree/main]
