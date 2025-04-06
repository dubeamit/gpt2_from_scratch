# GPT-2 From Scratch

This repository contains a minimal implementation of the GPT-2 model architecture built from scratch using PyTorch, implementing key components like Multi-Head Self-Attention, Layer Normalization, and GELU activation.

This code was developed while learning from the **Build LLM From Scratch** playlist by **Vizuara** on YouTube:
[Playlist link](https://www.youtube.com/watch?v=Xpr8D6LeAtw&list=PLPTV0NXA_ZSgsLAr8YCgCwhPIJNNtexWu)

## Features

*   Core GPT-2 components (Attention, LayerNorm, FeedForward, GELU).
*   Token and Positional Embeddings.
*   Data loading and preparation using `tiktoken` and `torch.utils.data.Dataset`.
*   Basic text generation capability.
*   Training loop with loss calculation and evaluation.
*   Plotting of training/validation losses.

## Project Structure

```
gpt2_from_scratch/
├── model/             # Core GPT model definition
│   ├── __init__.py
│   └── gpt2_model.py
├── utils/             # Utility functions
│   ├── __init__.py
│   ├── data_utils.py    # Base Dataset/Dataloader code
│   ├── train_utils.py   # Base training loop, eval, generation, weight loading
│   └── classifier_utils.py # Classifier-specific utilities
├── gpt2/              # Downloaded pre-trained models (ignored by git)
├── config.py          # Model configuration
├── train.py           # Script for base model training (from scratch)
├── finetune_classifier.py # Script for classification fine-tuning
├── gpt_download.py    # Helper script to download models
├── pretrained_gpt2.py # Script to load/test pretrained weights
├── the-verdict.txt    # Sample training data for base model
├── *.png              # Output plots (ignored by git)
├── *.csv              # Data splits (ignored by git)
├── *.pth              # Saved models (ignored by git)
└── README.md
```

## Usage

### 1. Base Model Training (From Scratch)

*   **Install dependencies:**
    ```bash
    pip install torch tiktoken matplotlib pandas
    ```
*   **Run the training script:**
    ```bash
    python train.py
    ```
    This trains the model defined in `model/gpt2_model.py` from random initialization using the data in `the-verdict.txt`.

### 2. Fine-tuning for Spam Classification

*   **Install additional dependency:**
    ```bash
    pip install pandas
    ```
*   **(Optional) Download Pre-trained Model Separately:**
    You can first download the weights if needed:
    ```bash
    python gpt_download.py --model_size 124M 
    # Or run pretrained_gpt2.py which also downloads:
    # python pretrained_gpt2.py 
    ```
*   **Run the fine-tuning script:**
    ```bash
    python finetune_classifier.py
    ```

This will:
*   Download the SMS Spam Collection dataset if not present.
*   Preprocess and split the data into balanced train/validation/test sets (`train.csv`, etc. - ignored by git).
*   Load the pre-trained GPT-2 (124M) model weights (downloading if necessary via `gpt_download.py`).
*   Adapt the model head for binary classification.
*   Freeze most base model layers and fine-tune the classification head and the last transformer block.
*   Print evaluation metrics (loss and accuracy).
*   Plot the fine-tuning loss and accuracy curves (`finetune-*.png` - ignored by git).
*   Evaluate the final model on the test set.
*   Save the fine-tuned classifier model to `spam_classifier_finetuned.pth` (ignored by git).
