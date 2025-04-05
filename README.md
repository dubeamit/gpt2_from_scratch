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
│   ├── data_utils.py    # Dataset/Dataloader code
│   └── train_utils.py   # Training loop, eval, generation helpers
├── config.py          # Model configuration
├── train.py           # Main training script
├── the-verdict.txt    # Sample training data
├── training_losses.png # Example output plot
└── README.md
```

## Usage

1.  **Install dependencies:**
    ```bash
    pip install torch tiktoken matplotlib
    ```
2.  **Run the training script:**
    ```bash
    python train.py
    ```

This will:
*   Download the sample data ('the-verdict.txt') if not present.
*   Load the model configuration from `config.py`.
*   Initialize the model, tokenizer, and data loaders.
*   Run the training loop defined in `train.py` using functions from `utils/`.
*   Print evaluation losses and generate sample text periodically.
*   Save a plot of the training/validation loss curves to `training_losses.png`.