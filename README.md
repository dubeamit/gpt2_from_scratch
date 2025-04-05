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
*   Example usage script demonstrating training on a sample text.

## Usage

1.  **Install dependencies:**
    ```bash
    pip install torch tiktoken matplotlib
    ```
2.  **Run the training script:**
    ```bash
    python gpt2.py
    ```

This will download the sample data ('the-verdict.txt'), initialize the model, train it for a few epochs using the configurations set in the script, print evaluation losses, generate sample text periodically, and finally plot the training/validation loss curves.