"""
Model configuration settings.
"""

# Configuration dictionary for the GPT-2 124M parameter model variant.
GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size (obtained from the original GPT-2 tokenizer).
    "context_length": 256, # Max sequence length the model can process. Reduced from 1024 for faster training/memory efficiency.
    "emb_dim": 768,         # Embedding dimension (size of token and positional embeddings).
    "n_heads": 12,          # Number of attention heads in the Multi-Head Attention layers.
    "n_layers": 12,         # Number of Transformer blocks (layers) in the model.
    "drop_rate": 0.1,       # Dropout rate for regularization. Set to 0.0 in train.py for initial testing.
    "qkv_bias": False       # Whether to include bias terms in the Query, Key, Value linear layers.
}

# You could add other configurations here later, e.g.:
# GPT_CONFIG_355M = { ... }
