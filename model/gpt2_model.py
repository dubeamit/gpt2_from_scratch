"""
Core GPT-2 model architecture components.
Includes Layer Normalization, GELU activation, Feed Forward network,
Multi-Head Attention, Transformer Block, and the main GPTModel class.
"""

import torch
import torch.nn as nn
import math

class LayerNorm(nn.Module):
    """Implements Layer Normalization.

    Normalizes the features across the embedding dimension for stability.
    Includes learnable scale (gamma) and shift (beta) parameters.
    """
    def __init__(self, emb_dim, eps=1e-5):
        """Initializes Layer Normalization parameters.

        Args:
            emb_dim (int): The dimension of the features to normalize.
            eps (float): A small value added to the variance for numerical stability.
        """
        super().__init__()
        self.eps = eps
        # Learnable affine parameters: scale (gamma) and shift (beta)
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        """Applies Layer Normalization.
        Calculates mean and variance across the last dimension (emb_dim).
        Normalizes the input and then applies the learnable scale and shift.

        Args:
            x (torch.Tensor): Input tensor (..., emb_dim).

        Returns:
            torch.Tensor: Normalized tensor with the same shape as input.
        
        """
        # Calculate mean and variance along the last dimension (embedding dimension)
        mean = x.mean(dim=-1, keepdim=True)
        # Use population variance (unbiased=False) as is common in LLMs
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        # Normalize the input tensor
        norm_x = (x - mean) / torch.sqrt(var + self.eps) # Add eps for numerical stability

        # Apply learnable scale and shift parameters
        return self.scale * norm_x + self.shift

class GELU(nn.Module):
    """Implements the Gaussian Error Linear Unit activation function (approximation).

    Provides a smooth, non-linear activation, commonly used in Transformers.
    Uses the common approximation formula for GELU.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """Applies the GELU activation function.

        Uses the approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with GELU applied element-wise.
        """
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
        ))

class FeedForward(nn.Module):
    """Implements the Position-wise Feed-Forward Network.

    Applies two linear transformations with a GELU activation in between.
    Expands the embedding dimension in the first layer and projects back in the second.
    """
    def __init__(self, cfg):
        """Initializes the Feed Forward layers.

        Args:
            cfg (dict): Configuration dictionary containing 'emb_dim'.
        """
        super().__init__()
        # Standard FFN architecture: Linear -> Activation -> Linear
        # The intermediate dimension is typically 4 * emb_dim
        hidden_dim = 4 * cfg['emb_dim']
        self.layers = nn.Sequential(
            nn.Linear(cfg['emb_dim'], hidden_dim), # Expansion layer
            GELU(),                                # Activation function
            nn.Linear(hidden_dim, cfg['emb_dim']),  # Projection layer
        )

    def forward(self, x):
        """Applies the Feed Forward network.

        Args:
            x (torch.Tensor): Input tensor (B, T, D).

        Returns:
            torch.Tensor: Output tensor (B, T, D).
        """
        return self.layers(x)

class MultiHeadAttention(nn.Module):
    """Implements Multi-Head Self-Attention with optional causal masking.

    Allows the model to jointly attend to information from different representation
    subspaces at different positions.
    Includes optional causal masking for decoder-style architectures.
    """
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        """Initializes the Multi-Head Attention layer.

        Args:
            d_in (int): Input dimension (embedding dimension).
            d_out (int): Output dimension (must be divisible by num_heads).
            context_length (int): Maximum sequence length for the causal mask.
            dropout (float): Dropout rate.
            num_heads (int): Number of attention heads.
            qkv_bias (bool): Whether to include bias in Q, K, V linear layers.
        """
        super().__init__()
        if d_out % num_heads != 0:
            raise ValueError("d_out must be divisible by num_heads")

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # Dimension per attention head

        # Linear layers for Query, Key, Value projections
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        # Final output projection layer
        self.out_proj = nn.Linear(d_out, d_out)
        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)

        # Causal mask to prevent attention to future tokens
        # register_buffer ensures the mask is part of the module's state,
        # but not updated by the optimizer
        self.register_buffer(
            "mask",
            # Creates an  upper triangular of ones and a lower triangular matrix of zeros and
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        """Performs the forward pass of the Multi-Head Attention.

        Args:
            x (torch.Tensor): Input tensor (B, T, d_in).

        Returns:
            torch.Tensor: Output tensor (B, T, d_out).
        """
        b, num_tokens, d_in = x.shape # Batch size, Sequence length, Input dimension

        # Project input into Q, K, V for all heads, but in a batch
        # (B, T, d_in) -> (B, T, d_out)
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Reshape Q, K, V to split the last dimension into (num_heads, head_dim)
        # and then transpose to (B, num_heads, T, head_dim)
        # (B, T, D_out) -> (B, T, num_heads, head_dim) -> (B, num_heads, T, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores (scaled dot-product)
        # queries shape: (B, num_heads, T, head_dim)
        # keys.transpose shape: (B, num_heads, head_dim, T)
        # result shape: (B, num_heads, T, T)
        attn_scores = queries @ keys.transpose(-2, -1) # Use -2, -1 for robustness

        # Apply causal mask: Original mask truncated to the number of tokens and converted to boolean
        # Get the mask relevant for the current sequence length
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        # Fill positions corresponding to True in the mask with -infinity
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # Normalize scores into weights (probabilities)
        # Scale by sqrt(head_dim) before softmax
        attn_weights = torch.softmax(attn_scores / math.sqrt(self.head_dim), dim=-1)
        attn_weights = self.dropout(attn_weights) # Apply dropout to attention weights

        # Calculate context vector (weighted sum of values)
        # (B, num_heads, T, T) @ (B, num_heads, T, head_dim) -> (B, num_heads, T, head_dim)
        context_vec = attn_weights @ values

        # Transpose and reshape context vector back to match input shape expectations
        # (B, num_heads, T, head_dim) -> (B, T, num_heads, head_dim)
        context_vec = context_vec.transpose(1, 2)
        # Concatenate heads: (B, T, num_heads, head_dim) -> (B, T, D_out)
        # .contiguous() ensures memory layout is suitable for .view()
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)

        # Final linear projection
        context_vec = self.out_proj(context_vec) # (B, T, d_out) -> (B, T, d_out)

        return context_vec

class TransformerBlock(nn.Module):
    """A single Transformer block using Multi-Head Attention and Feed Forward layers.

    Implements the standard pre-LayerNorm architecture:
    Input -> LayerNorm -> MultiHeadAttention -> Residual + Dropout ->
    LayerNorm -> FeedForward -> Residual + Dropout -> Output
    """
    def __init__(self, cfg):
        """Initializes the Transformer block components.

        Args:
            cfg (dict): Configuration dictionary.
        """
        super().__init__()
        self.att = MultiHeadAttention(
            d_in = cfg['emb_dim'],
            d_out = cfg['emb_dim'],
            context_length = cfg['context_length'],
            num_heads = cfg['n_heads'],
            dropout = cfg['drop_rate'],
            qkv_bias = cfg['qkv_bias']
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg['emb_dim'])
        self.norm2 = LayerNorm(cfg['emb_dim'])
        self.drop_shortcut = nn.Dropout(cfg['drop_rate'])

    def forward(self, x):
        """Performs the forward pass of the Transformer block.

        Args:
            x (torch.Tensor): Input tensor (B, T, D).

        Returns:
            torch.Tensor: Output tensor (B, T, D).
        """
        # --- Multi-Head Attention Sub-layer ---
        shortcut = x # Store input for residual connection
        # Pre-normalization
        x_norm1 = self.norm1(x)
        # Attention mechanism [batch_size, num_tokens, emb_size]
        attn_output = self.att(x_norm1)
        # Residual connection with dropout
        x = shortcut + self.drop_shortcut(attn_output)

        # --- Feed Forward Sub-layer ---
        shortcut = x # Store intermediate result for residual connection
        # Pre-normalization
        x_norm2 = self.norm2(x)
        # Feed Forward network
        ff_output = self.ff(x_norm2)
        # Residual connection with dropout
        x = shortcut + self.drop_shortcut(ff_output)

        return x


class GPTModel(nn.Module):
    """The main GPT-2 model architecture.

    Combines token embeddings, positional embeddings, multiple Transformer blocks,
    a final Layer Normalization, and an output linear layer to produce logits.
    """
    def __init__(self, cfg):
        """Initializes the GPT-2 model components.

        Args:
            cfg (dict): Configuration dictionary with model hyperparameters.
        """
        super().__init__()
        self.cfg = cfg # Store config
        self.tok_emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.pos_emb = nn.Embedding(cfg['context_length'], cfg['emb_dim'])
        self.drop_emb = nn.Dropout(cfg['drop_rate'])

        # Use ModuleList for sequential layers if you need to access them individually later,
        # otherwise nn.Sequential is fine.
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg['n_layers'])]
        )

        # Final normalization layer before output head
        self.final_norm = LayerNorm(cfg['emb_dim'])
        # Output linear layer projecting to vocabulary size
        self.out_head = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False)

    def forward(self, in_idx):
        """Performs the forward pass of the model.

        Args:
            in_idx (torch.Tensor): Input tensor of token indices (B, T).

        Returns:
            torch.Tensor: Output logits (B, T, vocab_size).
        """
        batch_size, seq_len = in_idx.shape

        # Handle sequences longer than context length
        if seq_len > self.cfg['context_length']:
            # Truncate if longer
            print(f"Warning: Input sequence length ({seq_len}) longer than context ({self.cfg['context_length']}). Truncating.")
            in_idx = in_idx[:, -self.cfg['context_length']:]
            seq_len = self.cfg['context_length'] # Update seq_len accordingly

        # Token embeddings: (B, T) -> (B, T, D)
        tok_embeds = self.tok_emb(in_idx)
        # Positional embeddings: (T) -> (T, D)
        pos_indices = torch.arange(seq_len, device=in_idx.device)
        pos_embeds = self.pos_emb(pos_indices) # (T, D)

        # Combine embeddings (broadcasts positional embeddings across batch)
        x = tok_embeds + pos_embeds # (B, T, D)
        # Apply dropout after combining embeddings
        x = self.drop_emb(x)

        # Pass through the stack of Transformer blocks
        x = self.trf_blocks(x) # (B, T, D)

        # Apply final layer normalization
        x = self.final_norm(x) # (B, T, D)
        # Project to vocabulary logits
        logits = self.out_head(x) # (B, T, vocab_size)
        return logits 