"""
Minimal implementation of GPT-2 model architecture and components.

Based on the paper "Language Models are Unsupervised Multitask Learners"
(https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

Vizuara Youtube playlist --> Build LLM from Scratch 
https://www.youtube.com/watch?v=Xpr8D6LeAtw&list=PLPTV0NXA_ZSgsLAr8YCgCwhPIJNNtexWu
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import tiktoken
import math # For GELU approximation
import os
import urllib.request


GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False     # Query-key-Value bias
}


class GPTModel(nn.Module):
    """The main GPT-2 model architecture."""
    def __init__(self, cfg):
        """Initializes the GPT-2 model components.

        Args:
            cfg (dict): Configuration dictionary with model hyperparameters.
        """
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.pos_emb = nn.Embedding(cfg['context_length'], cfg['emb_dim'])
        self.drop_emb = nn.Dropout(cfg['drop_rate'])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg['n_layers'])]
        )

        self.final_norm = LayerNorm(cfg['emb_dim'])
        self.out_head = nn.Linear(
            cfg['emb_dim'], cfg['vocab_size'], bias=False
        )

    def forward(self, in_idx):
        """Performs the forward pass of the model.

        Args:
            in_idx (torch.Tensor): Input tensor of token indices (B, T).

        Returns:
            torch.Tensor: Output logits (B, T, vocab_size).
        """
        batch_size, seq_len = in_idx.shape
        
        # Validate sequence length
        if seq_len > self.cfg['context_length']:
            raise ValueError(f"Input sequence length ({seq_len}) exceeds context length ({self.cfg['context_length']})")

        # # Token embeddings: Look up embedding vector for each token ID (B, T) -> (B, T, D)
        tok_embeds = self.tok_emb(in_idx)
        # Positional embeddings: Look up embedding vector for each position [0, 1, ..., seq_len-1] (T) -> (T, D)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        
        # Combine token and positional embeddings: (B, T, D) + (T, D) -> (B, T, D)
        x = tok_embeds + pos_embeds
        # Apply dropout to the combined embeddings
        x = self.drop_emb(x) 
        
        # Pass through Transformer blocks
        x = self.trf_blocks(x)
        
        # Final layer normalization and output projection
        x = self.final_norm(x)
        logits = self.out_head(x) # (B, T, D) -> (B, T, vocab_size)
        return logits
    
class TransformerBlock(nn.Module):
    """A single Transformer block containing Multi-Head Attention and Feed Forward layers."""
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
        
        # Multi-Head Attention block with pre-LayerNorm and residual connection
        shortcut = x
        x = self.norm1(x)
        x = self.att(x) # shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Feed Forward block with pre-LayerNorm and residual connection
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut 

        return x


class MultiHeadAttention(nn.Module):
    """Implements Multi-Head Self-Attention with optional causal masking."""
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
        self.head_dim = d_out // num_heads # Dimension of each head's key, query, value

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out) # Final linear projection
        self.dropout = nn.Dropout(dropout)

        # Causal mask to prevent attention to future tokens
        # register_buffer ensures the mask is part of the module's state,
        # but not updated by the optimizer
        self.register_buffer(
            "mask",
            # Create upper triangular matrix of 1s
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        """Performs the forward pass of the Multi-Head Attention.

        Args:
            x (torch.Tensor): Input tensor (B, T, d_in).

        Returns:
            torch.Tensor: Output tensor (B, T, d_out).
        """
        b, num_tokens, d_in = x.shape

        # Project input into Q, K, V for all heads, but in a batch
        # (B, T, d_in) -> (B, T, d_out)
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Reshape Q, K, V to split the last dimension into (num_heads, head_dim)
        # (B, T, d_out) -> (B, T, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose Q, K, V to bring heads dimension forward for batch matrix multiplication
        # (B, T, num_heads, head_dim) -> (B, num_heads, T, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        # (B, num_heads, T, head_dim) @ (B, num_heads, head_dim, T) -> (B, num_heads, T, T)
        attn_scores = queries @ keys.transpose(2, 3)

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        # Fill attention scores with -infinity where mask is True (preventing attention to future positions)
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # Normalize scores into weights (probabilities)
        # Scale by sqrt(head_dim) before softmax
        attn_weights = torch.softmax(attn_scores / math.sqrt(self.head_dim), dim=-1)
        attn_weights = self.dropout(attn_weights) # Apply dropout to attention weights

        # Calculate context vector (weighted sum of values)
        # (B, num_heads, T, T) @ (B, num_heads, T, head_dim) -> (B, num_heads, T, head_dim)
        context_vec = attn_weights @ values
        # context_vec = (attn_weights @ values).transpose(1, 2) 

        # Transpose and reshape context vector back to match input shape expectations
        # (B, num_heads, T, head_dim) -> (B, T, num_heads, head_dim)
        context_vec = context_vec.transpose(1, 2)
        # (B, T, num_heads, head_dim) -> (B, T, d_out)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)

        # Final linear projection
        context_vec = self.out_proj(context_vec) # (B, T, d_out) -> (B, T, d_out)

        return context_vec


class LayerNorm(nn.Module):
    """Implements Layer Normalization."""
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

        Args:
            x (torch.Tensor): Input tensor (..., emb_dim).

        Returns:
            torch.Tensor: Normalized tensor with the same shape as input.
        """
        # Calculate mean and variance along the last dimension (embedding dimension)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False) # Use population variance

        # Normalize
        norm_x = (x - mean) / torch.sqrt(var + self.eps)

        # Scale and shift
        return self.scale * norm_x + self.shift
    

class FeedForward(nn.Module):
    """Implements the Position-wise Feed-Forward Network."""
    def __init__(self, cfg):
        """Initializes the Feed Forward layers.

        Args:
            cfg (dict): Configuration dictionary.
        """
        super().__init__()
        # Standard FFN architecture: Linear -> Activation -> Linear
        # The intermediate dimension is typically 4 * emb_dim
        self.layers = nn.Sequential(
            nn.Linear(cfg['emb_dim'], 4 * cfg['emb_dim']),
            GELU(),
            nn.Linear(4 * cfg['emb_dim'], cfg['emb_dim']),
            # Dropout can optionally be added here as well
        )

    def forward(self, x):
        """Applies the Feed Forward network.

        Args:
            x (torch.Tensor): Input tensor (B, T, D).

        Returns:
            torch.Tensor: Output tensor (B, T, D).
        """
        return self.layers(x)


class GELU(nn.Module):
    """Implements the Gaussian Error Linear Unit activation function (approximation)."""
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
    
def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (batch, n_tokens) array of indices in the current context
    ## Eg Input batch:
    ## tensor([[6109, 3626, 6100,  345],
    ##         [6109, 1110, 6622,  257]])

    for _ in range(max_new_tokens):
        # crop current context if it exceeds the supported context size
        # Eg: if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used.
        # qns what does it mean if LLM supports only 5 tokens??
        idx_cond = idx[:, -context_size:]

        # get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_tokens, embedding_dim) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Apply softmax to get probabilities 
        probas = torch.softmax(logits, dim=-1) # (batch, vocab_size)

        # Get the idx of the vocab entry with the highest probability value
        idx_next = torch.argmax(probas, dim=-1, keepdim=True) #(batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1) # (batch, n_token+1)

    return idx

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={'<|endoftext|>'})

        # use a sliding window to chunk the data into overlapping sequences of max_length
        for i in range(0, len(token_ids)-max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i+1: max_length]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

        def __len__(self):
            return len(self.input_ids)
        
        def __getitem__(self, idx):
            return self.input_ids[idx], self.target_ids[idx]
        

def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128,
                         shuffle=True, drop_last=True, num_workers=0):
    
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding('gpt2')

    # create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0,1), target_batch.flatten())
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0
    if len(data_loader) == 0:
        return float('nan')
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the
        # data loader if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
    return total_loss / num_batches


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()

def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() # Calculate loss gradients
            optimizer.step() # Update model weights using loss gradients
            tokens_seen += input_batch.numel() # Returns the total number of elements (or tokens) in the input_batch.
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0: 
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Print a sample text after each epoch
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


file_path = "the-verdict.txt"
url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

if not os.path.exists(file_path):
    with urllib.request.urlopen(url) as response:
        text_data = response.read().decode('utf-8')
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text_data)
else:
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()
        
train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]


torch.manual_seed(123)

train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)

val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)

if __name__ == "__main__":
    torch.manual_seed(123) # Set seed for reproducibility

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Model Initialization ---
    print("Initializing model...")
    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)
    model.eval() # Set model to evaluation mode (disables dropout)
    print("Model initialized.")

    # --- Tokenization ---
    print("Setting up tokenizer...")
    tokenizer = tiktoken.get_encoding("gpt2")
    print("Tokenizer set up.")

    # --- Prepare Input Batch ---
    print("Preparing input batch...")
    batch_texts = [
        "Every effort moves you",
        "Every day holds a"
    ]
    # Encode texts and convert to tensors
    batch_indices = [torch.tensor(tokenizer.encode(txt)) for txt in batch_texts]
    # Stack tensors into a single batch tensor
    # Note: This simple stacking assumes sequences have the same length.
    # In practice, padding would be needed for variable lengths.
    batch = torch.stack(batch_indices, dim=0)
    print("Input batch prepared.")
    print("Input batch shape:", batch.shape)
    print("Input batch indices:\n", batch)


    # --- Model Forward Pass ---
    print("Performing forward pass...")
    with torch.no_grad(): # Disable gradient calculation for inference
        out = model(batch)
    print("Forward pass complete.")
    print("\nOutput logits shape:", out.shape)
    # print("Output logits:\n", out) # Optional: Print the large output tensor

    # --- Parameter & Size Calculation ---
    print("Calculating model parameters and size...")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters: {total_params:,}")

    # Compare embedding and output layer shapes (should match vocab size)
    print("Token embedding layer shape:", model.tok_emb.weight.shape)
    # Note: out_head weight is transposed compared to token_emb
    print("Output layer shape:", model.out_head.weight.shape)

    # Estimate model size in memory (assuming float32, 4 bytes per parameter)
    total_size_bytes = total_params * 4
    total_size_mb = total_size_bytes / (1024 * 1024)
    print(f"Estimated model size: {total_size_mb:.2f} MB")