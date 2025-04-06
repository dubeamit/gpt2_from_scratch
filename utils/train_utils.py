"""
Utility functions for training, evaluation, and text generation.

Includes helpers for tokenization, greedy text generation, loss calculation,
model evaluation on datasets, and the main training loop.
"""

import torch
import tiktoken
import torch.nn.functional as F
import numpy as np # Add numpy for weight loading

# --- Text Generation Helpers --- #

def text_to_token_ids(text, tokenizer):
    """Encodes a string of text into token IDs using the provided tokenizer.

    Args:
        text (str): The input text string.
        tokenizer: An initialized tiktoken tokenizer instance.

    Returns:
        torch.Tensor: A tensor containing the token IDs (shape: [1, sequence_length]).
    """
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    # Add a batch dimension (1, T) as the model expects batch input
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    """Decodes a tensor of token IDs back into a string of text.

    Args:
        token_ids (torch.Tensor): Tensor of token IDs (usually shape [1, sequence_length]).
        tokenizer: An initialized tiktoken tokenizer instance.

    Returns:
        str: The decoded text string.
    """
    # Remove the batch dimension if present (1, T) -> (T)
    flat = token_ids.squeeze(0)
    # Decode the list of token IDs
    return tokenizer.decode(flat.tolist())

def generate_text_simple(model, idx, max_new_tokens, context_size):
    """Generates text sequentially using the model via greedy decoding.

    Takes the starting sequence `idx`, predicts the next token, appends it,
    and repeats until `max_new_tokens` are generated.
    Truncates context if it exceeds `context_size`.

    Args:
        model (nn.Module): The GPT model instance.
        idx (torch.Tensor): The starting sequence of token indices (B, T).
        max_new_tokens (int): The maximum number of new tokens to generate.
        context_size (int): The maximum sequence length the model can handle.

    Returns:
        torch.Tensor: The generated sequence of token indices (B, T + max_new_tokens).
    """
    # Loop to generate the specified number of new tokens
    for _ in range(max_new_tokens):
        # Crop the context sequence if it exceeds the model's context size
        idx_cond = idx[:, -context_size:]

        # Get model predictions (logits) without calculating gradients
        with torch.no_grad():
            logits = model(idx_cond) # (B, T_cropped, vocab_size)

        # Focus only on the logits for the last token in the sequence
        # Shape: (B, T_cropped, vocab_size) -> (B, vocab_size)
        logits = logits[:, -1, :]

        # Convert logits to probabilities using softmax
        probas = torch.softmax(logits, dim=-1) # Shape: (B, vocab_size)

        # Perform greedy decoding: select the token with the highest probability
        # keepdim=True maintains the batch dimension, resulting in shape (B, 1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)

        # Append the newly generated token index to the sequence
        # Shape: (B, T) cat (B, 1) -> (B, T+1)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

# --- Loss Calculation --- #

def calc_loss_batch(input_batch, target_batch, model, device):
    """Calculates the cross-entropy loss for a single batch of data.

    Moves data to the specified device, performs a forward pass, and computes loss.

    Args:
        input_batch (torch.Tensor): Input token IDs (B, T).
        target_batch (torch.Tensor): Target token IDs (B, T).
        model (nn.Module): The GPT model.
        device: The torch device (e.g., 'cuda' or 'cpu').

    Returns:
        torch.Tensor: The calculated cross-entropy loss for the batch (scalar tensor).
    """
    # Move batch to the target device (e.g., "cuda" or "cpu")
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    # Perform model forward pass
    logits = model(input_batch) # Shape: (B, T, vocab_size)
    # Calculate cross-entropy loss
    # `logits` needs to be (B*T, vocab_size) and `target_batch` needs to be (B*T)
    loss = F.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    """Calculates the average loss over batches from a DataLoader.

    Iterates through the DataLoader, calculates batch loss, and averages.
    Optionally evaluates only a specified number of batches.
    Sets model to eval mode during calculation.

    Args:
        data_loader (DataLoader): The DataLoader to evaluate.
        model (nn.Module): The GPT model.
        device: The torch device.
        num_batches (int, optional): Number of batches to evaluate. If None, evaluates all.
                                    Defaults to None.

    Returns:
        float: The average loss over the evaluated batches. Returns float('nan')
               if the loader is empty or no batches are evaluated.
    """
    total_loss = 0.
    # Basic checks for valid DataLoader
    if not data_loader:
        print("Warning: Invalid or empty DataLoader passed to calc_loss_loader.")
        return float('nan')
    if len(data_loader) == 0:
        print("Warning: DataLoader has zero length in calc_loss_loader.")
        return float('nan')

    # Determine the number of batches to evaluate
    actual_num_batches = len(data_loader)
    if num_batches is None:
        num_batches_to_eval = actual_num_batches
    else:
        num_batches_to_eval = min(num_batches, actual_num_batches)

    # Check if evaluation is possible
    if num_batches_to_eval == 0:
        print("Warning: Zero batches selected for evaluation in calc_loss_loader.")
        return float('nan')

    evaluated_batches = 0
    model.eval() # Set model to evaluation mode (disables dropout etc.)
    with torch.no_grad(): # Disable gradient computation for efficiency
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i >= num_batches_to_eval:
                break # Stop after evaluating the specified number of batches
            try:
                # Calculate loss for the current batch
                loss = calc_loss_batch(input_batch, target_batch, model, device)
                total_loss += loss.item() # Accumulate loss (use .item() to get scalar)
                evaluated_batches += 1
            except Exception as e:
                # Handle potential errors during batch processing
                print(f"Error calculating loss for batch {i} in calc_loss_loader: {e}")
                # Depending on severity, might skip or return NaN
    model.train() # IMPORTANT: Set model back to training mode

    # Check if any batches were successfully evaluated
    if evaluated_batches == 0:
        print("Warning: No batches were successfully processed in calc_loss_loader.")
        return float('nan') # Avoid division by zero if loop didn't run

    # Return the average loss over evaluated batches
    return total_loss / evaluated_batches

# --- Evaluation During Training --- #

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """Evaluates the model's average loss on training and validation sets.

    Calls `calc_loss_loader` for both data loaders.
    Ensures the model is in evaluation mode during the process.

    Args:
        model (nn.Module): The GPT model.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        device: The torch device.
        eval_iter (int): Number of batches to use for evaluation from each loader.

    Returns:
        tuple[float, float]: A tuple containing (average train loss, average validation loss).
    """
    model.eval() # Set model to evaluation mode
    with torch.no_grad(): # Ensure no gradients are calculated
        # Calculate average loss on a subset of training data
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        # Calculate average loss on a subset of validation data
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train() # Set model back to training mode
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    """Generates a text sample using the model and prints it.

    Uses `generate_text_simple` for generation and handles tokenization/detokenization.
    Ensures model is in evaluation mode during generation.

    Args:
        model (nn.Module): The GPT model instance.
        tokenizer: An initialized tiktoken tokenizer.
        device: The torch device.
        start_context (str): The initial text prompt for generating samples.
    """
    model.eval() # Set model to evaluation mode

    # Determine context size from model config if possible
    if hasattr(model, 'cfg') and 'context_length' in model.cfg:
        context_size = model.cfg["context_length"]
    else:
        # Provide a fallback if config is missing
        print("Warning: Model config not found or lacks 'context_length'. Using default 256 for generation.")
        context_size = 256

    # Encode the starting prompt
    encoded = text_to_token_ids(start_context, tokenizer).to(device)

    # Generate new tokens
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
        # Decode the full sequence back to text
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        # Extract only the newly generated part
        generated_part = decoded_text.replace(start_context, '').strip()
        # Print the prompt and the generated completion
        print(f'Prompt: "{start_context}" ---> Generated: "{generated_part}"')

    model.train() # Set model back to training mode

# --- Main Training Loop --- #

def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    """Main function to train the GPT model.

    Iterates through epochs and batches, performs training steps, evaluates periodically,
    and generates sample text.

    Args:
        model (nn.Module): The GPT model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        optimizer (torch.optim.Optimizer): The PyTorch optimizer instance.
        device: The torch device ('cuda' or 'cpu').
        num_epochs (int): Number of epochs to train for.
        eval_freq (int): Frequency (in steps) to perform evaluation.
        eval_iter (int): Number of batches to use for evaluation loss calculation.
        start_context (str): Initial text prompt for generating samples during training.
        tokenizer: An initialized tiktoken tokenizer instance.

    Returns:
        tuple[list[float], list[float], list[int]]: A tuple containing lists of
            (train losses, validation losses, tokens seen at evaluation points).
    """
    # Lists to store losses for plotting later
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # Start the training process
    print("---- Starting Training ----")
    for epoch in range(num_epochs):
        model.train() # Ensure model is in training mode at the start of each epoch

        epoch_loss = 0.0 # Accumulator for average loss per epoch
        num_batches_epoch = len(train_loader)

        # Check if the dataloader is empty for this epoch
        if num_batches_epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs}: Skipping - train_loader is empty.")
            continue # Skip to the next epoch

        # Iterate through batches in the training data loader
        for i, (input_batch, target_batch) in enumerate(train_loader):
            optimizer.zero_grad() # Clear gradients from the previous step

            try:
                # Calculate loss for the current batch
                loss = calc_loss_batch(input_batch, target_batch, model, device)
                # Compute gradients w.r.t. model parameters
                loss.backward()
                # Update model parameters based on gradients
                optimizer.step()

                # Track progress
                tokens_seen += input_batch.numel() # Count tokens processed
                global_step += 1
                epoch_loss += loss.item() # Accumulate batch loss for epoch average

            except Exception as e:
                # Basic error handling for a training step failure
                print(f"ERROR during training step {global_step} (Batch {i}): {e}")
                # Options: break, continue, log, etc.
                continue # Skip to the next batch for now

            # --- Periodic Evaluation ---
            if global_step % eval_freq == 0:
                # Evaluate model performance on train and validation subsets
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)

                # Store losses if they are valid numbers
                if not (torch.isnan(torch.tensor(train_loss)) or torch.isnan(torch.tensor(val_loss))):
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)
                    # Print evaluation results
                    print(f"Epoch {epoch+1:02d}/{num_epochs:02d} | Step {global_step:06d} | "
                          f"Batch {i+1:04d}/{num_batches_epoch:04d} | "
                          f"Avg Train Loss: {train_loss:.3f} | Avg Val Loss: {val_loss:.3f}")
                else:
                    # Handle cases where evaluation might return NaN
                    print(f"Epoch {epoch+1:02d}/{num_epochs:02d} | Step {global_step:06d} | "
                          f"Batch {i+1:04d}/{num_batches_epoch:04d} | "
                          f"Evaluation failed (Loss is NaN).")

        # --- End of Epoch --- #
        # Calculate and print average training loss for the completed epoch
        avg_epoch_loss = epoch_loss / num_batches_epoch if num_batches_epoch > 0 else float('nan')
        print(f"--- Epoch {epoch+1} Completed | Average Epoch Training Loss: {avg_epoch_loss:.3f} ---")
        # Generate and print a sample text to observe model improvement
        generate_and_print_sample(model, tokenizer, device, start_context)
        print("-" * 50) # Separator for clarity

    print("---- Training Finished ----")
    return train_losses, val_losses, track_tokens_seen 

# --- Weight Loading Utilities --- #

def assign(left, right):
    """Helper function to assign NumPy array values to PyTorch parameters.

    Handles shape checking and tensor conversion.

    Args:
        left (torch.nn.Parameter): The PyTorch parameter to assign to.
        right (np.ndarray): The NumPy array containing the weights/biases.

    Returns:
        torch.nn.Parameter: The updated PyTorch parameter.

    Raises:
        ValueError: If the shapes of `left` and `right` do not match.
    """
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    # Convert NumPy array to PyTorch tensor and wrap in Parameter
    # Use .clone().detach() to ensure it's a new tensor unrelated to the numpy array's memory
    return torch.nn.Parameter(torch.tensor(right).clone().detach())

def load_weights_into_gpt(gpt_model, params):
    """Loads pre-trained GPT-2 weights from a dictionary into a GPTModel instance.

    Maps the parameter names from the downloaded checkpoint format (OpenAI's) 
    to the layer names in the custom GPTModel implementation.

    Args:
        gpt_model (GPTModel): An instance of the GPTModel class.
        params (dict): A dictionary containing the pre-trained weights (typically loaded from NumPy files).
                      Expected keys match the OpenAI checkpoint structure (e.g., 'wte', 'wpe', 'blocks').
    """
    print("Loading weights into model...")

    # Assign positional and token embeddings
    gpt_model.pos_emb.weight = assign(gpt_model.pos_emb.weight, params['wpe']) # Positional embeddings
    gpt_model.tok_emb.weight = assign(gpt_model.tok_emb.weight, params['wte']) # Token embeddings

    # Iterate through each Transformer block
    for b in range(len(params["blocks"])):
        # --- Attention Weights --- #
        # Load Query, Key, Value weights and biases
        # OpenAI's checkpoints store QKV weights combined in 'c_attn'. We split them.
        # Weights need transposing (.T) due to different dimension conventions.
        q_w, k_w, v_w = np.split(params["blocks"][b]["attn"]["c_attn"]["w"], 3, axis=-1)
        gpt_model.trf_blocks[b].att.W_query.weight = assign(gpt_model.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt_model.trf_blocks[b].att.W_key.weight = assign(gpt_model.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt_model.trf_blocks[b].att.W_value.weight = assign(gpt_model.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(params["blocks"][b]["attn"]["c_attn"]["b"], 3, axis=-1)
        # Ensure the model's Linear layers have bias=True when loading these
        if gpt_model.trf_blocks[b].att.W_query.bias is not None:
            gpt_model.trf_blocks[b].att.W_query.bias = assign(gpt_model.trf_blocks[b].att.W_query.bias, q_b)
            gpt_model.trf_blocks[b].att.W_key.bias = assign(gpt_model.trf_blocks[b].att.W_key.bias, k_b)
            gpt_model.trf_blocks[b].att.W_value.bias = assign(gpt_model.trf_blocks[b].att.W_value.bias, v_b)
        else:
            print(f"Warning: Skipping QKV bias loading for block {b} as model bias is None.")

        # Load Attention output projection weights and biases
        gpt_model.trf_blocks[b].att.out_proj.weight = assign(
            gpt_model.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T # Transpose needed
        )
        if gpt_model.trf_blocks[b].att.out_proj.bias is not None:
            gpt_model.trf_blocks[b].att.out_proj.bias = assign(
                gpt_model.trf_blocks[b].att.out_proj.bias,
                params["blocks"][b]["attn"]["c_proj"]["b"]
            )
        else:
             print(f"Warning: Skipping Attn output projection bias loading for block {b} as model bias is None.")

        # --- FeedForward Weights --- #
        # Load first linear layer (expansion) weights and biases
        gpt_model.trf_blocks[b].ff.layers[0].weight = assign(
            gpt_model.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T # Transpose needed
        )
        if gpt_model.trf_blocks[b].ff.layers[0].bias is not None:
            gpt_model.trf_blocks[b].ff.layers[0].bias = assign(
                gpt_model.trf_blocks[b].ff.layers[0].bias,
                params["blocks"][b]["mlp"]["c_fc"]["b"]
            )
        else:
            print(f"Warning: Skipping FFN layer 0 bias loading for block {b} as model bias is None.")

        # Load second linear layer (projection) weights and biases
        gpt_model.trf_blocks[b].ff.layers[2].weight = assign(
            gpt_model.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T # Transpose needed
        )
        if gpt_model.trf_blocks[b].ff.layers[2].bias is not None:
            gpt_model.trf_blocks[b].ff.layers[2].bias = assign(
                gpt_model.trf_blocks[b].ff.layers[2].bias,
                params["blocks"][b]["mlp"]["c_proj"]["b"]
            )
        else:
             print(f"Warning: Skipping FFN layer 2 bias loading for block {b} as model bias is None.")

        # --- Layer Normalization Weights --- #
        # Load LayerNorm scales (gamma) and shifts (beta)
        gpt_model.trf_blocks[b].norm1.scale = assign(gpt_model.trf_blocks[b].norm1.scale, params["blocks"][b]["ln_1"]["g"])
        gpt_model.trf_blocks[b].norm1.shift = assign(gpt_model.trf_blocks[b].norm1.shift, params["blocks"][b]["ln_1"]["b"])
        gpt_model.trf_blocks[b].norm2.scale = assign(gpt_model.trf_blocks[b].norm2.scale, params["blocks"][b]["ln_2"]["g"])
        gpt_model.trf_blocks[b].norm2.shift = assign(gpt_model.trf_blocks[b].norm2.shift, params["blocks"][b]["ln_2"]["b"])

    # --- Final Layer Normalization Weights --- #
    gpt_model.final_norm.scale = assign(gpt_model.final_norm.scale, params["g"])
    gpt_model.final_norm.shift = assign(gpt_model.final_norm.shift, params["b"])

    # --- Output Head Weight Tying --- #
    # Assign the token embedding weights to the output layer (weight tying)
    gpt_model.out_head.weight = assign(gpt_model.out_head.weight, params["wte"])

    print("Finished loading weights.") 