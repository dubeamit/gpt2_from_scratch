"""
Script for running inference using the fine-tuned spam classifier model.
"""

import torch
import tiktoken
import argparse

# Project specific imports
from model.gpt2_model import GPTModel # Need the model definition
# from config import GPT_CONFIG_124M 

def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    """Classifies a single text review as spam or not spam using the fine-tuned model.

    Args:
        text (str): The input text message.
        model (GPTModel): The loaded fine-tuned GPT model.
        tokenizer: An initialized tiktoken tokenizer.
        device: The torch device ('cuda' or 'cpu').
        max_length (int, optional): Maximum sequence length for padding/truncation. 
                                   If None, attempts to use model's config context_length.
                                   Defaults to None.
        pad_token_id (int, optional): Token ID for padding. Defaults to 50256.

    Returns:
        str: The predicted label ('spam' or 'not spam (ham)').
    """
    model.eval() # Ensure model is in evaluation mode

    # Tokenize the input text
    input_ids = tokenizer.encode(text)

    # Determine the maximum length for padding/truncation
    if max_length is None:
         # Try to get context length from model config
         if hasattr(model, 'cfg') and 'context_length' in model.cfg:
            max_length = model.cfg["context_length"]
         else:
             # Fallback if config not available or doesn't have the key
             print("Warning: max_length not specified and cannot get from model config. Using default 256.")
             max_length = 256

    # Truncate the sequence if it exceeds max_length
    input_ids = input_ids[:max_length]

    # Pad the sequence if it's shorter than max_length
    padded_length = max_length # Use the determined max_length for padding
    if len(input_ids) < padded_length:
        input_ids += [pad_token_id] * (padded_length - len(input_ids))

    # Convert to tensor, add batch dimension, and move to device
    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        # Get logits for the last token only, as required for classification head
        logits = model(input_tensor)[:, -1, :]

    # Get the index of the highest logit (predicted class index)
    predicted_label = torch.argmax(logits, dim=-1).item()

    # Return the corresponding string label
    return 'spam' if predicted_label == 1 else 'not spam (ham)'


def load_model_for_inference(checkpoint_path, device):
    """Loads the fine-tuned model state dict and configuration.

    Args:
        checkpoint_path (str): Path to the saved .pth checkpoint file.
        device: The torch device to load the model onto.

    Returns:
        tuple[GPTModel, dict] | tuple[None, None]: A tuple containing the loaded model 
                                                   and its configuration, or (None, None) if loading fails.
    """
    try:
        # Load the checkpoint dictionary
        checkpoint = torch.load(checkpoint_path, map_location=device)
        print(f"Loaded checkpoint from {checkpoint_path}")
    except FileNotFoundError:
        print(f"ERROR: Checkpoint file not found at {checkpoint_path}")
        return None, None
    except Exception as e:
        print(f"ERROR loading checkpoint: {e}")
        return None, None

    # Extract config and model state dict
    if 'config' not in checkpoint:
        print("ERROR: Checkpoint does not contain 'config'. Cannot initialize model.")
        return None, None
    if 'model_state_dict' not in checkpoint:
        print("ERROR: Checkpoint does not contain 'model_state_dict'.")
        return None, None
        
    cfg = checkpoint['config']
    state_dict = checkpoint['model_state_dict']

    # Initialize the model with the saved configuration
    try:
        model = GPTModel(cfg)
        print("Model initialized with saved configuration.")
    except Exception as e:
        print(f"ERROR initializing model from saved config: {e}")
        return None, None
    
    # --- Adapt the model for classification BEFORE loading state_dict ---
    # Get embedding dimension and define number of classes
    emb_dim = cfg.get("emb_dim", None) 
    if emb_dim is None:
        print("ERROR: Cannot find 'emb_dim' in the loaded model configuration.")
        return None, None
    num_classes = 2 # Hardcoded for binary spam classification

    # Replace the output head
    model.out_head = torch.nn.Linear(in_features=emb_dim, out_features=num_classes)
    print(f"Replaced model output head for {num_classes}-class classification.")
    # -------------------------------------------------------------------

    # Load the state dict
    try:
        # Load with strict=False initially if needed, then investigate missing/unexpected keys
        # model.load_state_dict(state_dict, strict=False) 
        model.load_state_dict(state_dict) 
        model.to(device) # Move model to the correct device
        model.eval() # Set to evaluation mode
        print("Model state dict loaded successfully.")
        return model, cfg
    except Exception as e:
        print(f"ERROR loading state dict into model: {e}")
        return None, None


if __name__ == "__main__":
    # --- Argument Parsing --- #
    parser = argparse.ArgumentParser(description="Classify SMS messages using a fine-tuned GPT model.")
    parser.add_argument("text", type=str, help="The SMS message text to classify.")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="spam_classifier_finetuned.pth", 
        help="Path to the fine-tuned model checkpoint (.pth file)."
    )
    parser.add_argument(
        "--max_len", 
        type=int, 
        default=None, 
        help="Maximum sequence length for input. Defaults to model config context_length or 256."
    )
    args = parser.parse_args()

    # --- Setup --- #
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # --- Load Model --- #
    print(f"Loading model from {args.model_path}...")
    model, model_cfg = load_model_for_inference(args.model_path, DEVICE)

    if model is None:
        print("Exiting due to model loading failure.")
        exit(1)

    # --- Initialize Tokenizer --- #
    try:
        tokenizer = tiktoken.get_encoding("gpt2")
        print("Tokenizer initialized.")
    except Exception as e:
        print(f"ERROR initializing tokenizer: {e}")
        exit(1)

    # --- Perform Classification --- #
    print("Classifying text...")
    # Determine max_length, potentially using loaded config
    max_length_for_inference = args.max_len if args.max_len is not None else model_cfg.get('context_length', None)
    
    prediction = classify_review(
        text=args.text,
        model=model,
        tokenizer=tokenizer,
        device=DEVICE,
        max_length=max_length_for_inference # Pass the determined max_length
    )

    # --- Print Result --- #
    print("--- Result ---")
    print(f"Input Text: \"{args.text}\"")
    print(f"Prediction: {prediction}")
    print("-------------") 

# python classify_inference.py 'You are a winner you have been specially selected to receive $1000 cash or a $2000 award.'
# python classify_inference.py "Hey, just wanted to check if we're still on for dinner tonight? Let me know!"
