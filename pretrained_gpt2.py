"""
Example script demonstrating how to load pre-trained GPT-2 weights 
into the custom GPTModel implementation and optionally test generation.
"""

import torch
import tiktoken # For tokenizer loading (if needed for testing)

# Project specific imports
from config import GPT_CONFIG_124M
from model.gpt2_model import GPTModel
from gpt_download import download_and_load_gpt2 # Function to download weights
from utils.train_utils import load_weights_into_gpt, text_to_token_ids, token_ids_to_text, generate_text_simple # Import the loading function and generation utils


def main(model_size='124M', models_dir='gpt2', run_generation_test=False):
    """Downloads weights, initializes model, loads weights, and returns the model.

    Optionally runs a sample generation test.

    Args:
        model_size (str): The size of the GPT-2 model to download ('124M', '355M', etc.).
        models_dir (str): The directory to download model files into.
        run_generation_test (bool): If True, run a simple text generation test after loading.

    Returns:
        GPTModel | None: The initialized and weight-loaded GPTModel instance, or None if an error occurs.
    """

    print(f"Attempting to download and load GPT-2 model: {model_size}")
    # --- 1. Download Pre-trained Weights --- #
    try:
        settings, params = download_and_load_gpt2(model_size=model_size, models_dir=models_dir)
        print(f"Successfully loaded pre-trained weights and settings for {model_size}.")
        print("Settings:", settings)
        print("Parameter dictionary keys:", params.keys())
    except Exception as e:
        print(f"ERROR: Failed to download or load weights: {e}")
        return None # Return None on failure

    # --- 2. Configure Model --- #
    model_config = GPT_CONFIG_124M.copy()
    model_config.update({
        "context_length": settings.get("n_ctx", 1024),
        "qkv_bias": True,       # Pre-trained GPT-2 weights include bias terms in QKV layers
        "emb_dim": settings.get("n_embd", 768),
        "n_layers": settings.get("n_layer", 12),
        "n_heads": settings.get("n_head", 12)
    })
    print(f"Initializing model with configuration: {model_config}")

    # --- 3. Initialize Model --- #
    try:
        model = GPTModel(model_config)
        model.eval() # Set to evaluation mode (disables dropout)
        print("Custom GPTModel initialized successfully.")
    except Exception as e:
        print(f"ERROR: Failed to initialize GPTModel: {e}")
        return None # Return None on failure

    # --- 4. Load Weights --- #
    try:
        load_weights_into_gpt(model, params)
        print("Weights loaded successfully.") # Confirmation message
    except Exception as e:
        print(f"ERROR: An unexpected error occurred during weight loading: {e}")
        return None # Return None on failure

    # --- 5. Optional: Test Generation --- #
    if run_generation_test:
        print("--- Running Optional Generation Test ---")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print(f"Model moved to device: {device}")
        try:
            tokenizer = tiktoken.get_encoding("gpt2")
            print("Tokenizer initialized.")
        except Exception as e:
            print(f"ERROR initializing tokenizer: {e}")
            return model # Return model even if tokenizer fails

        torch.manual_seed(123)
        start_context = "Hello, I am a language model,"
        max_new = 50
        print(f"Generating text sample (max {max_new} tokens)...")
        try:
            encoded_context = text_to_token_ids(start_context, tokenizer).to(device)
            generated_ids = generate_text_simple(
                model=model, idx=encoded_context,
                max_new_tokens=max_new, context_size=model_config["context_length"]
            )
            generated_text = token_ids_to_text(generated_ids, tokenizer)
            print("---")
            print(f"Context: {start_context}")
            print(f"Generated: {generated_text}")
            print("---")
        except Exception as e:
            print(f"ERROR during text generation: {e}")
        print("--- Generation Test Complete ---")
        # Move model back to CPU if it was on GPU for the test, 
        # unless the main script will handle device placement later.
        # model.to("cpu") 

    # --- Return the loaded model --- #
    print("Returning loaded model.")
    return model


if __name__ == "__main__":
    # Example of running this script directly to test loading and generation
    loaded_model = main(model_size='124M', run_generation_test=True)
    if loaded_model:
        print("Script finished successfully, model returned.")
    else:
        print("Script finished with errors.")