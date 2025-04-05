"""
Main script to train the GPT-2 model from scratch.

This script orchestrates the training process by:
1. Setting up configuration and hyperparameters.
2. Loading and preparing the dataset.
3. Initializing the model, optimizer, and tokenizer.
4. Running the training loop.
5. Plotting training and validation losses.
"""

import torch
import tiktoken
import os
import urllib.request
import matplotlib.pyplot as plt

# Project specific imports - Relative imports assuming standard structure
from config import GPT_CONFIG_124M
from model.gpt2_model import GPTModel
from utils.data_utils import create_dataloader_v1
from utils.train_utils import train_model_simple


def main():
    """Main function to set up and run the training process."""

    # Set random seed for reproducibility across runs
    torch.manual_seed(123)

    # --- Configuration Loading & Setup ---
    # Start with the base configuration
    config = GPT_CONFIG_124M.copy() # Use .copy() to avoid modifying the original dict

    # --- Hyperparameter Overrides (Optional) ---
    # Modify specific config settings for this particular training run
    # config["context_length"] = 256 # Use a smaller context length (e.g., for faster training)
    # config["drop_rate"] = 0.0      # Disable dropout for initial testing/faster convergence check

    # --- Training Hyperparameters ---
    BATCH_SIZE = 4          # Number of sequences processed in parallel per step
    NUM_EPOCHS = 10         # Number of full passes over the training dataset
    LEARNING_RATE = 5e-4    # Step size for optimizer updates
    EVAL_FREQ = 100         # How often (in steps) to evaluate model performance
    EVAL_ITER = 10          # Number of batches to use during each evaluation
    START_CONTEXT = "Every step moves you" # Initial prompt for text generation samples
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # Automatically select GPU if available, else CPU

    print(f"Using device: {DEVICE}")
    print(f"Loaded configuration: {config}")
    print(f"Training Hyperparameters: Batch Size={BATCH_SIZE}, Epochs={NUM_EPOCHS}, LR={LEARNING_RATE}")

    # --- Data Loading and Preparation ---
    # Define file path and download URL for the dataset
    file_path = "the-verdict.txt"
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

    print(f"Loading data from {file_path}...")
    # Download the file if it doesn't exist locally
    if not os.path.exists(file_path):
        try:
            print(f"Downloading data from {url}...")
            with urllib.request.urlopen(url) as response:
                text_data = response.read().decode('utf-8') # Read and decode the text data
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(text_data) # Save the data locally
            print(f"Downloaded and saved data successfully.")
        except Exception as e:
            print(f"ERROR: Failed to download or save data: {e}")
            return # Stop execution if data cannot be obtained
    else:
        # Load data from the existing local file
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                text_data = file.read()
            print(f"Loaded data successfully from local file.")
        except Exception as e:
            print(f"ERROR: Failed to load data from {file_path}: {e}")
            return # Stop execution if data cannot be loaded

    # --- Data Splitting ---
    # Split the text data into training and validation sets
    train_ratio = 0.90 # 90% for training, 10% for validation
    split_idx = int(train_ratio * len(text_data))

    # Basic check to ensure splits are large enough for the context window
    if split_idx <= config["context_length"] or (len(text_data) - split_idx) <= config["context_length"]:
        print("ERROR: Dataset is too small for the specified context length and train/validation split.")
        print(f"  Required minimum length: > {config['context_length']} tokens for both train and validation.")
        return

    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]
    print(f"Data split: {len(train_data)} training characters, {len(val_data)} validation characters.")

    # --- Tokenizer Initialization ---
    # Initialize the tokenizer used for GPT-2
    print("Initializing tokenizer...")
    try:
        tokenizer = tiktoken.get_encoding("gpt2")
    except Exception as e:
        print(f"ERROR: Failed to initialize tokenizer: {e}")
        return
    print("Tokenizer initialized.")

    # --- DataLoader Creation ---
    print("Creating DataLoaders...")
    try:
        # Use a stride equal to context length for non-overlapping sequences
        stride = config["context_length"]
        # Create DataLoader for training data
        train_loader = create_dataloader_v1(
            train_data, batch_size=BATCH_SIZE, max_length=config["context_length"],
            stride=stride, drop_last=True, shuffle=True, num_workers=0, tokenizer=tokenizer
        )
        # Create DataLoader for validation data
        val_loader = create_dataloader_v1(
            val_data, batch_size=BATCH_SIZE, max_length=config["context_length"],
            stride=stride, drop_last=False, shuffle=False, num_workers=0, tokenizer=tokenizer
        )
        print(f"DataLoaders created: Train batches={len(train_loader)}, Val batches={len(val_loader)}")
    except ValueError as e:
        # Handle errors during dataset/dataloader creation (e.g., empty dataset)
        print(f"ERROR creating DataLoader: {e}")
        return
    except Exception as e:
        # Catch any other unexpected errors
        print(f"ERROR: An unexpected error occurred during DataLoader creation: {e}")
        return


    # --- Model Initialization ---
    print("Initializing model...")
    try:
        model = GPTModel(config) # Create an instance of the GPTModel
        model.to(DEVICE) # Move the model's parameters to the selected device (GPU or CPU)
        print(f"Model initialized successfully and moved to {DEVICE}.")
    except Exception as e:
        print(f"ERROR: Failed to initialize model: {e}")
        return

    # --- Optimizer Setup ---
    # AdamW is a common optimizer choice for Transformer models
    print("Setting up optimizer (AdamW)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    print("Optimizer initialized.")

    # --- Execute Training --- # 
    print("Starting training process...")
    try:
        train_losses, val_losses, tokens_seen = train_model_simple(
            model=model, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer,
            device=DEVICE, num_epochs=NUM_EPOCHS, eval_freq=EVAL_FREQ, eval_iter=EVAL_ITER,
            start_context=START_CONTEXT, tokenizer=tokenizer
        )
    except Exception as e:
        print(f"ERROR: An error occurred during training: {e}")
        # Optionally save checkpoint here before exiting
        return

    # --- Plotting Results (Optional) ---
    # Check if training produced any loss data to plot
    if train_losses and val_losses:
        print("Training complete. Plotting losses...")
        # Create x-axis values based on evaluation steps
        eval_steps = range(0, len(train_losses) * EVAL_FREQ, EVAL_FREQ)

        # Create the plot
        plt.figure(figsize=(10, 5))
        plt.plot(eval_steps, train_losses, label="Avg Train Loss per Eval")
        plt.plot(eval_steps, val_losses, label="Avg Validation Loss per Eval")
        plt.title("Training and Validation Losses")
        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

        # Save the plot to a file
        plot_filename = "training_losses.png"
        try:
            plt.savefig(plot_filename)
            print(f"Loss plot saved to {plot_filename}")
            # plt.show() # Uncomment to display the plot window immediately
        except Exception as e:
            print(f"ERROR saving plot: {e}")
    else:
        print("Training completed, but no loss data recorded for plotting.")

    print("Script finished successfully.")

# Standard Python entry point check
if __name__ == "__main__":
    main() # Call the main execution function 