"""
Main script for fine-tuning the pre-trained GPT-2 model on the SMS Spam Classification task.
"""

import urllib.request
import ssl
import zipfile
import os
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken 
import time
from pretrained_gpt2 import main
from utils.classifier_utils import (
    prepare_spam_dataset, 
    SpamDataset, 
    train_classifier_simple, 
    plot_values, 
    calc_accuracy_loader, 
    classify_review
)

url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_path = "sms_spam_collection.zip"
extracted_path = "sms_spam_collection"
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download and extraction.")
        return

    # Create an unverified SSL context
    ssl_context = ssl._create_unverified_context()

    # Downloading the file
    with urllib.request.urlopen(url, context=ssl_context) as response:
        with open(zip_path, "wb") as out_file:
            out_file.write(response.read())

    # Unzipping the file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)

    # Add .tsv file extension
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"File downloaded and saved as {data_file_path}")

download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)

df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
print(df.head())
print(df["Label"].value_counts())

def create_balanced_dataset(df):
    
    # Count the instances of "spam"
    num_spam = df[df["Label"] == "spam"].shape[0]
    
    # Randomly sample "ham" instances to match the number of "spam" instances
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
    
    # Combine ham "subset" with "spam"
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])

    return balanced_df

balanced_df = create_balanced_dataset(df)
print(balanced_df["Label"].value_counts())
#  convert the "string" class labels "ham" and "spam" into integer class labels 0 and 1
balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

def random_split(df, train_frac, validation_frac):
    # Shuffle the entire DataFrame
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    # Calculate split indices
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    # Split the DataFrame
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df

train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)

train_df.to_csv("train.csv", index=None)
validation_df.to_csv("validation.csv", index=None)
test_df.to_csv("test.csv", index=None)

def main_finetune():
    """Orchestrates the fine-tuning process for spam classification."""

    # --- Configuration & Hyperparameters --- #
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # Data parameters
    DATA_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    ZIP_PATH = "sms_spam_collection.zip"
    EXTRACTED_PATH = "sms_spam_collection"
    DATA_FILE_NAME = "SMSSpamCollection.tsv" # Changed name slightly for clarity
    DATA_FILE_PATH = Path(EXTRACTED_PATH) / DATA_FILE_NAME
    TRAIN_CSV_PATH = "train.csv"
    VAL_CSV_PATH = "validation.csv"
    TEST_CSV_PATH = "test.csv"

    # Fine-tuning hyperparameters
    NUM_EPOCHS = 5
    LEARNING_RATE = 5e-5 # Lower LR typically used for fine-tuning
    BATCH_SIZE = 8
    EVAL_FREQ = 50 # Evaluate every N steps during training
    EVAL_ITER = 5 # Number of batches for evaluation loss/accuracy calculation
    WEIGHT_DECAY = 0.1 # Weight decay for AdamW optimizer

    # Model loading parameters
    PRETRAINED_MODEL_SIZE = '124M'
    MODELS_DIR = 'gpt2'

    # --- 1. Prepare Dataset --- #
    print("--- Preparing Dataset --- ")
    prepare_spam_dataset(DATA_URL, ZIP_PATH, EXTRACTED_PATH, DATA_FILE_PATH)
    # Basic check if CSV files were created
    if not (os.path.exists(TRAIN_CSV_PATH) and os.path.exists(VAL_CSV_PATH) and os.path.exists(TEST_CSV_PATH)):
        print("ERROR: Required CSV files (train/validation/test.csv) not found. Exiting.")
        return
    print("Dataset preparation complete.")

    # --- 2. Load Pre-trained Model --- #
    print("--- Loading Pre-trained Model --- ")
    # Call the main function from pretrained_gpt2, disable its generation test
    model = main(model_size=PRETRAINED_MODEL_SIZE, models_dir=MODELS_DIR, run_generation_test=False)
    if model is None:
        print("ERROR: Failed to load pre-trained model. Exiting.")
        return
    model.to(DEVICE) # Ensure model is on the correct device
    print(f"Pre-trained model {PRETRAINED_MODEL_SIZE} loaded and moved to {DEVICE}.")

    # --- 3. Adapt Model for Classification --- #
    print("--- Adapting Model for Classification --- ")
    # Freeze all existing parameters
    for param in model.parameters():
        param.requires_grad = False
    print("Froze all base model parameters.")

    # Replace the output head with a new one for binary classification
    torch.manual_seed(123) # Seed for reproducible head initialization
    num_classes = 2
    # Get embedding dimension from the loaded model's config
    emb_dim = model.cfg["emb_dim"]
    model.out_head = torch.nn.Linear(in_features=emb_dim, out_features=num_classes)
    model.out_head.to(DEVICE) # Ensure the new head is on the correct device
    print(f"Replaced output head with a new Linear layer ({emb_dim} -> {num_classes}).")

    # Unfreeze the parameters of the new output head
    for param in model.out_head.parameters():
        param.requires_grad = True
    print("Unfroze output head parameters.")

    # Optionally unfreeze more layers for deeper fine-tuning (e.g., last block)
    # Unfreeze the last transformer block and the final layer norm
    if model.cfg["n_layers"] > 0: # Check if there are layers to unfreeze
        for param in model.trf_blocks[-1].parameters():
            param.requires_grad = True
        print("Unfroze parameters of the last Transformer block.")
    else:
         print("No Transformer blocks found to unfreeze.")
         
    for param in model.final_norm.parameters():
        param.requires_grad = True
    print("Unfroze parameters of the final LayerNorm.")

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters after adaptation: {trainable_params:,}")

    # --- 4. Create DataLoaders --- #
    print("--- Creating DataLoaders for Fine-tuning --- ")
    try:
        tokenizer = tiktoken.get_encoding("gpt2")
        # Use max_length determined from training data for consistency
        # First create train_dataset to find max_length if not hardcoded
        temp_train_dataset = SpamDataset(csv_file=TRAIN_CSV_PATH, tokenizer=tokenizer)
        max_len = temp_train_dataset.max_length
        del temp_train_dataset # Free memory
        print(f"Using max_length: {max_len} for all datasets.")

        train_dataset = SpamDataset(csv_file=TRAIN_CSV_PATH, max_length=max_len, tokenizer=tokenizer)
        val_dataset = SpamDataset(csv_file=VAL_CSV_PATH, max_length=max_len, tokenizer=tokenizer)
        test_dataset = SpamDataset(csv_file=TEST_CSV_PATH, max_length=max_len, tokenizer=tokenizer)

        num_workers = 0 # Keep simple for now
        torch.manual_seed(123) # Seed for DataLoader shuffling

        train_loader = DataLoader(
            dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=num_workers, drop_last=True
        )
        val_loader = DataLoader(
            dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=num_workers, drop_last=False
        )
        test_loader = DataLoader(
            dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=num_workers, drop_last=False
        )
        print("DataLoaders created successfully.")

    except Exception as e:
        print(f"ERROR creating DataLoaders: {e}")
        return

    # --- 5. Initialize Optimizer --- #
    print("--- Initializing Optimizer --- ")
    # Optimizer targets only the currently trainable parameters
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    print(f"Optimizer AdamW initialized with LR={LEARNING_RATE}, Weight Decay={WEIGHT_DECAY}.")

    # --- 6. Run Fine-tuning --- # 
    print("--- Starting Fine-tuning --- ")
    start_time = time.time()

    train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
        model, train_loader, val_loader, optimizer, DEVICE,
        num_epochs=NUM_EPOCHS, eval_freq=EVAL_FREQ, eval_iter=EVAL_ITER
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Fine-tuning completed in {execution_time_minutes:.2f} minutes.")

    # --- 7. Plot Results --- #
    if train_losses and val_losses:
        epochs_tensor = torch.linspace(0, NUM_EPOCHS, len(train_losses))
        # Create examples seen tensor based on steps and batch size (approximation)
        # examples_seen_plot = torch.linspace(0, examples_seen, len(train_losses))
        # Or use steps directly for x-axis
        steps_tensor = torch.arange(0, len(train_losses)*EVAL_FREQ, EVAL_FREQ)
        plot_values(steps_tensor, steps_tensor, train_losses, val_losses, label="loss")

    if train_accs and val_accs:
        epochs_tensor_acc = torch.linspace(0, NUM_EPOCHS, len(train_accs))
        # steps_tensor_acc = torch.arange(EVAL_FREQ, len(train_accs)*EVAL_FREQ + 1, EVAL_FREQ)
        plot_values(epochs_tensor_acc, epochs_tensor_acc * len(train_loader) * BATCH_SIZE, train_accs, val_accs, label="accuracy")

    # --- 8. Evaluate on Test Set --- #
    print("--- Evaluating on Test Set --- ")
    test_accuracy = calc_accuracy_loader(test_loader, model, DEVICE)
    print(f"Test accuracy: {test_accuracy*100:.2f}%" ) # Removed extra space

    # --- 9. Example Classification --- #
    print("--- Classifying Example Texts --- ")
    text_1 = (
        "You are a winner you have been specially"
        " selected to receive $1000 cash or a $2000 award."
    )
    text_2 = (
        "Hey, just wanted to check if we're still on"
        " for dinner tonight? Let me know!"
    )
    print(f"'{text_1}' -> {classify_review(text_1, model, tokenizer, DEVICE, max_length=max_len)}")
    print(f"'{text_2}' -> {classify_review(text_2, model, tokenizer, DEVICE, max_length=max_len)}")

    # --- 10. Save Fine-tuned Model --- #
    print("--- Saving Fine-tuned Model --- ")
    model_save_path = "spam_classifier_finetuned.pth"
    try:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': model.cfg # Save the config used by the model
            # Add other info if needed, e.g., epoch, loss
            },
            model_save_path
        )
        print(f"Fine-tuned model saved to {model_save_path}")
    except Exception as e:
        print(f"ERROR saving model: {e}")

    print("Fine-tuning script finished.")

# --- Main execution guard --- # 
if __name__ == "__main__":
    main_finetune()