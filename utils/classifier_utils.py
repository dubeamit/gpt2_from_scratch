"""
Utilities specific to the SMS Spam Classification task.

Includes data download/preprocessing, dataset class, evaluation metrics,
loss functions, training loop, plotting, and classification inference.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
import ssl
import zipfile
import os
from pathlib import Path
import time
import shutil

# --- Data Handling --- #

def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    """Downloads and prepares the SMS Spam Collection dataset."""
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download and extraction.")
        return

    print(f"Downloading dataset from {url}...")
    try:
        # Create an unverified SSL context to bypass potential SSL certificate issues
        ssl_context = ssl._create_unverified_context()
        with urllib.request.urlopen(url, context=ssl_context) as response, open(zip_path, "wb") as out_file:
            out_file.write(response.read())
    except Exception as e:
        print(f"ERROR: Failed to download {url}: {e}")
        return

    print(f"Unzipping {zip_path} to {extracted_path}...")
    try:
        os.makedirs(extracted_path, exist_ok=True) # Ensure directory exists
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extracted_path)
    except Exception as e:
        print(f"ERROR: Failed to unzip {zip_path}: {e}")
        return
    finally:
        # Clean up the zip file regardless of success
        if os.path.exists(zip_path):
            os.remove(zip_path)
            print(f"Removed temporary zip file: {zip_path}")

    # Rename the extracted file to include the .tsv extension
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    if original_file_path.exists():
        try:
            os.rename(original_file_path, data_file_path)
            print(f"Renamed extracted file to {data_file_path}")
        except Exception as e:
            print(f"ERROR: Failed to rename {original_file_path} to {data_file_path}: {e}")
    else:
        print(f"Warning: Expected file {original_file_path} not found after extraction.")

def create_balanced_dataset(df):
    """Creates a balanced dataset by downsampling the majority class ("ham")."""
    num_spam = df[df["Label"] == "spam"].shape[0]
    # Randomly sample "ham" instances to match the number of "spam"
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
    # Combine subset with all spam messages
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])
    print("Created balanced dataset.")
    return balanced_df

def random_split(df, train_frac, validation_frac):
    """Splits a DataFrame into train, validation, and test sets randomly."""
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]
    print(f"Split data: Train={len(train_df)}, Val={len(validation_df)}, Test={len(test_df)}")
    return train_df, validation_df, test_df

def prepare_spam_dataset(url, zip_path, extracted_path, data_file_path, force_download=False):
    """Orchestrates downloading, balancing, splitting, and saving the dataset."""
    if force_download and data_file_path.exists():
        print(f"Force download enabled. Removing existing {data_file_path}...")
        os.remove(data_file_path)
        # Optionally remove other generated files if needed

    download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)

    if not data_file_path.exists():
        print(f"ERROR: Data file {data_file_path} not found after download attempt. Exiting.")
        return None, None, None # Indicate failure

    try:
        df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
    except Exception as e:
        print(f"ERROR: Failed to read {data_file_path}: {e}")
        return None, None, None

    print("Original dataset counts:")
    print(df["Label"].value_counts())

    balanced_df = create_balanced_dataset(df)
    print("Balanced dataset counts:")
    print(balanced_df["Label"].value_counts())

    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

    train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)
    
    # Ensure the data directory exists
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Save splits to CSV files in the 'data' directory
    train_df.to_csv(data_dir / "train.csv", index=None)
    validation_df.to_csv(data_dir / "validation.csv", index=None)
    test_df.to_csv(data_dir / "test.csv", index=None)
    
    print(f"Saved train ({len(train_df)}), validation ({len(validation_df)}), and test ({len(test_df)}) sets to 'data/' directory.")
    
    # Clean up downloaded/extracted files
    if os.path.exists(zip_path):
        try:
            os.remove(zip_path)
            print(f"Removed downloaded zip: {zip_path}")
        except OSError as e:
            print(f"Error removing zip file {zip_path}: {e}")
            
    if os.path.exists(extracted_path):
        try:
            shutil.rmtree(extracted_path)
            print(f"Removed extracted directory: {extracted_path}")
        except OSError as e:
             print(f"Error removing directory {extracted_path}: {e}")

    return train_df, validation_df, test_df # Return the dataframes as well


class SpamDataset(Dataset):
    """Dataset class for SMS Spam classification.

    Handles loading data from CSV, tokenization, truncation, and padding.
    """
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        """
        Args:
            csv_file (str): Path to the CSV file (train.csv, validation.csv, etc.).
            tokenizer: An initialized tiktoken tokenizer.
            max_length (int, optional): Maximum sequence length. If None, uses the longest
                                       sequence in the dataset. Defaults to None.
            pad_token_id (int, optional): Token ID used for padding. Defaults to 50256 (often <|endoftext|>).
        """
        try:
            self.data = pd.read_csv(csv_file)
        except FileNotFoundError:
            print(f"ERROR: CSV file not found at {csv_file}")
            raise # Re-raise the exception
        except Exception as e:
            print(f"ERROR: Failed to read CSV {csv_file}: {e}")
            raise

        # Pre-tokenize texts for efficiency
        print(f"Tokenizing {csv_file}...")
        self.encoded_texts = [tokenizer.encode(text) for text in self.data['Text']]

        # Determine max_length
        if max_length is None:
            self.max_length = self._longest_encoded_length()
            print(f"Determined max_length for {csv_file}: {self.max_length}")
        else:
            # Truncate sequences longer than the provided max_length
            original_lengths = [len(t) for t in self.encoded_texts]
            self.encoded_texts = [ text[:max_length] for text in self.encoded_texts ]
            truncated_count = sum(1 for ol, tl in zip(original_lengths, self.encoded_texts) if ol > len(tl))
            if truncated_count > 0:
                print(f"Truncated {truncated_count} sequences in {csv_file} to max_length {max_length}.")
            self.max_length = max_length # Use the provided max_length

        # Pad sequences to max_length
        self.encoded_texts = [
            text + [pad_token_id] * (self.max_length - len(text))
            for text in self.encoded_texts
        ]
        print(f"Padding sequences in {csv_file} to length {self.max_length}.")

    def __getitem__(self, index):
        """Returns the tokenized input and label for a given index."""
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]['Label']
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.data)

    def _longest_encoded_length(self):
        """Helper function to find the length of the longest tokenized sequence."""
        return max(len(encoded_text) for encoded_text in self.encoded_texts)


# --- Evaluation Metric --- #

def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    """Calculates the classification accuracy for a given DataLoader.

    Assumes the model outputs logits for the *last* token, suitable for sequence classification.
    """
    model.eval() # Set model to evaluation mode
    correct_predictions, num_examples = 0, 0

    if not data_loader or len(data_loader) == 0:
        print("Warning: Invalid or empty DataLoader in calc_accuracy_loader.")
        return 0.0 # Return 0 accuracy if no data

    # Determine number of batches to process
    actual_num_batches = len(data_loader)
    if num_batches is None:
        num_batches_to_eval = actual_num_batches
    else:
        num_batches_to_eval = min(num_batches, actual_num_batches)

    if num_batches_to_eval == 0:
        print("Warning: Zero batches to evaluate in calc_accuracy_loader.")
        return 0.0

    with torch.no_grad(): # Disable gradient calculations
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i >= num_batches_to_eval:
                break
            # Move data to the target device
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            # Get model prediction (logits for the last token)
            logits = model(input_batch)[:, -1, :]
            # Get the predicted label index (0 or 1)
            predicted_labels = torch.argmax(logits, dim=-1)

            # Count correct predictions
            num_examples += target_batch.shape[0] # Use target_batch shape for robustness
            correct_predictions += (predicted_labels == target_batch).sum().item()

    model.train() # Set model back to training mode
    if num_examples == 0:
        print("Warning: No examples processed in calc_accuracy_loader.")
        return 0.0 # Avoid division by zero
    return correct_predictions / num_examples

# --- Loss Functions (Classifier specific) --- #

def calc_loss_batch_classifier(input_batch, target_batch, model, device):
    """Calculates the cross-entropy loss for classification (uses last token logit)."""
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    # Get logits for the last token only
    logits = model(input_batch)[:, -1, :]
    # Calculate cross-entropy loss between last token logits and target labels
    loss = F.cross_entropy(logits, target_batch)
    return loss

def calc_loss_loader_classifier(data_loader, model, device, num_batches=None):
    """Calculates the average classification loss over batches from a DataLoader."""
    total_loss = 0.
    if not data_loader or len(data_loader) == 0:
        print("Warning: Invalid or empty DataLoader in calc_loss_loader_classifier.")
        return float('nan')

    actual_num_batches = len(data_loader)
    num_batches_to_eval = min(num_batches, actual_num_batches) if num_batches is not None else actual_num_batches

    if num_batches_to_eval == 0:
        print("Warning: Zero batches to evaluate in calc_loss_loader_classifier.")
        return float('nan')

    evaluated_batches = 0
    model.eval()
    with torch.no_grad():
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i >= num_batches_to_eval:
                break
            try:
                # Use the classification-specific batch loss function
                loss = calc_loss_batch_classifier(input_batch, target_batch, model, device)
                total_loss += loss.item()
                evaluated_batches += 1
            except Exception as e:
                print(f"Error calculating loss for batch {i} in calc_loss_loader_classifier: {e}")
    model.train()

    if evaluated_batches == 0:
        print("Warning: No batches successfully processed in calc_loss_loader_classifier.")
        return float('nan')

    return total_loss / evaluated_batches

def evaluate_model_classifier(model, train_loader, val_loader, device, eval_iter):
    """Evaluates the classifier model loss on training and validation sets."""
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader_classifier(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader_classifier(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

# --- Training Loop (Classifier specific) --- #

def train_classifier_simple(model, train_loader, val_loader, optimizer, device,
                            num_epochs, eval_freq, eval_iter):
    """Fine-tunes the model for classification."""
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    print("---- Starting Classifier Fine-tuning ----")
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        num_batches_epoch = len(train_loader)

        if num_batches_epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs}: Skipping - train_loader empty.")
            continue

        for i, (input_batch, target_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            try:
                # Use the classification-specific loss function
                loss = calc_loss_batch_classifier(input_batch, target_batch, model, device)
                loss.backward()
                optimizer.step()
                examples_seen += input_batch.shape[0] # Track examples seen
                global_step += 1
                epoch_train_loss += loss.item()
            except Exception as e:
                print(f"ERROR during classifier training step {global_step} (Batch {i}): {e}")
                continue # Skip batch on error

            # Periodic evaluation (loss only)
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model_classifier(
                    model, train_loader, val_loader, device, eval_iter
                )
                if not (torch.isnan(torch.tensor(train_loss)) or torch.isnan(torch.tensor(val_loss))):
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    print(f"Epoch {epoch+1:02d}/{num_epochs:02d} | Step {global_step:06d} | "
                          f"Batch {i+1:04d}/{num_batches_epoch:04d} | "
                          f"Avg Train Loss: {train_loss:.3f} | Avg Val Loss: {val_loss:.3f}")
                else:
                    print(f"Epoch {epoch+1:02d}/{num_epochs:02d} | Step {global_step:06d} | "
                          f"Batch {i+1:04d}/{num_batches_epoch:04d} | "
                          f"Classifier evaluation failed (loss is NaN).")

        # --- End of Epoch Evaluation (Accuracy) ---
        avg_epoch_loss = epoch_train_loss / num_batches_epoch if num_batches_epoch > 0 else float('nan')
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=None) # Use all batches for epoch acc
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=None)
        print(f"--- Epoch {epoch+1} Completed | Avg Epoch Train Loss: {avg_epoch_loss:.3f} --- ")
        print(f"    Epoch Train Accuracy: {train_accuracy*100:.2f}% | Epoch Val Accuracy: {val_accuracy*100:.2f}% --- ")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)
        print("-"*50)

    print("---- Classifier Fine-tuning Finished ----")
    return train_losses, val_losses, train_accs, val_accs, examples_seen

# --- Plotting Utility --- #

def plot_values(epochs_seen, examples_seen, train_values, val_values, label="loss", output_path=None):
    """Plots training and validation loss or accuracy against epochs and examples seen.

    Args:
        epochs_seen: List or array of epoch numbers (or steps) for the x-axis.
        examples_seen: List or array of examples seen counts for the secondary x-axis.
        train_values: List or array of training metric values.
        val_values: List or array of validation metric values.
        label (str): The label for the y-axis (e.g., 'loss', 'accuracy').
        output_path (str, optional): Path to save the plot image file. 
                                      If None, generates a default name in the current directory.
                                      Defaults to None.
    """
    fig, ax1 = plt.subplots(figsize=(7, 4)) # Slightly larger figure

    # Plot values against epochs (primary x-axis)
    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, linestyle="--", label=f"Validation {label}") # Dashed validation line
    ax1.set_xlabel("Steps" if label == "loss" else "Epochs") # Use Steps for loss plot, Epochs for accuracy
    ax1.set_ylabel(label.capitalize())
    ax1.tick_params(axis='y')
    ax1.legend(loc='upper left')
    ax1.grid(True) # Add grid

    # Create a second x-axis for examples seen
    ax2 = ax1.twiny() # Share the same y-axis
    ax2.set_xlabel("Examples Seen")
    # Align the second x-axis - ensure examples_seen has the same length as epochs_seen
    if len(epochs_seen) == len(examples_seen):
        ax2.set_xticks(epochs_seen) # Place ticks at the same positions as epoch ticks
        ax2.set_xticklabels([f'{int(ex/1000)}k' for ex in examples_seen]) # Format labels (e.g., 10k)
    else:
         # Fallback if lengths mismatch - plot invisible data just to set limits
         ax2.plot(examples_seen, train_values, alpha=0) 

    ax2.tick_params(axis='x')

    fig.tight_layout() # Adjust layout
    
    # Determine filename based on output_path or default
    if output_path:
        plot_filename = output_path
        # Ensure the output directory exists
        output_dir = Path(plot_filename).parent
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        plot_filename = f"finetune-{label}-plot.png" # Default filename
        
    try:
        plt.savefig(plot_filename)
        print(f"Plot saved to {plot_filename}")
    except Exception as e:
        print(f"Error saving plot {plot_filename}: {e}")
    # plt.show()