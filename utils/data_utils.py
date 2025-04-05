"""
Utility functions and classes for data loading and processing.

Includes a Dataset class for creating input/target sequences from text
and a function to create a DataLoader instance.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

class GPTDatasetV1(Dataset):
    """Creates input/target sequences from a text corpus.

    Uses a sliding window approach to generate chunks of text for training.
    Each input sequence is paired with a target sequence shifted by one token.
    """
    def __init__(self, txt, tokenizer, max_length, stride):
        """
        Args:
            txt (str): The raw text data.
            tokenizer: The tokenizer instance.
            max_length (int): The maximum length of input sequences.
            stride (int): The step size for the sliding window.
        """
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={'<|endoftext|>'}) # Use default allowed special tokens

        # Check if the text is long enough to create at least one sequence
        if len(token_ids) <= max_length:
             print(f"Warning: Text length ({len(token_ids)} tokens) is less than or equal to max_length ({max_length}). No sequences created.")
             return

        # Iterate through the tokenized text with a sliding window
        # Stop iterating when the remaining tokens are not enough for a full max_length sequence
        for i in range(0, len(token_ids) - max_length, stride):
            # Extract the input sequence chunk
            input_chunk = token_ids[i : i + max_length]
            # Extract the target sequence chunk (shifted by one position)
            target_chunk = token_ids[i + 1 : i + max_length + 1]

            # Convert chunks to tensors and store them
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        """Returns the total number of input/target sequence pairs created.

        Returns:
            int: The number of sequences in the dataset.
        """
        return len(self.input_ids)

    def __getitem__(self, idx):
        """Retrieves the input and target sequence pair at the specified index.

        Args:
            idx (int): The index of the sequence to retrieve.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the input IDs
                                              and target IDs tensors.
        """
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128,
                         shuffle=True, drop_last=True, num_workers=0, tokenizer=None):
    """Creates a PyTorch DataLoader for the GPTDatasetV1.

    Handles dataset initialization and DataLoader configuration.

    Args:
        txt (str): The text data.
        batch_size (int): Number of sequences per batch.
        max_length (int): Maximum sequence length for the dataset.
        stride (int): Stride for creating sequences in the dataset.
        shuffle (bool): Whether to shuffle the data each epoch.
        drop_last (bool): Whether to drop the last incomplete batch.
        num_workers (int): Number of subprocesses to use for data loading.
        tokenizer (optional): An initialized tokenizer instance.
                            If None, initializes GPT-2 tokenizer by default.

    Returns:
        DataLoader: The configured PyTorch DataLoader instance.

    Raises:
        ValueError: If the created dataset is empty.
    """
    # Initialize the GPT-2 tokenizer if not provided externally
    if tokenizer is None:
        print("Warning: Tokenizer not provided to create_dataloader_v1. Initializing GPT-2 tokenizer by default.")
        tokenizer = tiktoken.get_encoding('gpt2')

    # Create the Dataset instance
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Check if the dataset was successfully populated
    if len(dataset) == 0:
        # Raise an error if no sequences were created (e.g., text too short)
        raise ValueError("Dataset created is empty. Check input text length, max_length, and stride parameters.")

    # Create and configure the DataLoader
    dataloader = DataLoader(
        dataset,                # The dataset to load from
        batch_size=batch_size,  # Number of sequences per batch
        shuffle=shuffle,        # Whether to shuffle sequences each epoch
        drop_last=drop_last,    # Drop the last batch if it's smaller than batch_size
        num_workers=num_workers # Number of subprocesses for data loading (0 means use main process)
    )

    return dataloader 