"""
PyTorch Dataset and DataLoader example for text data stored in omniio archives.

This example demonstrates how to:
1. Load text data from omniio blob archives using self-contained parquet metadata
2. Create a PyTorch Dataset with on-the-fly text loading
3. Use with PyTorch DataLoader for efficient batch loading
"""

import torch
from torch.utils.data import Dataset, DataLoader
import polars as pl
from pathlib import Path
from typing import Optional, Dict, Any

from omniio.interface import text_read


class TextArchiveDataset(Dataset):
    """
    PyTorch Dataset for loading text from omniio blob archives.

    Args:
        metadata_path: Path to the metadata.parquet file (or archive directory)
    """

    def __init__(self, metadata_path: str):
        # Load metadata - handle both parquet file and directory paths
        metadata_path = Path(metadata_path)
        if metadata_path.is_dir():
            metadata_path = metadata_path / "metadata.parquet"

        # Use polars for fast parquet reading
        self.metadata = pl.read_parquet(metadata_path)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Get metadata for this sample
        row = self.metadata.row(idx, named=True)

        # Get path directly from metadata (self-contained parquet)
        bin_path = row['path']

        # Load text using generic interface (handles local/remote automatically)
        text_data = text_read(
            bin_path,
            row['start_byte'],
            row['end_byte'] - row['start_byte']
        )

        # Return sample with metadata
        sample = {
            'text': text_data.text,
            'id': row['id'],
            'format': row.get('format', 'txt'),
        }

        # Add label if available
        if 'label' in row:
            sample['label'] = row['label']

        return sample


def collate_text_batch(batch):
    """
    Custom collate function for batching text samples.
    """
    batched = {
        'text': [sample['text'] for sample in batch],
        'id': [sample['id'] for sample in batch],
        'format': [sample['format'] for sample in batch],
    }

    if 'label' in batch[0]:
        batched['label'] = torch.tensor([sample['label'] for sample in batch])

    return batched


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Text Dataset Example - Loading from omniio archives")
    print("=" * 70)

    # Example 1: Basic text dataset
    print("\n[Example 1] Loading text data")
    print("-" * 70)

    dataset = TextArchiveDataset(
        metadata_path="./my_text_archive",  # or direct path to metadata.parquet
    )

    print(f"Dataset size: {len(dataset)} samples")

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_text_batch,
        pin_memory=True
    )

    # Iterate through batches
    print("\nLoading batches:")
    for batch_idx, batch in enumerate(dataloader):
        texts = batch['text']
        ids = batch['id']

        print(f"  Batch {batch_idx + 1}:")
        print(f"    Batch size: {len(texts)}")
        print(f"    First text (100 chars): {texts[0][:100]}...")
        print(f"    IDs: {ids[:3]}...")  # Print first 3 IDs

        if batch_idx >= 2:  # Show 3 batches
            break

    # Example 2: Larger batches
    print("\n[Example 2] Larger batch sizes")
    print("-" * 70)

    large_batch_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_text_batch
    )

    for batch_idx, batch in enumerate(large_batch_loader):
        print(f"  Batch {batch_idx + 1}:")
        print(f"    Batch size: {len(batch['text'])}")
        print(f"    Text lengths: min={min(len(t) for t in batch['text'])}, "
              f"max={max(len(t) for t in batch['text'])}")

        if batch_idx >= 1:  # Show 2 batches
            break

    # Example 3: With labels
    print("\n[Example 3] Dataset with labels")
    print("-" * 70)

    for batch_idx, batch in enumerate(dataloader):
        print(f"  Batch {batch_idx + 1}:")
        print(f"    Batch size: {len(batch['text'])}")

        if 'label' in batch:
            print(f"    Labels: {batch['label'].tolist()}")
        else:
            print(f"    No labels in dataset")

        if batch_idx >= 1:
            break

    # Example 4: Inspect metadata
    print("\n[Example 4] Inspecting metadata")
    print("-" * 70)

    print("Metadata columns:", dataset.metadata.columns)
    print("\nFirst 3 samples:")
    print(dataset.metadata.head(3))

    # Example 5: Access individual samples
    print("\n[Example 5] Individual sample access")
    print("-" * 70)

    sample = dataset[0]
    print(f"Sample 0:")
    print(f"  ID: {sample['id']}")
    print(f"  Format: {sample['format']}")
    print(f"  Text length: {len(sample['text'])} characters")
    print(f"  Text preview: {sample['text'][:150]}...")

    print("\n" + "=" * 70)
    print("Text dataset examples completed!")
    print("=" * 70)
