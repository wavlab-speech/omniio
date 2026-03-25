"""
PyTorch Dataset and DataLoader example for audio data stored in omniio archives.

This example demonstrates how to:
1. Load audio data from omniio blob archives using self-contained parquet metadata
2. Create a PyTorch Dataset with on-the-fly audio loading
3. Use with PyTorch DataLoader for efficient batch loading
"""

import torch
from torch.utils.data import Dataset, DataLoader
import polars as pl
from pathlib import Path
from typing import Optional, Dict, Any

from omniio.interface import audio_read


class AudioArchiveDataset(Dataset):
    """
    PyTorch Dataset for loading audio from omniio blob archives.

    Args:
        metadata_path: Path to the metadata.parquet file (or archive directory)
        sample_rate: If specified, only include samples with this sample rate
        start_time: Optional start time for audio slicing (seconds)
        end_time: Optional end time for audio slicing (seconds)
    """

    def __init__(
        self,
        metadata_path: str,
        sample_rate: Optional[int] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ):
        self.start_time = start_time
        self.end_time = end_time

        # Load metadata - handle both parquet file and directory paths
        metadata_path = Path(metadata_path)
        if metadata_path.is_dir():
            metadata_path = metadata_path / "metadata.parquet"

        # Use polars for fast parquet reading
        self.metadata = pl.read_parquet(metadata_path)

        # Filter by sample rate if specified
        if sample_rate is not None:
            self.metadata = self.metadata.filter(pl.col('sample_rate') == sample_rate)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Get metadata for this sample
        row = self.metadata.row(idx, named=True)

        # Get path directly from metadata (self-contained parquet)
        bin_path = row['path']

        # Load audio using generic interface (handles local/remote automatically)
        audio_data = audio_read(
            bin_path,
            row['start_byte'],
            row['end_byte'] - row['start_byte'],
            self.start_time,
            self.end_time
        )

        # Convert to torch tensor (frames, channels)
        audio_tensor = torch.from_numpy(audio_data.array)

        # Return sample with metadata
        sample = {
            'audio': audio_tensor,
            'sample_rate': audio_data.sample_rate,
            'id': row['id'],
            'format': row.get('format', audio_data.file_type),
        }

        # Add label if available
        if 'label' in row:
            sample['label'] = row['label']

        return sample


def collate_audio_batch(batch):
    """
    Custom collate function for batching audio samples with different lengths.
    Pads audio to the longest sample in the batch.
    """
    # Find max length in batch
    max_length = max(sample['audio'].shape[0] for sample in batch)

    # Pad audio samples
    padded_audio = []
    sample_rates = []
    ids = []
    labels = []

    for sample in batch:
        audio = sample['audio']
        padding = max_length - audio.shape[0]

        if padding > 0:
            # Pad with zeros (frames, channels)
            padded = torch.nn.functional.pad(audio, (0, 0, 0, padding))
        else:
            padded = audio

        padded_audio.append(padded)
        sample_rates.append(sample['sample_rate'])
        ids.append(sample['id'])

        if 'label' in sample:
            labels.append(sample['label'])

    # Stack into batch (batch, frames, channels)
    batched = {
        'audio': torch.stack(padded_audio),
        'sample_rate': torch.tensor(sample_rates),
        'id': ids,
    }

    if labels:
        batched['label'] = torch.tensor(labels)

    return batched


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Audio Dataset Example - Loading from omniio archives")
    print("=" * 70)

    # Example 1: Basic dataset with waveforms
    print("\n[Example 1] Loading audio waveforms")
    print("-" * 70)

    dataset = AudioArchiveDataset(
        metadata_path="./my_audio_archive",  # or direct path to metadata.parquet
    )

    print(f"Dataset size: {len(dataset)} samples")

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_audio_batch,
        pin_memory=True
    )

    # Iterate through a few batches
    print("\nLoading batches:")
    for batch_idx, batch in enumerate(dataloader):
        audio = batch['audio']  # Shape: (batch, frames, channels)
        sample_rates = batch['sample_rate']
        ids = batch['id']

        print(f"  Batch {batch_idx + 1}:")
        print(f"    Audio shape: {audio.shape}")
        print(f"    Sample rates: {sample_rates.tolist()}")
        print(f"    IDs: {ids[:3]}...")  # Print first 3 IDs

        if batch_idx >= 2:  # Show 3 batches
            break

    # Example 2: Filter by sample rate
    print("\n[Example 2] Filter by sample rate (16kHz only)")
    print("-" * 70)

    dataset_16k = AudioArchiveDataset(
        metadata_path="./my_audio_archive",
        sample_rate=16000,
    )

    print(f"Filtered dataset size: {len(dataset_16k)} samples")

    loader_16k = DataLoader(dataset_16k, batch_size=8, shuffle=False)

    for batch_idx, batch in enumerate(loader_16k):
        print(f"  Batch {batch_idx + 1}: {batch['audio'].shape}")
        if batch_idx >= 1:  # Show 2 batches
            break

    # Example 3: Time-sliced audio (3-second clips)
    print("\n[Example 3] Loading 3-second clips")
    print("-" * 70)

    clip_dataset = AudioArchiveDataset(
        metadata_path="./my_audio_archive",
        start_time=0.0,
        end_time=3.0,  # Load only first 3 seconds
    )

    clip_loader = DataLoader(
        clip_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_audio_batch
    )

    for batch_idx, batch in enumerate(clip_loader):
        audio = batch['audio']
        duration = audio.shape[1] / batch['sample_rate'][0].item()
        print(f"  Batch {batch_idx + 1}: {audio.shape}, ~{duration:.2f}s per sample")
        if batch_idx >= 1:
            break

    # Example 4: Inspect metadata
    print("\n[Example 4] Inspecting metadata")
    print("-" * 70)

    print("Metadata columns:", dataset.metadata.columns)
    print("\nFirst 3 samples:")
    print(dataset.metadata.head(3))

    print("\n" + "=" * 70)
    print("Audio dataset examples completed!")
    print("=" * 70)
