"""
PyTorch Dataset and DataLoader example for video data stored in omniio archives.

This example demonstrates how to:
1. Load video data from omniio blob archives using self-contained parquet metadata
2. Create a PyTorch Dataset with on-the-fly video loading
3. Handle both video and audio streams
4. Use with PyTorch DataLoader for efficient batch loading
"""

import torch
from torch.utils.data import Dataset, DataLoader
import polars as pl
from pathlib import Path
from typing import Optional, Dict, Any

from omniio.interface import video_read


class VideoArchiveDataset(Dataset):
    """
    PyTorch Dataset for loading video from omniio blob archives.

    Args:
        metadata_path: Path to the metadata.parquet file (or archive directory)
        load_audio: Whether to load audio track (default: True)
        start_frame: Optional start frame for video slicing
        end_frame: Optional end frame for video slicing
        start_time: Optional start time for video slicing (seconds)
        end_time: Optional end time for video slicing (seconds)
    """

    def __init__(
        self,
        metadata_path: str,
        load_audio: bool = True,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ):
        self.load_audio = load_audio
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.start_time = start_time
        self.end_time = end_time

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

        # Load video using generic interface (handles local/remote automatically)
        video_data = video_read(
            bin_path,
            row['start_byte'],
            row['end_byte'] - row['start_byte'],
            self.start_frame,
            self.end_frame,
            self.start_time,
            self.end_time
        )

        # Convert video to torch tensor (frames, height, width, channels)
        video_tensor = torch.from_numpy(video_data.video_array)

        # Prepare sample
        sample = {
            'video': video_tensor,
            'fps': video_data.fps,
            'height': video_data.height,
            'width': video_data.width,
            'id': row['id'],
        }

        # Add audio if requested
        if self.load_audio and video_data.audio_array is not None:
            audio_tensor = torch.from_numpy(video_data.audio_array)
            sample['audio'] = audio_tensor
            sample['sample_rate'] = video_data.sample_rate

        # Add label if available
        if 'label' in row:
            sample['label'] = row['label']

        return sample


def collate_video_batch(batch):
    """
    Custom collate function for batching video samples with different lengths.
    Pads videos to the longest in the batch.
    """
    # Find max frames in batch
    max_frames = max(sample['video'].shape[0] for sample in batch)

    # Pad video samples
    padded_videos = []
    audios = []
    fps_list = []
    ids = []
    labels = []

    for sample in batch:
        video = sample['video']
        padding = max_frames - video.shape[0]

        if padding > 0:
            # Pad with zeros (frames, H, W, C)
            pad_shape = list(video.shape)
            pad_shape[0] = padding
            pad_tensor = torch.zeros(pad_shape, dtype=video.dtype)
            padded = torch.cat([video, pad_tensor], dim=0)
        else:
            padded = video

        padded_videos.append(padded)
        fps_list.append(sample['fps'])
        ids.append(sample['id'])

        if 'audio' in sample:
            audios.append(sample['audio'])

        if 'label' in sample:
            labels.append(sample['label'])

    # Stack into batch
    batched = {
        'video': torch.stack(padded_videos),
        'fps': torch.tensor(fps_list),
        'id': ids,
    }

    if audios:
        # Pad audio to same length
        max_audio_len = max(a.shape[0] for a in audios)
        padded_audios = []
        for audio in audios:
            padding = max_audio_len - audio.shape[0]
            if padding > 0:
                padded_audio = torch.nn.functional.pad(audio, (0, 0, 0, padding))
            else:
                padded_audio = audio
            padded_audios.append(padded_audio)
        batched['audio'] = torch.stack(padded_audios)

    if labels:
        batched['label'] = torch.tensor(labels)

    return batched


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Video Dataset Example - Loading from omniio archives")
    print("=" * 70)

    # Example 1: Basic video dataset
    print("\n[Example 1] Loading video frames")
    print("-" * 70)

    dataset = VideoArchiveDataset(
        metadata_path="./my_video_archive",  # or direct path to metadata.parquet
    )

    print(f"Dataset size: {len(dataset)} samples")

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_video_batch,
        pin_memory=True
    )

    # Iterate through batches
    print("\nLoading batches:")
    for batch_idx, batch in enumerate(dataloader):
        video = batch['video']  # Shape: (batch, frames, height, width, channels)
        fps = batch['fps']

        print(f"  Batch {batch_idx + 1}:")
        print(f"    Video shape: {video.shape}")
        print(f"    FPS: {fps.tolist()}")

        if 'audio' in batch:
            print(f"    Audio shape: {batch['audio'].shape}")

        if batch_idx >= 2:  # Show 3 batches
            break

    # Example 2: Video without audio
    print("\n[Example 2] Loading video only (no audio)")
    print("-" * 70)

    video_only_dataset = VideoArchiveDataset(
        metadata_path="./my_video_archive",
        load_audio=False
    )

    video_only_loader = DataLoader(
        video_only_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_video_batch
    )

    for batch_idx, batch in enumerate(video_only_loader):
        print(f"  Batch {batch_idx + 1}:")
        print(f"    Video shape: {batch['video'].shape}")
        print(f"    Has audio: {'audio' in batch}")

        if batch_idx >= 1:  # Show 2 batches
            break

    # Example 3: Time-sliced video (first 2 seconds)
    print("\n[Example 3] Loading 2-second clips")
    print("-" * 70)

    time_dataset = VideoArchiveDataset(
        metadata_path="./my_video_archive",
        start_time=0.0,
        end_time=2.0,
    )

    time_loader = DataLoader(
        time_dataset,
        batch_size=4,
        collate_fn=collate_video_batch
    )

    for batch_idx, batch in enumerate(time_loader):
        video = batch['video']
        fps = batch['fps'][0].item()
        duration = video.shape[1] / fps  # frames / fps
        print(f"  Batch {batch_idx + 1}: {video.shape}, ~{duration:.2f}s per sample")

        if batch_idx >= 1:
            break

    # Example 4: Frame-based slicing (frames 10-50)
    print("\n[Example 4] Loading specific frame ranges")
    print("-" * 70)

    frame_dataset = VideoArchiveDataset(
        metadata_path="./my_video_archive",
        start_frame=10,
        end_frame=50,
        load_audio=False
    )

    frame_loader = DataLoader(
        frame_dataset,
        batch_size=2,
        collate_fn=collate_video_batch
    )

    for batch_idx, batch in enumerate(frame_loader):
        print(f"  Batch {batch_idx + 1}: {batch['video'].shape}")

        if batch_idx >= 1:
            break

    # Example 5: Inspect metadata
    print("\n[Example 5] Inspecting metadata")
    print("-" * 70)

    print("Metadata columns:", dataset.metadata.columns)
    print("\nFirst 3 samples:")
    print(dataset.metadata.head(3))

    print("\n" + "=" * 70)
    print("Video dataset examples completed!")
    print("=" * 70)
