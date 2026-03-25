# OmniIO PyTorch Examples

This directory contains PyTorch Dataset and DataLoader implementations for using omniio archives in machine learning workflows.

## Examples

### 1. Audio Dataset (`audio_dataset.py`)

PyTorch Dataset for loading audio from omniio blob archives.

**Features:**
- On-the-fly audio loading from archives
- Support for time-based slicing
- Sample rate filtering
- Audio transforms (mono conversion, normalization, spectrograms)
- Custom collate function for variable-length audio
- Integration examples with audio classification models

**Usage:**
```python
from audio_dataset import AudioArchiveDataset, ToMono, Normalize, collate_audio_batch
from torch.utils.data import DataLoader
from torchvision import transforms

# Can pass either the parquet file directly or the archive directory
dataset = AudioArchiveDataset(
    metadata_path="./my_audio_archive/metadata.parquet",  # or just "./my_audio_archive"
    sample_rate=16000,
    transform=transforms.Compose([
        ToMono(),
        Normalize(),
    ])
)

dataloader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_audio_batch
)

for batch in dataloader:
    audio = batch['audio']  # (batch, frames, channels)
    sample_rates = batch['sample_rate']
    # Training code here
```

**Transforms:**
- `ToMono` - Convert stereo to mono
- `Normalize` - Normalize audio amplitude
- `ToSpectrogram` - Convert to mel spectrogram (requires torchaudio)

### 2. Video Dataset (`video_dataset.py`)

PyTorch Dataset for loading video from omniio blob archives.

**Features:**
- On-the-fly video loading with both video and audio streams
- Frame-based and time-based slicing
- Video transforms (resize, normalize, subsample)
- Custom collate function for variable-length videos
- Integration examples with 3D CNNs

**Usage:**
```python
from video_dataset import VideoArchiveDataset, ResizeVideo, NormalizeVideo, ToChannelsFirst
from torch.utils.data import DataLoader
from torchvision import transforms

# Can pass either the parquet file directly or the archive directory
dataset = VideoArchiveDataset(
    metadata_path="./my_video_archive/metadata.parquet",  # or just "./my_video_archive"
    transform=transforms.Compose([
        ResizeVideo(height=224, width=224),
        NormalizeVideo(),
        ToChannelsFirst(),
    ]),
    load_audio=True
)

dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=2,
    collate_fn=collate_video_batch
)

for batch in dataloader:
    video = batch['video']  # (batch, frames, channels, H, W)
    audio = batch['audio']  # (batch, samples, channels)
    # Training code here
```

**Transforms:**
- `ResizeVideo` - Resize video frames
- `NormalizeVideo` - Normalize to [0, 1]
- `ToChannelsFirst` - Convert to (F, C, H, W)
- `TemporalSubsample` - Sample fixed number of frames
- `RandomCrop` - Random spatial crop

### 3. Text Dataset (`text_dataset.py`)

PyTorch Dataset for loading text from omniio blob archives.

**Features:**
- On-the-fly text loading and decompression
- Integration with HuggingFace transformers tokenizers
- Simple word/character-level tokenizer
- Text preprocessing transforms
- BERT fine-tuning example

**Usage:**
```python
from text_dataset import TextArchiveDataset, Lowercase, collate_text_batch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Can pass either the parquet file directly or the archive directory
dataset = TextArchiveDataset(
    metadata_path="./my_text_archive/metadata.parquet",  # or just "./my_text_archive"
    tokenizer=tokenizer,
    max_length=128,
    transform=Lowercase()
)

dataloader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_text_batch
)

for batch in dataloader:
    input_ids = batch['input_ids']  # (batch, seq_len)
    attention_mask = batch['attention_mask']
    # Training code here
```

**Transforms:**
- `Lowercase` - Convert to lowercase
- `RemoveExtraWhitespace` - Clean whitespace
- `TruncateText` - Limit character length

## Requirements

### Core
```bash
pip install torch torchvision
pip install omniio
```

### Audio Features
```bash
pip install torchaudio  # For spectrograms
```

### Text/NLP Features
```bash
pip install transformers  # For BERT, GPT, etc.
```

## Running Examples

Each example file can be run standalone:

```bash
python examples/audio_dataset.py
python examples/video_dataset.py
python examples/text_dataset.py
```

## Archive Structure

All examples expect archives with this structure:
```
archive_dir/
├── blob_0.bin          # Binary data
├── blob_1.bin          # More binary data (if archive size exceeds max_bin_size)
└── metadata.parquet    # Self-contained index with columns:
                        #   - id: unique identifier
                        #   - path: absolute path to bin file (SELF-CONTAINED!)
                        #   - start_byte: offset in bin file
                        #   - end_byte: end offset
                        #   - bin_index: which bin file (legacy)
                        #   - format, sample_rate, etc. (modality-specific)
                        #   - label (optional): for supervised learning
```

**Key Feature**: The `metadata.parquet` file is **self-contained** - it includes the full `path` to each bin file, so you can:
1. Copy just the parquet file to a different machine
2. Copy the referenced bin files
3. Use the dataset without knowing the original archive directory structure

This makes it easy to share datasets or move them between storage systems.

## Creating Archives

Use the omniio Blob class to create archives:

```python
from omniio.blob.blob import Blob

# Audio archive
audio_blob = Blob(archive_dir="./audio_archive", modality="audio")
audio_blob.append(
    items=["audio1.wav", "audio2.flac"],
    ids=["audio_001", "audio_002"],
    num_workers=4,
    target_format="flac"
)

# Video archive
video_blob = Blob(archive_dir="./video_archive", modality="video")
video_blob.append(
    items=["video1.mp4", "video2.mp4"],
    ids=["video_001", "video_002"],
    num_workers=2
)

# Text archive
text_blob = Blob(archive_dir="./text_archive", modality="text")
text_blob.append(
    items=["doc1.txt", "doc2.txt"],
    ids=["doc_001", "doc_002"],
    is_path=True,
    compression_level=3
)
```

## Performance Tips

1. **Use multiple workers**: Set `num_workers > 0` in DataLoader for parallel data loading
2. **Pin memory**: Use `pin_memory=True` for faster GPU transfer
3. **Prefetch**: Increase `prefetch_factor` in DataLoader for better pipelining
4. **Batch size**: Larger batches reduce I/O overhead
5. **Archive location**: Store archives on fast storage (SSD/NVMe)
6. **Remote archives**: For HTTP archives, enable caching and use CDN

## License

MIT License
