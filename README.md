# Omni-IO

Efficient Python library for reading and writing multimedia data (audio, video, text) from binary archive blobs with support for both local and remote HTTP range requests.

## Features

- **Multi-format support**: Audio (FLAC, WAV, WebM/Opus), Video (MP4), Text (zstandard compressed)
- **Local and remote access**: Seamlessly read from local files or remote URLs using HTTP range requests
- **Efficient storage**: Binary blob archives with PyArrow/Parquet metadata indexing
- **Time-based slicing**: Extract specific time ranges from audio/video without loading entire files
- **Parallel processing**: Multi-process append operations for fast archive creation
- **Streaming operations**: Memory-efficient handling of large multimedia files

## Why Omni-IO?

Most multimedia datasets outgrow naive storage approaches quickly. Omni-IO is designed for the scale and access patterns that matter in practice.

**Raw files on disk** work fine for small datasets, but millions of small audio or text files create serious filesystem overhead — inode exhaustion, slow directory scans, and poor I/O throughput when files are scattered across disk. Omni-IO packs everything into a small number of large `.bin` files, eliminating per-file metadata overhead and enabling sequential I/O patterns that storage systems are optimized for. This also makes mass storage scans and transfer extremely fast, as it minimizes the number of accessed files.

**WebDataset** solves the small-files problem with sequential tar shards, but trades away random access. Filtering a WebDataset by label, duration, or speaker requires scanning the entire dataset. Resuming mid-shard is awkward, and any preprocessing that requires re-ordering data means re-sharding. Omni-IO stores byte offsets in a Parquet file, so any item can be fetched in O(1) with a single range read — filter by any metadata column, shuffle freely, and access only what you need.

**HuggingFace Datasets** caches data locally and works well for text, but storing raw audio or video means either re-encoding everything into Arrow's columnar format (lossy or bloated) or falling back to large file caches that are opaque and hard to manage. Omni-IO keeps data in its native compressed format (FLAC, WebM, zstd) inside the blob, so storage is compact and the encoding pipeline is explicit.

**Storing audio/video directly in Parquet** is tempting since Parquet already handles metadata well, but Arrow's binary columns are not designed for large variable-length blobs. Each audio or video sample gets embedded as a `LargeBinary` value, which defeats columnar compression, inflates row-group sizes, and causes the Parquet reader to load entire row groups into memory even when you only need one sample. Parquet also lacks any concept of seeking within a stored value, so time-based slicing requires decoding the whole blob after retrieval. Omni-IO separates concerns cleanly: Parquet holds only lightweight columnar metadata (byte offsets, sample rates, durations), while the binary data lives in flat `.bin` files that support direct seek-and-read.

**Numpy array dumps** (`.npy`/`.npz`) store decoded, uncompressed PCM data, which means a 10-hour audio dataset that fits in ~3 GB as FLAC balloons to 50+ GB as float32 arrays. They also fix a single sample rate and channel count at write time, making mixed-format datasets impossible. There is no metadata layer, so filtering by duration or speaker ID requires loading and inspecting every file. Omni-IO stores audio in its native compressed format and decodes on demand, keeping storage compact while retaining all format metadata in queryable Parquet columns.

**Kaldiio** is a widely used solution in speech processing, but it stores features exclusively as float32 arrays — there is no support for compressed audio formats, video, or text. This locks users into pre-extracting features before archiving, which forecloses any future re-extraction with different parameters. Archives are also local-only; there is no mechanism for remote access.

**Lhotse** is a mature speech data toolkit with excellent manifests and cutting operations, but its storage backend ultimately relies on either raw files on disk or external formats like Kaldi archives. Lhotse manages *where* files are, but doesn't consolidate *how* they are stored — you still end up with millions of individual audio files and all the filesystem overhead that entails. Omni-IO handles both the metadata layer and the packed binary storage, and adds first-class remote access without requiring a separate serving layer.

**The key differentiator: the same Parquet metadata file works for both local and remote access.** Point Omni-IO at a local `.bin` file for training runs, or swap in an HTTPS URL for the same archive hosted on object storage — the API is identical. This means you can build and validate an archive locally, upload the bin files to S3 or GCS, and read from them remotely without any code changes. HTTP range requests fetch only the bytes needed for each sample, so remote reads are as efficient as local ones.

## Installation

```bash
pip install omniio
```

### Development Installation

```bash
git clone https://github.com/wavlab-speech/omniio.git
cd omniio
pip install -e .
```

## Quick Start

### Reading from Archives

#### Audio

```python
from omniio.interface import audio_read

# Read audio from local or remote archive
result = audio_read(
    archive_path="/path/to/archive.bin",  # or "https://example.com/archive.bin"
    start_offset=1024,
    file_size=50000,
    start_time=5.0,  # optional: start at 5 seconds
    end_time=10.0    # optional: end at 10 seconds
)

print(f"Sample rate: {result.sample_rate}")
print(f"Audio shape: {result.array.shape}")  # (frames, channels)
```

#### Video

```python
from omniio.video.read import video_read_local

# Read video with frame-based slicing
result = video_read_local(
    archive_path="/path/to/archive.bin",
    start_offset=2048,
    file_size=1000000,
    start_frame=100,
    end_frame=200
)

print(f"FPS: {result.fps}")
print(f"Video shape: {result.video_array.shape}")  # (frames, height, width, 3)
print(f"Audio shape: {result.audio_array.shape}")  # (samples, channels)
```

#### Text

```python
from omniio.text.read import text_read_local

# Read compressed text
result = text_read_local(
    archive_path="/path/to/archive.bin",
    start_offset=512,
    file_size=2048
)

print(result.text)
```

### Writing to Archives

#### Creating an Archive

```python
from omniio.blob.blob import Blob

# Initialize archive
blob = Blob(
    archive_dir="./my_archive",
    modality="audio",
    max_bin_size=320 * 1024 * 1024  # 320MB per bin file
)

# Append audio files in parallel
blob.append(
    items=["audio1.wav", "audio2.flac", "audio3.mp3"],
    ids=["sample_001", "sample_002", "sample_003"],
    num_workers=4,
    target_format="flac",
    target_bit_depth=16
)

# View archive statistics
blob.summary()
```

#### Audio Format Conversion

```python
from omniio.audio.write import audio_write

# Convert audio to different format
raw_bytes, metadata = audio_write(
    audio_path="input.wav",
    item_id="converted_audio",
    target_format="flac",  # 'flac', 'wav', 'webm'
    target_bit_depth=24
)

print(f"Channels: {metadata['channels']}")
print(f"Sample rate: {metadata['sample_rate']}")
print(f"Compressed size: {len(raw_bytes)} bytes")
```

#### Text Compression

```python
from omniio.text.write import text_write

# Compress text data
raw_bytes, metadata = text_write(
    path_or_string="document.txt",
    item_id="doc_001",
    is_path=True,
    compression_level=3
)

print(f"Original size: {metadata['original_size']} bytes")
print(f"Compressed size: {metadata['compressed_size']} bytes")
```

## Archive Structure

Archives are organized as follows:

```
archive_dir/
├── blob_0.bin          # Binary data (first chunk)
├── blob_1.bin          # Binary data (second chunk, if > max_bin_size)
└── metadata.parquet    # PyArrow table with byte offsets and metadata
```

The metadata table contains:
- `id`: Unique identifier for each entry
- `start_byte`: Byte offset where entry begins
- `end_byte`: Byte offset where entry ends
- `bin_index`: Which bin file contains the entry
- Format-specific metadata (sample_rate, channels, dimensions, etc.)

## Data Formats

### Audio
- **Input formats**: FLAC, WAV, OGG, WebM/Opus
- **Output shape**: `(frames, channels)` as `float32` normalized to [-1.0, 1.0]
- **Supported bit depths**: 8, 16, 24, 32 (PCM formats only)

### Video
- **Input formats**: MP4 with H.264/H.265 video and AAC/Opus audio
- **Video output shape**: `(frames, height, width, 3)` as `uint8` RGB24
- **Audio output shape**: `(samples, channels)` as `float32`

### Text
- **Compression**: Zstandard (levels 1-22)
- **Encoding**: UTF-8

## Requirements

- Python >= 3.8
- numpy
- av (PyAV)
- soundfile
- requests
- zstandard
- pyarrow

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
