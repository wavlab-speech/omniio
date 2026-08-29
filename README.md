# Omni-IO

Efficient Python library for reading and writing multimedia data (audio, video, text) from binary archive blobs with support for both local and remote HTTP range requests.

## Features

- **Multi-format support**: Audio (FLAC, WAV, WebM/Opus), Video (MP4), Text (zstandard compressed)
- **Local and remote access**: Seamlessly read from local files or remote URLs using HTTP range requests
- **Efficient storage**: Binary blob archives with PyArrow/Parquet metadata indexing
- **Frame-level slicing**: Extract specific time ranges from audio/video without loading entire files
- **Parallel processing**: Multi-process append operations for fast archive creation
- **Streaming operations**: Memory-efficient handling of large multimedia files

- **Kaldi ark/scp compatibility** via `omniio.kaldi`, an MIT-licensed drop-in for `kaldiio`

## Why Omni-IO?

Most multimedia datasets outgrow naive storage approaches quickly. Omni-IO is designed for the scale and access patterns that matter in practice.

- Raw files on disk create serious filesystem overhead at scale — inode exhaustion, slow directory scans, and poor I/O throughput. Omni-IO packs everything into large .bin files, enabling fast sequential I/O and efficient bulk transfers.
- WebDataset eliminates the small-files problem but sacrifices random access. Omni-IO stores byte offsets in Parquet, so any item can be fetched in O(1) with a single range read — filter by any metadata column and shuffle freely.
- HuggingFace Datasets / Parquet blobs force audio and video into columnar formats they weren't designed for, inflating storage and defeating compression. Omni-IO keeps data in its native format (FLAC, WebM, zstd) and reserves Parquet for lightweight metadata only.
- HDF5 binary blobs do not expose the byte-range access needed for frame-level seeking, making it inefficient for partial reads and remote access.
- Numpy dumps store uncompressed PCM, ballooning storage 10–15×. Omni-IO decodes on demand from compressed formats, keeping archives compact while retaining full metadata.
- Lhotse manages *where* files are, but doesn't consolidate *how* they are stored — you still end up with individual files or WebDataset.
- Remote support: the same Parquet metadata file works for local and remote access. Swap a local .bin path for an HTTPS URL and the API is identical — HTTP range requests fetch only the bytes needed per sample.

## Installation

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

## Kaldi ark/scp Compatibility

`omniio.kaldi` reads and writes Kaldi `ark`/`scp` archives, so a project that
only needs Kaldi I/O can drop its `kaldiio` dependency:

```python
from omniio import kaldi as kaldiio   # same names, same signatures

with kaldiio.ReadHelper("scp:feats.scp") as reader:
    for utt_id, feats in reader:
        ...

with kaldiio.WriteHelper("ark,scp:feats.ark,feats.scp") as writer:
    writer["utt1"] = feats                  # float32/float64 matrix or vector

kaldiio.save_ark("wav.ark", {"utt1": (16000, wave)}, scp="wav.scp")
array = kaldiio.load_mat("feats.ark:1234")  # random access via an scp entry
```

Exported: `ReadHelper`, `WriteHelper`, `load_ark`, `load_scp`,
`load_scp_sequential`, `load_mat`, `save_ark`, `save_mat`, `open_like_kaldi`,
`parse_wspecifier`, `parse_rspecifier`, `LazyLoader`.

Supported on-disk formats:

| Form | Notes |
|---|---|
| `FM`/`DM`, `FV`/`DV` | float32/float64 matrices and vectors |
| `CM`/`CM2`/`CM3` | compressed matrices, all seven `compression_method` values |
| `std::vector<int32>` | alignments |
| `WaveHolder` (bare RIFF) | decoded at its native PCM width |
| `AUDIO`-framed blobs | the extended archive layout, any container `soundfile` can decode |
| text (`ark,t:`) | matrices and vectors |

Also supported: `segments` files, `scp` random access with a bounded file
descriptor cache (`max_cache_fd`), and Kaldi extended filenames including pipes
(`sox ... |`, `| gzip -c > x.gz`), `-` for stdin/stdout, and `.gz`.

Archives written here are byte-identical to Kaldi's own. Two deliberate
divergences from `kaldiio` are documented in `omniio/kaldi/compression.py`:
`kSpeechFeature` compression of matrices with fewer than five rows follows
Kaldi rather than `kaldiio`'s wrapping arithmetic, and the fixed-range methods
`kOneByteUnsignedInteger`/`kOneByteZeroOne` clip out-of-range input instead of
letting it wrap.

This code is written against the format description in Kaldi itself
(Apache-2.0) and carries omniio's MIT license; `kaldiio` is not a dependency
and none of its code is used.

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
- **Output shape**: `(frames, channels)` as `float32`
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
