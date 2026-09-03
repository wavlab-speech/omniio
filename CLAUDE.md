# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`omniio` is a Python library for reading and writing multimedia data (audio, video, text) from binary archive blobs. It supports both local file access and remote HTTP range requests, enabling efficient random access to large multimedia archives.

## Architecture

### Modality-Based Structure

The library is organized by data modality, with each supporting read and write operations:

- **audio/**: Audio file I/O (FLAC, WAV, WebM/Opus)
- **video/**: Video file I/O (MP4 with audio streams)
- **text/**: Text file I/O (zstandard compressed)
- **blob/**: Binary archive management with PyArrow metadata
- **tools/**: Format-specific helpers that are not omniio's own archive layout;
  currently **tools/kaldi/**, a `ark`/`scp` compatibility layer (drop-in for `kaldiio`)

### Core Components

**definitions.py**: Dataclass definitions for read operations
- `ArchiveRead`: Base class with file_type and modality
- `AudioRead`: Extends with sample_rate and numpy array
- `VideoRead`: Extends with fps, dimensions, and separate audio/video arrays
- `TextRead`: Extends with text string

**__init__.py**: Public import aliases. The source tree groups modules by what
they are, but the stable import paths are the short ones (`omniio.kaldi`, not
`omniio.tools.kaldi`). A meta path finder built from the `_ALIASES` dict maps
public names -- and their submodules -- onto the real ones, lazily, so
`import omniio` pulls in nothing extra. To relocate a subpackage, move it and
add a line to `_ALIASES`; importers do not change. The finder is *prepended* to
`sys.meta_path` on purpose: left to the normal path finder, a submodule such as
`omniio.kaldi.compression` would be loaded a second time under the alias, with
duplicate module state, because the aliased parent's `__path__` still points at
the real directory.

**interface.py**: Main entry points that route to local or remote readers
- `audio_read()`: Checks if path exists, routes to local or remote implementation
- `text_read()`: Same pattern for text data

**blob/blob.py**: Main `Blob` class for managing binary archives
- Archive structure: `archive_dir/blob_0.bin`, `blob_1.bin`, ..., `metadata.parquet`
- Uses PyArrow tables for efficient metadata operations
- Supports parallel append operations with ProcessPoolExecutor
- Manages multiple bin files with configurable max size (default 320MB)

**blob/write.py**: Registry mapping modalities to their write functions
- `modality_writer` dict: {'audio': audio_write, 'text': text_write}

### Key Design Patterns

**Local vs Remote Reading**: All read functions check `os.path.exists()` and route to either:
- Local: Direct file seek/read operations
- Remote: HTTP range requests via `requests.get()` with `Range: bytes=start-end` headers

**Format Detection**: Uses magic bytes for audio format detection:
- `b"fLaC"` → FLAC
- `b"RIFF"` → WAV
- `b"\x1aE\xdf\xa3"` → WebM/Matroska
- `b"OggS"` → OGG

**Time-Based Slicing**: Audio and video readers support optional time-based extraction:
- `start_time`, `end_time` parameters in seconds
- Video also supports frame-based slicing with `start_frame`, `end_frame`
- Frame indices take priority over time when both provided

**Streaming Operations**: Archive operations avoid loading entire files into memory:
- Blob append uses `shutil.copyfileobj()` for streaming concatenation
- Only file sizes are read to calculate byte offset shifts
- Metadata operations use PyArrow for efficient columnar operations

## Reading from Archives

### Audio Reading

```python
from omniio.interface import audio_read

# archive_path: path or URL to .bin file
# start_offset: byte offset where entry begins
# file_size: number of bytes for this entry
# start_time/end_time: optional time slicing in seconds
result = audio_read(archive_path, start_offset, file_size, start_time=5.0, end_time=10.0)
# Returns AudioRead with sample_rate and array (frames, channels) as float32
```

### Video Reading

```python
from omniio.video.read import video_read_local, video_read_remote

# Supports frame-based or time-based slicing
result = video_read_local(archive_path, start_offset, file_size,
                          start_frame=100, end_frame=200)
# Returns VideoRead with:
#   - video_array: (frames, height, width, 3) uint8
#   - audio_array: (samples, channels) float32
#   - fps, sample_rate, height, width
```

### Text Reading

```python
from omniio.text.read import text_read_local, text_read_remote

result = text_read_local(archive_path, start_offset, file_size)
# Returns TextRead with decompressed text string (zstandard)
```

## Writing to Archives

### Using the Blob Class

```python
from omniio.blob.blob import Blob

# Initialize or open existing archive
blob = Blob(archive_dir="./my_archive", modality="audio", max_bin_size=320*1024*1024)

# Append items (parallelized)
blob.append(
    items=[audio_path1, audio_path2, ...],
    ids=["id1", "id2", ...],  # optional, defaults to sequential integers
    num_workers=4,  # 0 for single-process
    overwrite=False,  # True to skip duplicates
    target_format="flac",  # modality-specific kwargs
    target_bit_depth=16
)

blob.summary()  # Print archive statistics
blob.clear(confirm=True)  # Delete all bin files and metadata
```

### Audio Writing

```python
from omniio.audio.write import audio_write

raw_bytes, metadata = audio_write(
    audio_path="input.wav",
    item_id="sample_001",
    target_format="flac",  # 'flac', 'wav', 'webm'
    target_bit_depth=16    # 8, 16, 24, 32 (ignored for webm)
)
# Returns raw bytes and metadata dict with sample_rate, channels, samples, format, bit_depth
```

### Text Writing

```python
from omniio.text.write import text_write

raw_bytes, metadata = text_write(
    path_or_string="/path/to/file.txt",  # or raw string
    item_id="text_001",
    is_path=True,  # False to treat first arg as string
    compression_level=3  # zstandard level 1-22
)
# Returns compressed bytes and metadata with original_size, compressed_size
```

## Kaldi Compatibility Layer

The code lives in `omniio/tools/kaldi/` and is imported as `omniio.kaldi`
(see the alias note above). It is independent of the blob machinery -- it
depends only on numpy and, for audio payloads, soundfile.

- **compression.py**: the `CompressedMatrix` codec (`CM`/`CM2`/`CM3`). Encode
  and decode formulas are float32 with a specific operation ordering; changing
  the grouping changes the last bit and breaks byte-compatibility.
- **matio.py**: object codec plus `load_ark`/`load_scp`/`load_mat`/`save_ark`/
  `save_mat` and the `LazyLoader` mapping.
- **highlevel.py**: `ReadHelper`/`WriteHelper` and `segments` support.
- **specifier.py**, **stream.py**: rspecifier/wspecifier parsing and Kaldi
  extended filenames (pipes, `-`, `.gz`).

`tests/test_kaldi.py` runs without `kaldiio`. `tests/test_kaldi_interop.py`
skips unless `kaldiio` happens to be installed, and then asserts byte-identical
output in both directions. `kaldiio` must never become a dependency: its
license forbids redistribution, which is the reason this module exists.

## Dependencies

Key libraries used throughout the codebase:
- **av (PyAV)**: Video/audio codec operations, supports WebM/Opus
- **soundfile**: Audio I/O for FLAC, WAV, OGG formats
- **numpy**: Array operations for audio/video data
- **requests**: HTTP range requests for remote reading
- **zstandard**: Text compression/decompression
- **pyarrow/parquet**: Efficient metadata storage and operations

## Common Implementation Notes

- Audio data is normalized to float32 in range [-1.0, 1.0] with shape (frames, channels)
- Video frames are RGB24 format with shape (frames, height, width, 3) as uint8
- WebM/Opus always uses 48kHz sample rate internally (PyAV handles resampling)
- Blob workers check for duplicate IDs before writing; set `overwrite=True` to skip
- Archive byte offsets use [start_byte, end_byte) convention (end is exclusive)
- Remote reads use HTTP 206 Partial Content with `Range` headers
