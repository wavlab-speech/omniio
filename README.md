# Omni-IO

Efficient Python library for reading and writing multimedia data (audio, video, text, images, MIDI) from binary archive blobs with support for both local and remote HTTP range requests.

## Features

- **Multi-format support**: Audio (FLAC, WAV, WebM/Opus), Video (MP4), Text (zstandard compressed), Images (PNG/JPEG), MIDI (Standard MIDI Files, optionally zstd-compressed)
- **Local and remote access**: Seamlessly read from local files or remote URLs using HTTP range requests
- **Efficient storage**: Binary blob archives with PyArrow/Parquet metadata indexing
- **Frame-level slicing**: Extract specific time ranges from audio/video/MIDI without loading entire files
- **MIDI synthesis**: Render a MIDI entry (or a time window of it) to a waveform through FluidSynth
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
pip install -e .              # audio, video, text, image
pip install -e '.[midi]'      # + MIDI (pretty_midi)
pip install -e '.[synth]'     # + MIDI synthesis (pyfluidsynth; also needs libfluidsynth, see below)
```

MIDI support is an extra so that tools that use omniio only for audio or text do not pull
in `pretty_midi`. Without it everything else works, and the MIDI entry points raise an
`ImportError` naming the extra.

### Import paths

Import a modality from `omniio.<modality>` (`omniio.audio`, `omniio.midi`, ...), never from
`omniio.modalities.<modality>`. The `modalities/` directory is where the code happens to
live; the short paths are the stable public ones and keep working when the tree is
reorganised (see `omniio/__init__.py`).

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

##### Windowed reads of WebM/Opus

FLAC, WAV and OGG windows are seeked by sample through libsndfile and are exact. WebM can
only be seeked to a cluster boundary, so for WebM/Opus the reader seeks 0.25 s early,
places the decoded frames on the stream's own sample timeline, and trims to the window. A
window is the same samples a full decode has at that offset, with two documented limits:

- **Not bit-exact after a seek.** The decoder starts cold at a cluster, so the first
  samples of a seeked window differ from a full decode by about 2e-4 relative RMS. If you
  need bit-exact windows, read from 0 and slice.
- **Streams with timestamp gaps** (DTX silence, dropped packets): a seeked read follows
  the timestamps, a full decode concatenates frames, so the two disagree by the size of
  any gap before the window.

The reader anchors a seeked run on the first packet's pts, measured from the pts of the
stream's first packet. That is the one timestamp that means the same thing across
PyAV/ffmpeg builds; the two it deliberately does not use both changed:

| PyAV | ffmpeg (libavcodec) | trimmed frame's pts after a seek | `stream.start_time` | windowed reads |
|---|---|---|---|---|
| 9.2 (conda-forge) | 4.4 (58.134) | moved past the discarded samples | first packet's pts | correct on pre-encoded files (48 windows, 3 files); the test fixtures cannot be encoded there (no libopus encoder in that build) |
| 10.0 (conda-forge) | 5.1 (59.37) | moved | first packet's pts; files muxed *by* ffmpeg 5.1 start at -7 ms | correct |
| 10.0, 11.0 (conda-forge) | 6.0 (60.3), 6.1 (60.31) | moved | first packet's pts | correct |
| 12.0, 12.3 | 6.1 (60.31) | moved | first packet's pts | correct |
| 13.0, 13.1 | 7.0 (61.3) | moved | first packet's pts | correct |
| 14.0, 14.2, 15.1 | 7.1 (61.19) | moved | first packet's pts | correct |
| 16.1 | 8.0 (62.11) | moved | first packet's pts | correct |
| 17.1, 18.1 | 8.1 (62.28) | moved | first packet's pts **plus the codec delay** | correct |

Rows without a channel are the PyPI wheels, which bundle their own ffmpeg; the conda-forge
rows pair the same PyAV source with a conda ffmpeg and show the same timestamps as the
wheels of that ffmpeg generation (PyAV 12 and later require ffmpeg 6.1 or newer). "Correct"
means `tests/test_webm_opus_windows.py` passes in full (139 cases on real libopus files
across mux layouts, source rates, channel counts and window edges, plus 152 on synthetic
streams that pin the framing by hand). The wheel sweep is reproducible with
`scripts/sweep_pyav_webm.sh`. Earlier anchors fail on every build in the table: measuring
from the trimmed frame's pts puts every seeked window one codec delay (336 samples at
48 kHz) late, and measuring from `stream.start_time` is off by the same amount on PyAV 17
and later. A build we could not test is one whose decoder does not discard the codec delay
after a seek; the real-file tests would catch that at once.

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


#### MIDI

```python
from omniio.interface import midi_read

# Read a MIDI entry; optionally restrict it to a time window (seconds, like audio)
result = midi_read(
    archive_path="/path/to/archive.bin",
    start_offset=1024,
    file_size=5000,
    start_time=5.0,   # optional
    end_time=10.0,    # optional
)

result.midi        # pretty_midi.PrettyMIDI, times re-zeroed to the window
result.notes       # structured array: onset, offset, pitch, velocity, program, is_drum, instrument
result.duration    # 5.0
result.to_bytes()  # the window as a Standard MIDI File; result.write("slice.mid") also works

# Synthesize while reading: `array` is (frames, channels) float32 covering exactly `duration`
result = midi_read(archive_path, start_offset, file_size, start_time=5.0, end_time=10.0,
                   synthesize=True, sample_rate=24000)
result.array.shape  # (120000, 1)
```

Notes that *sound* inside the window are kept and clipped to it (so a note held across
`start_time` becomes a note starting at 0); pass `include_partial=False` to keep only notes
whose onset lies inside. Sustain-pedal / pitch-bend state at `start_time` is carried in, so
a synthesized window sounds as it would in the full file.

`notes` is sorted by `(onset, is_drum, program, pitch)`, a deterministic order for chords
that a sequence model can be trained against.

Synthesis uses FluidSynth when available (`pip install omniio[synth]` plus the
`libfluidsynth` shared library, e.g. `conda install -c conda-forge fluidsynth`) with the GM
SoundFont bundled in `pretty_midi`, or the one named by `$OMNIIO_SOUNDFONT` / the
`soundfont=` argument. Without FluidSynth it falls back to additive sine tones
(`backend="sine"`, drums silent).

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
    target_bit_depth=16,
    skip_errors=True,   # skip unreadable items instead of aborting; returns [(id, error), ...]
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


#### MIDI Archives

```python
from omniio.blob.blob import Blob
from omniio.midi.common import midi_from_notes

blob = Blob(archive_dir="./my_midi_archive", modality="midi")

# Items may be .mid paths, raw Standard-MIDI-File bytes (e.g. a parquet binary column),
# or pretty_midi.PrettyMIDI objects — mix freely.
blob.append(
    items=["a.mid", midi_bytes, midi_from_notes([(0.0, 0.5, 60), (0.5, 1.0, 64)])],
    ids=["a", "b", "c"],
    num_workers=4,
    compress=True,   # zstd; the reader detects it
)
```

Per-entry metadata: `duration`, `n_notes`, `n_instruments`, `programs`, `has_drums`,
`resolution`, `min_pitch`, `max_pitch`, `format` (`midi` / `midi.zst`), sizes.

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
`load_scp_sequential`, `load_wav_scp`, `load_mat`, `load_segments`, `save_ark`,
`save_mat`, `open_like_kaldi`, `parse_specifier`, `parse_rspecifier`,
`parse_wspecifier`, `LazyLoader`, `SegmentedLoader`, `ReadError`.

`load_scp` and `load_scp_sequential` take `separator=` for an scp with its own
delimiter and `segments=` to key the result by a segments file instead of by
recording; `load_mat` takes `fd_dict=`, a caller-owned handle cache worth using
when reading many entries out of a few archives.

The code lives in `omniio/tools/kaldi/`, but `omniio.kaldi` is the supported
import path and `import omniio.kaldi`, `from omniio import kaldi` and
`from omniio.kaldi import ReadHelper` all work.

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

Every public name `kaldiio` exports is present. The remaining differences are
additive optional arguments (`endian=` on the readers, `write_kwargs=` on
`WriteHelper`, `return_position=` on `load_ark`) and the name of the first
parameter on four functions, which matters only if you pass it by keyword:

| | `kaldiio` | `omniio.kaldi` |
|---|---|---|
| `ReadHelper` | `wspecifier` | `rspecifier` — it reads, so that is what it takes |
| `load_ark` | `fname` | `file_or_fd` — an open file is accepted too |
| `load_mat` | `ark_name` | `name` |
| `save_mat` | `fname` | `path` |

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

### MIDI
- **Input**: Standard MIDI Files (`.mid`/`.midi`), raw SMF bytes, or `pretty_midi.PrettyMIDI`
- **Storage**: native SMF bytes, optionally zstd-compressed
- **Output**: `pretty_midi.PrettyMIDI` + a `(onset, offset, pitch, velocity, program, is_drum, instrument)` note table; optional synthesized waveform `(frames, channels)` float32

## Requirements

- Python >= 3.8
- numpy
- av (PyAV)
- soundfile
- requests
- zstandard
- pyarrow
- pretty_midi (optional: `omniio[midi]`, MIDI read/write)
- pyfluidsynth + libfluidsynth (optional: `omniio[synth]`, MIDI synthesis)

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
