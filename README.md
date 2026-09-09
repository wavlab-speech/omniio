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

#### Discrete sequences (VQ / RVQ codes, token ids)

```python
from omniio.interface import discrete_read

# Codec agnostic: an entry is N integer streams, each with its own length, alphabet size
# and optional rate; RVQ is N equal-length streams sharing one rate.
r = discrete_read(archive_path, start_offset, file_size,
                  start_level=0, end_level=2,      # optional: a window of streams / RVQ levels [start, end)
#                 streams=[0, 3],                  # ... or an explicit list of stream indices
                  start_frame=75, end_frame=188)   # optional: partial read in frames (elements)
#                 start_time=1.0, end_time=2.5     # ... or in seconds through each stream's own rate
r.array          # (n_streams, T) when the lengths agree; uint8/16/32 = narrowest dtype for the alphabet
r.streams        # list of 1-D arrays (ragged lengths are fine); r.to_array(pad_value=-1) pads
r.vocab_sizes, r.rates, r.lengths, r.stream_indices, r.frame_windows
r.bytes_read     # the I/O actually done for this read (vs r.entry_size)
```

Partial reads are true partial reads, not decode-then-slice: the reader fetches the
216-byte header, then only the byte span of each requested stream window (frame windows map
to a bit offset inside a stream) — locally via seek, remotely as one HTTP range request per
span; a level window with no frame window is a single contiguous read because streams are
stored back to back. For a 10-minute 8×75 Hz RVQ entry (450 KB), the first two codebooks
read 25% of the bytes and a 3 s window reads 0.5%. Entries written with `compress=True`
(zstd over the whole payload) are the exception: they read the entry once and window after
decompressing.

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

#### Discrete Archives

```python
from omniio.modalities.discrete.write import discrete_write

raw_bytes, metadata = discrete_write(
    codes,                     # (n_streams, T) array (codebook-major), 1-D array, list of 1-D arrays
    "utt_001",                 # (ragged ok), .npy/.npz path, torch tensor, or {"streams", "vocab_sizes", "rates", "codec"}
    vocab_size=1024,           # int or one per stream; sets the bit width (pass it: max+1 is the fallback)
    rate=75.0,                 # units/second, int or per stream; None = no time axis (plain token ids)
    codec="encodec_24khz_6kbps",   # free-form provenance: stored, never interpreted
    compress=False,            # zstd-wrap the bit-packed payload (only pays off for very repetitive codes)
)
# metadata: n_streams, lengths, vocab_sizes, bits, rates, duration, n_tokens, ragged, codec,
#           format ('discrete' | 'discrete.zst'), original_size, stored_size
```

Storage: each stream is bit-packed at `ceil(log2(vocab))` bits — 1.25 bytes/token for
1024-way codes, the fixed-width optimum (zstd on top gains nothing on near-uniform VQ
codes) — and streams sit back to back, so a subset decodes without the rest.

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

### Discrete sequences

`b"ODSQ"` header (version, flags, `n_streams`, per stream `length | bits | vocab | rate`) followed by the
bit-packed streams; `discrete.zst` wraps the payload in zstandard.

## Requirements

- Python >= 3.8
- numpy
- av (PyAV)
- soundfile
- requests
- zstandard
- pyarrow
- pretty_midi (MIDI)
- pyfluidsynth + libfluidsynth (optional, MIDI synthesis)

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
