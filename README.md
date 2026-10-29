# Omni-IO

**Random access to multi-modal training data, stored as the bytes the codecs produce.**

Omni-IO packs audio, video, images, text, MIDI, token streams and single-cell / genomics data
into flat binary archives behind one Parquet index. Every sample is a byte range, so any item,
or a time window inside it, is one read, locally or over HTTP, through one API.

## Why Omni-IO?

Training corpora are stored in the order they were collected and read through formats that
trade random access for throughput. Per-file storage pays a filesystem operation per read
and collapses past a few million files; sequential containers (WebDataset, TFRecord) reach a
sample only by scanning its shard, so training falls back to a shuffle buffer; indexed
stores (HDF5, Parquet payloads) hold decoded arrays or impose row-group reads. Omni-IO keeps
the native codec bytes, indexes them by byte range, and makes a random read as cheap as a
sequential one, so a loader can sample uniformly without a buffer and read a window of a
long recording without decoding the rest.

## Features

- **Multi-format**: one container and one index for every modality; modality-specific
  writers and readers are registered by name, and files of any other type are stored as
  they are.
- **Fast reading**: one `pread` per item on a cached descriptor, a lookahead pool to keep
  reads in flight in random order, batched reads for DataLoaders.
- **Native codecs**: payloads are stored as the codec bytes they arrive in; decoding happens
  on read, never on write.
- **Efficient storage**: large blob files plus a Parquet index that filters with Arrow,
  pandas or DuckDB without touching the media.
- **Partial reads**: a time or frame window of an audio, video or MIDI entry, a token or
  level window of a code stream, a genomic interval, read in place.
- **Parallel writing**: lock-free ingestion, every worker writing its own shard, merged by
  rename.
- **Remote access**: the same index served over HTTP range requests, from a range server or
  the Hugging Face Hub.
- **Kaldi ark/scp** read and write through `omniio.kaldi`, an MIT-licensed drop-in for `kaldiio`.

## Modalities

| modality | input | partial reads |
|---|---|---|
| audio | FLAC, WAV, Ogg/Opus, MP3, WebM/Opus, anything libsndfile opens | time window |
| video | MP4, MOV, MKV, WebM, AVI, anything PyAV opens | frame or time window, with or without audio |
| image | PNG, JPEG, WebP, GIF, TIFF, BMP | |
| text | UTF-8 strings and text files (zstd-compressed) | |
| midi | Standard MIDI Files, `pretty_midi` objects | time window, optional synthesis |
| discrete | integer streams from any neural codec (NumPy arrays, token ids) | frame, time or codebook window |
| cell | sparse rows (counts, peaks, markers) from SciPy matrices or `.h5ad` files | gene subset after the read |
| seq | FASTA genomes | genomic interval |
| track | bigWig signal tracks | genomic interval |

## Installation

```bash
pip install -e .                  # audio, video, text, image, discrete, cell, seq, track
pip install -e '.[midi]'          # + MIDI;  '.[synth]' adds FluidSynth synthesis
pip install -e '.[cell,genome]'   # + batched single-cell reads (scipy), bigWig ingestion (pyBigWig)
```

Python 3.10 or newer.

## Experiments

See `experiments/README.md` for code/scripts to reproduce the paper experiments.

## Quickstart

### Write

```python
from omniio.blob.blob import Blob

blob = Blob("./corpus", modality="audio")
blob.append(items=wav_paths, ids=utt_ids, num_workers=16, target_format="flac")
```

`fast_mode=True` stores the bytes as they are, with no transcode and no metadata; it works
for every modality and for any file type Omni-IO has no decoder for.

```python
Blob("./pdfs", modality="pdf").append(items=pdf_paths, ids=doc_ids, num_workers=16)
```

### Read

```python
from omniio.interface import audio_read

m = blob.get_metadata().slice(i, 1).to_pydict()
path, s, e = m["path"][0], m["start_byte"][0], m["end_byte"][0]

x = audio_read(path, s, e - s)                              # the whole entry
x = audio_read(path, s, e - s, start_time=12.0, end_time=15.0)   # a 3 s window, read in place
x.array, x.sample_rate                                      # (frames, channels) float32
```

Every modality has the same entry point: `video_read`, `image_read`, `text_read`,
`midi_read`, `discrete_read`, `cell_read`, `seq_read`, `track_read`, and `bytes_read` for the
raw bytes of anything.

## Archive layout

```
corpus/
  blob_0.bin, blob_1.bin, ...   native codec bytes, concatenated (320 MB per file by default)
  metadata.parquet              one row per entry: id, path, start_byte, end_byte, bin_index + per-modality columns
```

## More

- `examples/`: PyTorch datasets over archives, remote loading, single-cell ingestion from `.h5ad`.
- `experiments/`: the code behind the paper's figures, one script per figure.

## License

MIT.
