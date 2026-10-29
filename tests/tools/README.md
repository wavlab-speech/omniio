# tests/tools/

Development tools you run by hand; not part of the installed package.

## `sweep_pyav_webm.sh`

Reruns `tests/test_webm_opus_windows.py` against several PyAV wheels, each installed
into its own directory and put on `PYTHONPATH` for that run only, and prints one row
per version: the `av` it actually imported, the bundled libavcodec, what the build
reports for `stream.start_time` and for the first frame after a seek, and the test
result. A row whose imported `av` does not match the requested version is reported
as a mismatch rather than labelled. Used to set the validated range in
`omniio/modalities/audio/read.py` (`_TESTED_PYAV`, `_TESTED_LIBAVCODEC`).

```bash
tests/tools/sweep_pyav_webm.sh                 # one wheel per ffmpeg generation
tests/tools/sweep_pyav_webm.sh 18.1.0 19.0.0   # specific versions
```

## `webm_decode_check.py`

The decode-only half of the sweep, for a PyAV that has no Opus encoder (the
conda-forge 9.2 build for linux-aarch64 is one): give it a WebM/Opus blob
encoded elsewhere and it reads 25 windows through `omniio` and compares them
with the full decode. Exit status 1 means a window was misaligned. Builds that
cannot be installed from PyPI wheels (PyAV 9.2, 10.0, 11.0 need Python <= 3.10)
come from conda-forge: `conda create -p ENV -c conda-forge python=3.10 av=9.2.0
numpy pytest requests pysoundfile`, then `pip install zstandard pyarrow` into it.

## `compare_with_kaldiio.py`

Reads every Kaldi archive under some directories with **both** `omniio` and
`kaldiio` and compares what comes back. Use it to check `omniio` against data
you already have, rather than against fixtures someone else wrote.

```bash
pip install omniio kaldiio
python tests/tools/compare_with_kaldiio.py /path/to/your/dump
```

Exits non-zero if anything mismatched, so it works as a gate.

### What it reports

The summary is grouped **by on-disk format**, because "1420 files, 0
mismatches" is worth very little if all 1420 were the same kind of record:

```
=== by format (this is the part that matters) ===
  WaveHolder (RIFF)                     1200   mismatched 0
  CM (compressed, per-column)            220   mismatched 0
```

If a format you care about is missing from that list, it was not exercised —
point the script at a directory that has some.

### Useful flags

| flag | meaning |
|---|---|
| `--max-per-file N` | entries sampled per `scp` (default 25; `0` reads all) |
| `--max-files N` | stop after N `scp` files |
| `--allow-pipes` | also read `scp` entries that are shell pipes |
| `--out FILE` | write one line per problem, as TSV |

### About `--allow-pipes`

Off by default, because reading such an entry means **running the command in
it**, and the command comes out of a file rather than from you. With the flag:

- Each stage's executable must match a key of `ALLOWED_PROGRAMS`
  (`sox`, `sph2pipe`, `ffmpeg`, `flac`, …) **exactly**. A path-qualified
  `/tmp/payload/sox` has the right basename but is not sox, so it is skipped —
  put the directory on `PATH` instead, which is the same thing without letting
  the `scp` file choose which binary runs.
- The *arguments* are checked too, not just the program. Every one of these
  except `cat` will write a file if told to, and two of them delete their
  input while doing it (`gunzip archive.gz`, `flac -d x.flac`). A Kaldi pipe
  always writes its object to stdout, so that is what is required — which
  costs nothing on real data and makes "read-only" true rather than intended.
- Stages are exec'd directly, never through a shell, so nothing in the entry
  is interpreted as shell syntax.
- Output is streamed and capped (`PIPE_MAX_BYTES`, 512 MiB) under a timeout
  (`PIPE_TIMEOUT`, 300 s), so a stage that never stops cannot fill memory or
  the scratch disk.

Anything that fails those checks is counted as skipped, and the summary says
why.

Turn it on if your data has speed perturbation. `sox` piping to stdout cannot
seek back to patch its WAV header, so it leaves a placeholder size there — and
that is what `sph2pipe … | sox … speed 0.9 |` produces. omniio before 0.1.2
refused those outright.

A pipe is run **once** and the captured bytes are given to both libraries.
Running the command twice would differ on the samples regardless of what the
libraries do: a `sox` stage without `-R` dithers non-repeatably.

### Read-only

Nothing outside `--out` is written, and your archives are only ever opened for
reading.

### Why kaldiio

It is the reference implementation here, and is needed only to run this
script. It is **not** a dependency of omniio, and omniio's Kaldi layer was
written without reading its source — see
[`omniio/tools/kaldi/PROVENANCE.md`](../omniio/tools/kaldi/PROVENANCE.md).
