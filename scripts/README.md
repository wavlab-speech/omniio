# scripts/

Things you run by hand, not part of the installed package.

## `compare_with_kaldiio.py`

Reads every Kaldi archive under some directories with **both** `omniio` and
`kaldiio` and compares what comes back. Use it to check `omniio` against data
you already have, rather than against fixtures someone else wrote.

```bash
pip install omniio kaldiio
python scripts/compare_with_kaldiio.py /path/to/your/dump
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
it**. With the flag, only pipelines built from a fixed allow-list of programs
(`sox`, `sph2pipe`, `ffmpeg`, `flac`, …) with no shell metacharacters are run;
anything else is counted as skipped rather than executed.

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
