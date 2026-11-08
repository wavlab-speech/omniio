# OMAC: omniio multichannel audio codec

Status: draft, version 1.

**Name.** OMAC stands for **O**mniio **M**ultichannel **A**udio **C**odec. It is
unrelated to OMAC, the One-Key MAC (NIST CMAC) used in cryptography. Entries
start with the magic bytes `b"OMAC"`, the Rust crate and the Python module are
both `omac`, and the archive `format` column is `"omac"`.

OMAC is a lossless codec for integer PCM with any number of channels, built for
reading time windows and channel subsets of long multi-microphone recordings
out of omniio archives, locally or over HTTP range requests. The prediction and
residual coding follow FLAC; the container does not.

## 1. Goals and non-goals

Goals:

- 1 to 65535 channels, 4 to 32 bits per sample, any sample rate.
- Random access in time: finding the bytes of block `k` is arithmetic plus one
  index lookup; no bisection search, no sync-code scan.
- Random access in channels: each channel of each block is a separate,
  byte-aligned, length-prefixed payload, so a reader skips channels without
  parsing them. Decoding one channel of 16 costs 1/16 of the work.
- All channels of a block are adjacent, so a time window of every channel is
  one contiguous byte range (one HTTP request).
- Compressed size within about 1% of FLAC on speech arrays.
- Corruption is detected (CRC-32C per block and over the header).
- The reference encoder is deterministic: the same samples and settings give
  the same bytes.

Non-goals:

- **Inter-channel coding of any kind**, including FLAC's stereo modes. Measured
  on CHiME-6 (S02 U01, 4 ch) and AMI (ES2003a Array1, 8 ch), predicting a
  channel from a reference channel saved 1.8% and 2.2%: the residual is
  dominated by noise that is uncorrelated across microphones. That gain does
  not pay for making one channel's decode depend on another's.
- Lossy coding, floating-point samples, streaming (append-only) writes.
- Container metadata (tags, channel names, geometry). That lives in the
  archive's `metadata.parquet`.

## 2. Conventions

- Header fields are little-endian unsigned integers unless stated otherwise.
- Payloads (section 5) are bitstreams read MSB-first; signed `n`-bit fields
  are two's complement.
- `B` = `block_size`, `C` = `channels`, `NB` = `num_blocks`,
  `bps` = `bits_per_sample`.
- Offsets are relative to the first byte of the entry (the `O` of `OMAC`).
- A decoder MUST reject any field value this document does not allow,
  including non-zero reserved fields and unknown flag bits.

## 3. Entry layout

```
+-------------------------------+  0
| fixed header (40 bytes)       |
+-------------------------------+  40
| valid_frames: u64 x C         |
+-------------------------------+  40 + 8C
| block_offset: u64 x (NB + 1)  |
+-------------------------------+  40 + 8C + 8(NB + 1) = block_offset[0]
| block 0                       |
| block 1                       |
| ...                           |
| block NB-1                    |
+-------------------------------+  block_offset[NB] = entry size
```

### 3.1 Fixed header

| Offset | Size | Field             | Constraint                                     |
|-------:|-----:|-------------------|------------------------------------------------|
| 0      | 4    | `magic`           | `b"OMAC"`                                      |
| 4      | 2    | `version`         | `1`                                            |
| 6      | 2    | `flags`           | bit 0 `PADDED` (3.2); other bits 0             |
| 8      | 2    | `channels`        | 1..65535                                       |
| 10     | 1    | `bits_per_sample` | 4..32                                          |
| 11     | 1    | reserved          | 0                                              |
| 12     | 4    | `sample_rate`     | > 0, Hz                                        |
| 16     | 4    | `block_size`      | 16..65536 frames                               |
| 20     | 4    | `num_blocks`      | `ceil(num_frames / B)`; 0 iff `num_frames` = 0 |
| 24     | 8    | `num_frames`      | frames per channel                             |
| 32     | 4    | `header_crc`      | CRC-32C of bytes `[0, 32)` and `[40, block_offset[0])` |
| 36     | 4    | reserved          | 0                                              |

### 3.2 Channel lengths

`valid_frames[c]` is the number of real samples in channel `c`;
`valid_frames[c] <= num_frames`. Samples past it are zero padding (section 7).
`PADDED` is set iff some `valid_frames[c] < num_frames`. Decoders return
`num_frames` frames for every channel and expose `valid_frames`.

### 3.3 Block index

`block_offset[k]` is the offset of block `k`; `block_offset[NB]` is the entry
size, so block `k` occupies `[block_offset[k], block_offset[k+1])`. Offsets
are strictly increasing and `block_offset[0] = 40 + 8C + 8(NB + 1)`.

Block `k` holds frames `[k*B, min((k+1)*B, num_frames))`; only the last block
may be shorter. Every quantity needed to locate a block is either in the
header or arithmetic on it, so a reader that knows `C` and `B` (they are also
written to `metadata.parquet`) can fetch exactly the index entries of a
window without reading the rest of the header.

## 4. Block layout

```
u32  block_crc              CRC-32C of the rest of the block
u32  payload_size[C]        bytes, each >= 1
     payload[0] ... payload[C-1]
```

Channel `c` starts at `4 + 4C + sum(payload_size[0:c])` from the block start,
and `4 + 4C + sum(payload_size)` equals the block's size from the index. A reader that wants a subset of
channels reads the size table and seeks; it MAY skip `block_crc` verification
when it does not read the whole block, and MUST verify it otherwise unless the
caller disabled checking.

## 5. Channel payload

One payload codes `n` samples (`n` = the block's frame count) of one channel.
It starts and ends on a byte boundary; bits after the last field up to the
boundary are zero.

```
2 bits   type        0 CONSTANT, 1 VERBATIM, 2 FIXED, 3 LPC
6 bits   wasted      0..bps-1 (see 5.1)
...      type body
```

Let `w = bps - wasted`, the width of the coded values.

| Type     | Body                                                                              |
|----------|-----------------------------------------------------------------------------------|
| CONSTANT | `w` bits signed: the value of every sample                                        |
| VERBATIM | `n` x `w` bits signed                                                             |
| FIXED    | 3 bits order `p` (0..4); `w` bits signed `x[0]`; residual (5.3)                   |
| LPC      | 5 bits `p-1` (`p` = 1..32); 4 bits `prec-1` (`prec` = 1..15); 5 bits `shift` (0..31); `p` x `prec` bits signed `coef[1..p]`; `w` bits signed `x[0]`; residual (5.3) |

### 5.1 Wasted bits

Samples are coded shifted right by `wasted`: the decoder computes the values
`v[i]` and outputs `v[i] << wasted`. It removes trailing zero bits shared by
every sample of the block (for example 16-bit audio stored in a 24-bit file).

### 5.2 Prediction

`x` are the coded values (before the `wasted` shift). For `i = 1..n-1`,
`x[i] = pred(i) + r[i]` where `r` is the residual and:

| `i`              | `pred(i)`                                  |
|------------------|--------------------------------------------|
| `i = 1 < p`      | `x[0]`                                     |
| `2 <= i < p`     | `2x[i-1] - x[i-2]`                         |
| `i >= p`         | FIXED: the order-`p` polynomial predictor below; LPC: `(sum_{j=1..p} coef[j] * x[i-j]) >> shift` |

The first two rows (warm-up) apply only while `i < p`.

Fixed polynomial predictors (FLAC's):
`p=0: 0`, `p=1: x[i-1]`, `p=2: 2x[i-1] - x[i-2]`,
`p=3: 3x[i-1] - 3x[i-2] + x[i-3]`,
`p=4: 4x[i-1] - 6x[i-2] + 4x[i-3] - x[i-4]`.

Arithmetic is on 64-bit signed integers; `>>` is an arithmetic shift (floor).
For `bps <= 32`, `prec <= 15` and `p <= 32` the LPC sum fits in 51 bits.
Every reconstructed `x[i]` MUST lie in `[-2^(w-1), 2^(w-1))`; a decoder
rejects the block otherwise.

The warm-up rule (rows 1 and 2) replaces FLAC's verbatim warm-up samples:
blocks stay independent, and only `x[0]` is stored raw.

### 5.3 Residual

The residual covers `r[1..n-1]`, coded with partitioned Rice codes:

```
4 bits   po             partition order 0..15; n must be divisible by 2^po
for each of the 2^po partitions:
  5 bits   k            Rice parameter 0..30; 31 = escape
  if escape: 6 bits  e  raw width 0..63, then each residual as e bits signed
                        (e = 0: every residual of the partition is 0)
  else:     each residual Rice-coded with parameter k
```

Partition `j` holds `n >> po` residuals, except partition 0, which holds
`(n >> po) - 1` (because `r[0]` does not exist).

Rice code of a residual `r`: map it to `u = (r << 1) ^ (r >> 63)` (zigzag),
write `u >> k` zero bits, a one bit, then the low `k` bits of `u`.

Reading past the end of the payload is an error.

## 6. Integrity

- `header_crc` covers the fixed header (except the CRC and the reserved word
  after it), `valid_frames` and the block index.
- `block_crc` covers each block from its size table to its last byte.
- CRC-32C (Castagnoli) is used because x86 (SSE4.2) and ARMv8 compute it in
  hardware.

## 7. Channels of unequal length

The encoder takes channels from one multichannel file, several files (one per
channel or group), or an array. If their lengths differ, `length_mismatch`
decides:

| Value       | Effect                                                                                                           |
|-------------|------------------------------------------------------------------------------------------------------------------|
| `"error"`   | Default. Raise `ValueError` naming every channel and its length, e.g. `S12_U05.CH1.wav: 1,234,567 frames (77.2 s)` |
| `"truncate"`| `num_frames = min` length; samples past it are dropped; `valid_frames[c] = num_frames`                           |
| `"pad"`     | `num_frames = max` length; shorter channels are zero-filled; `valid_frames[c]` = original length; `PADDED` set |

For both `"truncate"` and `"pad"` the encoder also records the original lengths
in the archive row (`source_frames`). Differing sample rates are always an
error. Differing integer bit depths are coded at the largest; floating-point
input is an error.

Why the default is an error: in CHiME-6 the four channels of one Kinect always
match, but different arrays of one session differ by 0.2 to 2.8 s, and by
934 s in S12. Truncating silently there would drop 15 minutes from five
arrays.

## 8. Reading

Non-normative; describes the intended reader.

- **Time window** `[t0, t1)`: frames `[f0, f1)` by rounding as audio slices do
  today; blocks `k0 = f0 // B` to `k1 = (f1 - 1) // B`; bytes
  `[block_offset[k0], block_offset[k1 + 1])`; trim the decoded frames.
- **Header and index**: the first read takes the first 512 KiB of the entry,
  which holds the fixed header, `valid_frames` and the whole block index of
  entries up to 65536 blocks (about 4.6 h at 16 kHz, 1.5 h at 48 kHz, with
  `B` = 4096). A longer index is read whole if it is at most 4 MiB (one more
  request), and otherwise only the entries `k0..k1+1` of the window are read. The header and index are cached per entry and process,
  so a cold remote read costs two requests and a warm one a single request.
- **Channel subset**: when the subset is small relative to `C`, the reader
  MAY fetch the size tables and then coalesced runs of payloads instead of
  whole blocks; otherwise it fetches whole blocks and skips locally.
- **Output**: `(frames, channels)`, float32 in [-1, 1) by default, or the
  integer dtype on request, decoded straight into the output array.

## 9. Reference implementation

- Rust crate (`omac`), Python bindings through pyo3, wheels through maturin.
  Shipped as an optional extra (`omniio[omac]`) imported lazily, as with MIDI,
  so omniio itself stays pure Python.
- Decode parallel over (block, channel) pairs with rayon, GIL released. The
  thread count is a parameter, default 1, because a PyTorch DataLoader already
  runs one process per worker.
- Encoder (non-normative choices): Levinson-Durbin on windowed
  autocorrelation, orders up to 32 at the highest level, FIXED and CONSTANT
  tried for every payload, partition order and Rice parameters chosen by exact
  bit count. Compression levels 0..8.
- omniio integration: `_detect_format` maps `b"OMAC"` to `omac`;
  `audio_write(..., target_format="omac", length_mismatch=...)` accepts a
  list of paths; `audio_read(..., channels=[...])`. Archive rows carry
  `channels, sample_rate, bits_per_sample, block_size, num_frames,
  num_blocks`.

## 10. Testing

- Bit-exact round trip on: CHiME-6 arrays, AMI MDM arrays, random noise,
  silence (CONSTANT), full-scale square waves, 1-frame and `B±1`-frame
  entries, `bps` 4/8/16/24/32, `C` = 1, 2, 16, 300.
- Every field limit and every error path (truncated payload, bad CRC, reserved
  bits, out-of-range sample) has a test; the decoder is fuzzed with
  `cargo-fuzz`.
- Size compared with FLAC, and decode speed compared with soundfile (whole
  entry, a 1 min window, one channel), on the same CHiME-6 and AMI excerpts as
  the prototype.

## 11. Open questions

- Default `B`: 4096 frames (256 ms at 16 kHz) matches FLAC; a smaller `B`
  makes short windows cheaper and costs header bits.
- Packaging: a separate optional wheel (proposed above) versus building
  omniio itself with maturin.
