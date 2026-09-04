# Provenance of `omniio.kaldi`

This module reads and writes Kaldi `ark`/`scp` archives and exposes the same
API as [`kaldiio`](https://github.com/nttcslab-sp/kaldiio), so that a project
can drop that dependency. This file records how it was written, because
`kaldiio`'s licence makes the question worth answering in writing rather than
from memory.

## Why it exists

`kaldiio`'s LICENSE is an NTT "SOFTWARE LICENSE AGREEMENT FOR EVALUATION".
Section 1 grants use only for internally evaluating the method in a specific
2017 paper, and section 4(b) forbids distributing, transferring or reproducing
the software. PyPI carries no licence metadata for it and GitHub reports
NOASSERTION. That is a problem for anything published downstream — see
[espnet/espnet#6529](https://github.com/espnet/espnet/issues/6529).

`omniio.kaldi` is MIT-licensed, like the rest of omniio.

## What it was written from

- The on-disk format and the compression algorithms come from **Kaldi itself**,
  which is Apache-2.0: `src/matrix/compressed-matrix.{h,cc}` for
  `CompressedMatrix` (the `CM`/`CM2`/`CM3` layouts, the `CompressionMethod`
  enum, `FloatToUint16`/`FloatToChar`/`CharToFloat`, and `ComputeColHeader`),
  and `src/util/kaldi-table-inl.h` for the `<key> <space> <object>` record
  framing. Individual functions cite their source in their docstrings.
- The remaining details — the exact float32 operation grouping in the
  quantisation formulas, the `AUDIO` framing of the extended archive layout,
  and the width of its length prefix — were established **empirically**, by
  writing archives with existing tools and reading the bytes back.

## What it was not written from

**`kaldiio`'s source code was not read, and none of it is reused.**

`kaldiio` was used only as a black-box oracle during development and is used
that way in CI today: it is installed in one isolated job, told to write an
archive, and the bytes it produces are compared with the bytes omniio produces
for the same input. See `tests/test_kaldi_interop.py`. Running the software to
evaluate an independent implementation is within the evaluation licence;
copying it is not, and would defeat the purpose of this module.

`kaldiio` is not a dependency of omniio and must not become one.

The public API surface — function names and signatures — was inspected, since
matching it is the point of a drop-in replacement. Implementation was not.

## Evidence

Mechanical comparison of `omniio/tools/kaldi/` against an installed `kaldiio`,
over code with comments, docstrings and blank lines stripped:

| | |
|---|---|
| Highest similarity of any file pair | **10.6 %** |
| Identical non-trivial lines | **46 of 982 (4.7 %)** |
| Shared internal helper names | **0** (`kaldiio` has 15, omniio has 32) |
| Identical comments or docstrings | **0** (of 148 and 158) |
| Code lines | 1847 vs 982 — the same job in about half |

Zero overlap in internal names is the load-bearing number. Copied code keeps
its private helper names; there would be no reason to rename all 32.

The 46 identical lines fall into four groups, none of which is protectable
expression:

1. **Public API signatures**, such as the parameter list of `save_ark`. These
   are matched deliberately — a drop-in replacement that changed them would not
   be one. The longest identical run, 11 lines, is exactly that signature.
2. **Kaldi's own constants**, `kAutomaticMethod` through `kOneByteZeroOne`,
   taken by both projects from the same Apache-2.0 upstream in its own naming.
3. **Formulas transcribed from Kaldi**, such as
   `max_value = min_value + (1.0 + abs(min_value))` and `p75 = p25 + 1`.
4. **Python boilerplate**: `def close(self):`, `c = fd.read(1)`, and the like.

Byte-identical *output* is not evidence of copying — it is the requirement.
The format is fixed by Kaldi, so any correct implementation produces the same
bytes. Where omniio deliberately differs from `kaldiio` it follows Kaldi
instead: `kSpeechFeature` compression of matrices with fewer than five rows,
and clipping rather than wrapping in the fixed-range compression methods. Both
are documented in `compression.py`.

## One disclosure

`kaldiio`'s module filenames were visible in stack traces while its behaviour
was being probed. This module also has a `matio.py` and a `highlevel.py`, and
whether those names were arrived at independently cannot be claimed either way.
Their contents were written from the sources above.

## Reproducing the comparison

```python
import ast, difflib, itertools, os, pathlib, re, kaldiio

def code_lines(directory):
    """Code only: no docstrings, comments or blanks."""
    for p in sorted(pathlib.Path(directory).glob("*.py")):
        src = re.sub(r'("""|\'\'\')(?:.|\n)*?\1', "", p.read_text())
        for line in src.splitlines():
            line = re.sub(r"#.*$", "", line).strip()
            if line:
                yield line

a = list(code_lines(os.path.dirname(kaldiio.__file__)))
b = list(code_lines("omniio/tools/kaldi"))
print("similarity:", difflib.SequenceMatcher(None, a, b).ratio())
```
