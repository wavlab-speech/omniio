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
| Identical non-trivial lines | **55 of 1075 (5.1 %)** |
| Shared internal helper names | **0** (`kaldiio` has 15, omniio has 33) |
| Identical comments or docstrings | **0** (of 148 and 194) |
| Code lines | 1847 vs 1075 — the same job in well under half |
| Similarity over the whole tree | **3.7 %** |

Produced by the script at the end of this file, which prints exactly these
rows; re-run it after any change here rather than trusting the numbers above.

Zero overlap in internal names is the load-bearing number. Copied code keeps
its private helper names; there would be no reason to rename all 32.

The identical lines fall into four groups, none of which is protectable
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

Run from the repository root, with `kaldiio` installed:

```python
"""Reproduce every row of the evidence table in PROVENANCE.md."""

import ast
import difflib
import itertools
import os
import pathlib
import re

import kaldiio

THEIRS = pathlib.Path(os.path.dirname(kaldiio.__file__))
OURS = pathlib.Path("omniio/tools/kaldi")
TRIVIAL = re.compile(r"^(import |from |return$|else:|try:|pass$|continue$|break$|\)|\]|\}|@)")


def code_lines(path):
    """Code only: docstrings, comments and blank lines removed."""
    src = re.sub(r'("""|\'\'\')(?:.|\n)*?\1', "", path.read_text())
    for line in src.splitlines():
        line = re.sub(r"#.*$", "", line).strip()
        if line:
            yield line


def per_file(directory):
    return {p.name: list(code_lines(p)) for p in sorted(directory.glob("*.py"))}


def helper_names(directory, private=True):
    found = set()
    for p in directory.glob("*.py"):
        for node in ast.walk(ast.parse(p.read_text())):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name.startswith("_") == private and not node.name.startswith("__"):
                    found.add(node.name)
    return found


def prose(directory):
    """Comments and docstring lines long enough to be more than a fragment."""
    out = []
    for p in directory.glob("*.py"):
        src = p.read_text()
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                doc = ast.get_docstring(node)
                if doc:
                    out += [ln.strip() for ln in doc.splitlines() if len(ln.strip()) > 25]
        out += [m.group(1).strip() for m in re.finditer(r"#\s*(.+)", src) if len(m.group(1).strip()) > 25]
    return out


theirs, ours = per_file(THEIRS), per_file(OURS)
their_all = list(itertools.chain.from_iterable(theirs.values()))
our_all = list(itertools.chain.from_iterable(ours.values()))

best = max(
    (difflib.SequenceMatcher(None, a, b).ratio(), an, bn)
    for an, a in theirs.items()
    for bn, b in ours.items()
)
identical = {
    line
    for line in set(their_all) & set(our_all)
    if not TRIVIAL.match(line) and len(line) > 12
}
their_priv, our_priv = helper_names(THEIRS), helper_names(OURS)
their_prose, our_prose = prose(THEIRS), prose(OURS)

print(f"highest file-pair similarity   {best[0]:.1%}  ({best[1]} vs {best[2]})")
print(f"identical non-trivial lines    {len(identical)} of {len(our_all)} ({len(identical)/len(our_all):.1%})")
print(f"shared internal helper names   {len(their_priv & our_priv)}  (theirs {len(their_priv)}, ours {len(our_priv)})")
print(f"identical comments/docstrings  {len(set(their_prose) & set(our_prose))}  (theirs {len(their_prose)}, ours {len(our_prose)})")
print(f"code lines                     {len(their_all)} vs {len(our_all)}")
print(f"whole-tree similarity          {difflib.SequenceMatcher(None, their_all, our_all).ratio():.1%}")
```

Output at the time of writing:

```
highest file-pair similarity   10.6%  (compression_header.py vs compression.py)
identical non-trivial lines    55 of 1075 (5.1%)
shared internal helper names   0  (theirs 15, ours 33)
identical comments/docstrings  0  (theirs 148, ours 194)
code lines                     1847 vs 1075
whole-tree similarity          3.7%
```
