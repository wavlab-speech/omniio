#!/usr/bin/env python3
"""Read every Kaldi archive under some roots with both omniio and kaldiio, and
compare what comes back.

    python compare_with_kaldiio.py /path/to/dump [more roots...]

Reports by on-disk format, so the summary says what was actually exercised
rather than just how many files were opened -- a run that touched only
``FM`` matrices tells you nothing about compressed or piped audio.

Read-only: nothing outside ``--out`` is written, and ``scp`` entries that are
shell pipes are skipped unless ``--allow-pipes`` is given, because reading one
means running the command in it.

kaldiio is the reference and is needed only to run this script. It is not a
dependency of omniio, and omniio's own implementation was written without
reading its source; see ``omniio/tools/kaldi/PROVENANCE.md``.

    pip install omniio kaldiio
"""

import argparse
import collections
import importlib.metadata as md
import os
import random
import re
import shlex
import subprocess
import sys

import numpy as np

TOKENS = {
    b"\x00BCM ": "CM (compressed, per-column)",
    b"\x00BCM2": "CM2 (compressed, 2-byte)",
    b"\x00BCM3": "CM3 (compressed, 1-byte)",
    b"\x00BFM ": "FM (float32 matrix)",
    b"\x00BDM ": "DM (float64 matrix)",
    b"\x00BFV ": "FV (float32 vector)",
    b"\x00BDV ": "DV (float64 vector)",
}

#: Programs a pipe entry may invoke under --allow-pipes. Everything a Kaldi
#: recipe writes into a wav.scp is on this list; anything else is skipped
#: rather than run, so an unexpected command shape cannot execute by accident.
ALLOWED_PROGRAMS = {"sox", "sph2pipe", "ffmpeg", "flac", "shorten", "wav-copy", "cat", "gunzip"}

#: Shell metacharacters that would make the entry more than a plain pipeline.
FORBIDDEN = set(";&$`><*?()[]{}\n")


def classify(fd, offset):
    """Name the record format at ``offset``, from its first bytes."""
    try:
        fd.seek(offset or 0)
        head = fd.read(8)
    except OSError:
        return "unreadable"
    if head[:4] == b"RIFF":
        return "WaveHolder (RIFF)"
    if head[:5] == b"AUDIO":
        return "AUDIO (extended)"
    if head[:2] == b"\x00B":
        for prefix, name in TOKENS.items():
            if head.startswith(prefix):
                return name
        if not head[2:3].isalpha():
            return "int32 vector (alignment)"
        return "binary, other: {!r}".format(head[2:6])
    return "text"


def entries(scp, allow_pipes):
    """Yield ``(key, extended_filename)``, skipping pipes unless asked."""
    with open(scp, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            key, name = parts
            if name.rstrip().endswith("|") and not allow_pipes:
                continue
            yield key, name


def resolves(name):
    """Does this scp value point at a file we could actually read?"""
    path, _, tail = name.rpartition(":")
    if not (tail.isdigit() and path):
        path = name
    return os.path.exists(path)


def is_archive_scp(rows, probe=10):
    """True if any of the first few entries names something that exists.

    ESPnet uses the ``.scp`` suffix for shape files (``utt 288,83``), lexicons
    (``ヴぉ v_o``) and other two-column text. Opening those as archives
    produced ten thousand FileNotFoundErrors on the first run, which buried the
    result rather than being it.

    A pipe entry counts as evidence on its own -- there is no file to stat, and
    a wav.scp made entirely of pipes is exactly the speed-perturbed case this
    is most worth running on.
    """
    for _, name in rows[:probe]:
        if name.rstrip().endswith("|") or resolves(name):
            return True
    return False


def pipeline_is_safe(command):
    """True if every stage is an allowed program and there is no shell trickery."""
    if FORBIDDEN & set(command):
        return False
    try:
        shlex.split(command)
    except ValueError:
        return False
    heads = [command.split()[0]] + re.findall(r"\|\s*(\S+)", command)
    return all(os.path.basename(h) in ALLOWED_PROGRAMS for h in heads)


def same(a, b):
    if isinstance(a, tuple) != isinstance(b, tuple):
        return False
    if isinstance(a, tuple):
        if a[0] != b[0]:
            return False
        a, b = a[1], b[1]
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)


def describe(v):
    if isinstance(v, tuple):
        return "(rate={}, {} {})".format(v[0], np.asarray(v[1]).shape, np.asarray(v[1]).dtype)
    v = np.asarray(v)
    return "{} {}".format(v.shape, v.dtype)


def oneline(text):
    """Keep a record on one line, so the TSV can be counted with wc -l."""
    return " ".join(str(text).split())


def read_both(kaldiio, om, name, scratch):
    """Return ``(kaldiio_result, omniio_result)``, each a value or an Exception.

    A pipe is run **once** and the captured bytes are handed to both. Running
    the command twice would differ on the samples no matter what the libraries
    do: a ``sox`` stage without ``-R`` dithers non-repeatably.
    """
    target = name
    if name.rstrip().endswith("|"):
        command = name.rstrip()[:-1].strip()
        blob = subprocess.run(command, shell=True, stdout=subprocess.PIPE, check=True).stdout
        with open(scratch, "wb") as fh:
            fh.write(blob)
        target = scratch

    out = []
    for mod in (kaldiio, om):
        try:
            out.append(mod.load_mat(target))
        except Exception as exc:  # noqa: BLE001 - failures are a result here
            out.append(exc)
    return out[0], out[1]


def verdict(a, b):
    """Classify one comparison as ``(kind, detail)``."""
    a_failed, b_failed = isinstance(a, Exception), isinstance(b, Exception)
    if a_failed and b_failed:
        # Not an archive at all, most likely. Both refusing is agreement, not a
        # divergence -- the exception types differ by design, but omniio's
        # ReadError subclasses RuntimeError so `except RuntimeError` catches
        # either.
        return "both-reject", "kaldiio {}: {} | omniio {}: {}".format(
            type(a).__name__, oneline(a), type(b).__name__, oneline(b)
        )
    if a_failed:
        return "MISMATCH", "kaldiio raised {}: {} but omniio returned {}".format(
            type(a).__name__, oneline(a), describe(b)
        )
    if b_failed:
        return "MISMATCH", "omniio raised {}: {} but kaldiio returned {}".format(
            type(b).__name__, oneline(b), describe(a)
        )
    if not same(a, b):
        return "MISMATCH", "kaldiio {} vs omniio {}".format(describe(a), describe(b))
    return "ok", ""


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("roots", nargs="+", help="directories to walk")
    p.add_argument(
        "--max-per-file",
        type=int,
        default=25,
        help="entries sampled per scp; 0 for all (default: 25)",
    )
    p.add_argument("--max-files", type=int, default=0, help="stop after N scp files")
    p.add_argument("--seed", type=int, default=0, help="sampling seed (default: 0)")
    p.add_argument(
        "--allow-pipes",
        action="store_true",
        help="also read scp entries that are shell pipes. This RUNS the command "
        "in them; only the pipelines in ALLOWED_PROGRAMS are run, the rest are "
        "counted as skipped.",
    )
    p.add_argument("--out", default="", help="write one line per problem here (TSV)")
    args = p.parse_args()

    import kaldiio

    from omniio import kaldi as om

    print("omniio {}   kaldiio {}\n".format(md.version("omniio"), kaldiio.__version__))

    scps = []
    for root in args.roots:
        for dirpath, _, names in os.walk(root):
            scps += [os.path.join(dirpath, n) for n in names if n.endswith(".scp")]
    scps.sort()
    if args.max_files:
        scps = scps[: args.max_files]
    print("{} scp files under {}\n".format(len(scps), ", ".join(args.roots)))

    rng = random.Random(args.seed)
    by_format = collections.Counter()
    bad_by_format = collections.Counter()
    skips = collections.Counter()
    checked = skipped = 0
    problems = []
    scratch = os.path.abspath(".omniio_compare_scratch")

    for i, scp in enumerate(scps, 1):
        rows = list(entries(scp, args.allow_pipes))
        if not rows:
            skips["empty, or all pipes and --allow-pipes not given"] += 1
            continue
        if not is_archive_scp(rows):
            skips["not an archive scp (shape file, lexicon, text...)"] += 1
            continue
        if args.max_per_file and len(rows) > args.max_per_file:
            rows = rng.sample(rows, args.max_per_file)

        handles = {}
        for key, name in rows:
            is_pipe = name.rstrip().endswith("|")
            if is_pipe:
                if not pipeline_is_safe(name.rstrip()[:-1].strip()):
                    skips["pipe command not in the allow-list"] += 1
                    continue
                fmt = "pipe ({})".format(os.path.basename(name.split()[0]))
            else:
                path, _, tail = name.rpartition(":")
                offset = int(tail) if tail.isdigit() else None
                if offset is None:
                    path = name
                if not os.path.exists(path):
                    skipped += 1
                    continue
                fd = handles.get(path)
                if fd is None:
                    try:
                        fd = handles[path] = open(path, "rb")
                    except OSError:
                        skipped += 1
                        continue
                fmt = classify(fd, offset)

            try:
                a, b = read_both(kaldiio, om, name, scratch)
            except subprocess.CalledProcessError as exc:
                skips["pipe command itself failed (rc={})".format(exc.returncode)] += 1
                continue

            kind, detail = verdict(a, b)
            if kind == "both-reject":
                skips["both libraries reject it (not an archive?)"] += 1
                problems.append((scp, key, fmt, "BOTH-REJECT " + detail))
                continue
            checked += 1
            by_format[fmt] += 1
            if kind == "MISMATCH":
                bad_by_format[fmt] += 1
                problems.append((scp, key, fmt, "MISMATCH " + detail))

        for fd in handles.values():
            fd.close()
        if i % 50 == 0 or i == len(scps):
            print(
                "  {}/{} scp   checked={}  mismatched={}".format(
                    i, len(scps), checked, sum(bad_by_format.values())
                )
            )

    if os.path.exists(scratch):
        os.unlink(scratch)

    print("\n=== by format (this is the part that matters) ===")
    for fmt, n in by_format.most_common():
        print("  {:<34} {:>7}   mismatched {}".format(fmt, n, bad_by_format[fmt]))
    print("\n=== skipped ===")
    for why, n in skips.most_common():
        print("  {:<34} {:>7}".format(why, n))

    mismatched = sum(bad_by_format.values())
    print("\nchecked {}   MISMATCHED {}   entries skipped {}".format(checked, mismatched, skipped))

    if problems:
        shown = [row for row in problems if row[3].startswith("MISMATCH")][:20]
        if shown:
            print("\n=== first {} mismatches ===".format(len(shown)))
            for scp, key, fmt, why in shown:
                print("  {}\n    {}  [{}]\n    {}".format(scp, key, fmt, why))
        if args.out:
            with open(args.out, "w") as fh:
                for row in problems:
                    fh.write("\t".join(oneline(c) for c in row) + "\n")
            print("\nfull list ({} lines): {}".format(len(problems), args.out))
    return 1 if mismatched else 0


if __name__ == "__main__":
    sys.exit(main())
