"""Byte layout of a ``seq`` entry.

    header (little-endian, 16 bytes)
        b"OSEQ"   magic
        u8  version (1)
        u8  flags   bit0 HAS_N (N bitmap present)  bit1 HAS_LOWER (soft-mask bitmap present)
        u16 reserved
        u64 length  (bases)
    payload
        bases   ceil(length / 4) bytes, 2 bits per base, base i at bits (2*(i%4)) of byte i//4
                A=0 C=1 G=2 T=3 (N and other IUPAC codes stored as A=0 and flagged in the N bitmap)
        nmask   ceil(length / 8) bytes, bit i = base i is N / unknown      (if HAS_N)
        lower   ceil(length / 8) bytes, bit i = base i was lowercase       (if HAS_LOWER)

A window ``[start, end)`` needs bytes ``[start//4, ceil(end/4))`` of ``bases`` and
``[start//8, ceil(end/8))`` of each bitmap: three preads, each a quarter / an eighth of
the window length.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

MAGIC = b"OSEQ"
VERSION = 1
FLAG_N = 1
FLAG_LOWER = 2
_HEAD = struct.Struct("<4sBBHQ")
HEADER_SIZE = _HEAD.size     # 16

_CODE = np.zeros(256, dtype=np.uint8)            # byte -> 2-bit code
for ch, v in (("A", 0), ("C", 1), ("G", 2), ("T", 3), ("a", 0), ("c", 1), ("g", 2), ("t", 3), ("U", 3), ("u", 3)):
    _CODE[ord(ch)] = v
_IS_N = np.ones(256, dtype=bool)                  # anything that is not ACGTU (either case) is "N"
for ch in "ACGTUacgtu":
    _IS_N[ord(ch)] = False
_IS_LOWER = np.zeros(256, dtype=bool)
for ch in "acgtun":
    _IS_LOWER[ord(ch)] = True
_BASES = np.frombuffer(b"ACGT", dtype=np.uint8)
_COMP = np.array([3, 2, 1, 0], dtype=np.uint8)


@dataclass
class SeqHeader:
    version: int
    flags: int
    length: int

    @property
    def has_n(self) -> bool:
        return bool(self.flags & FLAG_N)

    @property
    def has_lower(self) -> bool:
        return bool(self.flags & FLAG_LOWER)

    @property
    def bases_bytes(self) -> int:
        return (self.length + 3) // 4

    @property
    def mask_bytes(self) -> int:
        return (self.length + 7) // 8

    @property
    def n_offset(self) -> int:
        return HEADER_SIZE + self.bases_bytes

    @property
    def lower_offset(self) -> int:
        return self.n_offset + (self.mask_bytes if self.has_n else 0)

    @property
    def entry_size(self) -> int:
        return self.lower_offset + (self.mask_bytes if self.has_lower else 0)


def parse_header(buf: bytes) -> SeqHeader:
    if len(buf) < HEADER_SIZE:
        raise ValueError("seq entry too short for a header")
    magic, version, flags, _r, length = _HEAD.unpack_from(buf, 0)
    if magic != MAGIC:
        raise ValueError(f"not a seq entry (magic {magic!r})")
    if version != VERSION:
        raise ValueError(f"unsupported seq entry version {version}")
    return SeqHeader(version, flags, length)


def pack_bits(mask: np.ndarray) -> bytes:
    return np.packbits(mask, bitorder="little").tobytes()


def unpack_bits(buf, n: int, bit_offset: int = 0) -> np.ndarray:
    b = np.frombuffer(buf, dtype=np.uint8)
    return np.unpackbits(b, bitorder="little")[bit_offset: bit_offset + n].astype(bool)


def encode(seq: "str | bytes | np.ndarray", keep_case: bool = True) -> tuple[bytes, SeqHeader, dict]:
    """Pack a sequence. Returns ``(raw_bytes, header, stats)`` with ``n_count`` and ``lower_count``."""
    if isinstance(seq, str):
        a = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
    elif isinstance(seq, (bytes, bytearray, memoryview)):
        a = np.frombuffer(seq, dtype=np.uint8)
    else:
        a = np.asarray(seq, dtype=np.uint8)
    n = int(a.size)
    codes = _CODE[a]
    pad = (-n) % 4
    if pad:
        codes = np.concatenate([codes, np.zeros(pad, dtype=np.uint8)])
    quads = codes.reshape(-1, 4)
    bases = (quads[:, 0] | (quads[:, 1] << 2) | (quads[:, 2] << 4) | (quads[:, 3] << 6)).astype(np.uint8).tobytes()
    nmask = _IS_N[a]
    lower = _IS_LOWER[a] if keep_case else None
    flags = 0
    parts = [bases]
    n_count = int(nmask.sum())
    if n_count:
        flags |= FLAG_N
        parts.append(pack_bits(nmask))
    lower_count = int(lower.sum()) if lower is not None else 0
    if lower_count:
        flags |= FLAG_LOWER
        parts.append(pack_bits(lower))
    head = SeqHeader(VERSION, flags, n)
    return _HEAD.pack(MAGIC, VERSION, flags, 0, n) + b"".join(parts), head, {"n_count": n_count, "lower_count": lower_count}


def decode_window(head: SeqHeader, bases_buf, start: int, end: int, nmask_buf=None, lower_buf=None,
                  reverse_complement: bool = False, soft_mask: bool = True) -> str:
    """Bases ``[start, end)`` from the byte spans a reader fetched: ``bases_buf`` =
    bytes ``[start//4, ceil(end/4))`` of the bases block, the bitmaps = bytes
    ``[start//8, ceil(end/8))`` of theirs (``None`` when absent)."""
    n = end - start
    if n <= 0:
        return ""
    b = np.frombuffer(bases_buf, dtype=np.uint8)
    codes = np.empty(b.size * 4, dtype=np.uint8)
    codes[0::4] = b & 3
    codes[1::4] = (b >> 2) & 3
    codes[2::4] = (b >> 4) & 3
    codes[3::4] = b >> 6
    off = start % 4
    codes = codes[off: off + n]
    if reverse_complement:
        codes = _COMP[codes[::-1]]
    out = _BASES[codes].copy()
    if nmask_buf is not None:
        nm = unpack_bits(nmask_buf, n, start % 8)
        if reverse_complement:
            nm = nm[::-1]
        out[nm] = ord("N")
    if soft_mask and lower_buf is not None:
        lo = unpack_bits(lower_buf, n, start % 8)
        if reverse_complement:
            lo = lo[::-1]
        out[lo] |= 0x20                       # ASCII lowercase
    return out.tobytes().decode("ascii")
