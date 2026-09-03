"""Codec for Kaldi's ``CompressedMatrix``.

The on-disk layout and the quantisation formulas implemented here follow the
description in Kaldi's ``src/matrix/compressed-matrix.{h,cc}`` (Apache-2.0).

After the ``CM`` / ``CM2`` / ``CM3`` token the payload is::

    float32  min_value
    float32  range
    int32    num_rows
    int32    num_cols
    <data>

``CM``  (format 1) four ``uint16`` percentiles per column, then the
        column-major ``uint8`` payload.
``CM2`` (format 2) row-major ``uint16`` payload.
``CM3`` (format 3) row-major ``uint8`` payload.
"""

import numpy as np

# Kaldi ``CompressionMethod``.
kAutomaticMethod = 1
kSpeechFeature = 2
kTwoByteAuto = 3
kTwoByteSignedInteger = 4
kOneByteAuto = 5
kOneByteUnsignedInteger = 6
kOneByteZeroOne = 7

# Kaldi ``DataFormat`` -> ark token.
FORMAT_TO_TOKEN = {1: "CM", 2: "CM2", 3: "CM3"}
TOKEN_TO_FORMAT = {v: k for k, v in FORMAT_TO_TOKEN.items()}

_GLOBAL_HEADER_SIZE = 16


def _dt(endian, code):
    return np.dtype(endian + code)


def _float_to_uint16(min_value, range_, values):
    f = (values.astype(np.float32) - min_value) / range_
    np.clip(f, 0.0, 1.0, out=f)
    # Kaldi rounds with ``static_cast<int>(f * 65535 + 0.499)``.
    return (f * 65535 + 0.499).astype(np.int64).astype(np.uint16)


def _float_to_uint8(min_value, range_, values):
    f = (values.astype(np.float32) - min_value) / range_
    np.clip(f, 0.0, 1.0, out=f)
    return (f * 255 + 0.499).astype(np.int64).astype(np.uint8)


def _uint16_to_float(min_value, range_, values):
    # The grouping matters: everything is float32 and the division comes last.
    return np.float32(min_value) + (np.float32(range_) * values.astype(np.float32)) / np.float32(
        65535
    )


def _uint8_to_float(min_value, range_, values):
    return np.float32(min_value) + (np.float32(range_) * values.astype(np.float32)) / np.float32(
        255
    )


def _global_header(matrix, method):
    """Return ``(format, min_value, range)`` for ``matrix`` under ``method``."""
    if method == kAutomaticMethod:
        # Below nine rows the per-column headers do not pay for themselves.
        method = kSpeechFeature if matrix.shape[0] > 8 else kTwoByteAuto

    if method in (kSpeechFeature, kTwoByteAuto, kOneByteAuto):
        min_value = float(matrix.min())
        max_value = float(matrix.max())
        # min()/max() propagate NaN and inf into the global header, from which
        # every element decodes to NaN -- one bad frame would silently destroy
        # the whole matrix. Kaldi asserts here; so do we, but with a message.
        if not (np.isfinite(min_value) and np.isfinite(max_value)):
            raise ValueError(
                "Cannot compress a matrix containing NaN or inf: the value "
                "range is stored once for the whole matrix, so a single "
                "non-finite element makes every element decode to NaN."
            )
        if max_value == min_value:
            max_value = min_value + (1.0 + abs(min_value))
        range_ = max_value - min_value
        if range_ <= 0.0:
            range_ = 1.0
        fmt = {kSpeechFeature: 1, kTwoByteAuto: 2, kOneByteAuto: 3}[method]
        return fmt, min_value, range_
    if method == kTwoByteSignedInteger:
        return 2, -32768.0, 65535.0
    if method == kOneByteUnsignedInteger:
        return 3, 0.0, 255.0
    if method == kOneByteZeroOne:
        return 3, 0.0, 1.0
    raise ValueError("compression_method must be an integer in 1..7, got {}".format(method))


def _col_headers(matrix, min_value, range_):
    """Per-column ``(p0, p25, p75, p100)`` percentiles as ``uint16``.

    Follows Kaldi's ``ComputeColHeader``: the four values are the 0th, 25th,
    75th and 100th percentile of the column, kept strictly increasing, with
    headroom reserved at the top of the ``uint16`` range for the ``+1`` bumps.

    .. note::
       For columns shorter than five rows Kaldi walks upwards from the
       previous percentile.  ``kaldiio`` instead pads the sorted column with
       ``+1.0`` steps and lets the ``uint16`` conversion wrap around, so the
       two disagree bit-for-bit on such matrices.  Both are valid archives and
       decode to the same values within quantisation error; real feature
       matrices have far more than four frames.
    """
    num_rows = matrix.shape[0]
    srt = np.sort(matrix, axis=0)
    if num_rows >= 5:
        quarter = num_rows // 4
        idx = (0, quarter, 3 * quarter, num_rows - 1)
    else:
        idx = (0, 1, 2, 3)

    def u16(i):
        return _float_to_uint16(min_value, range_, srt[i, :]).astype(np.int64)

    p0 = np.minimum(u16(idx[0]), 65532)
    if num_rows > 1:
        p25 = np.minimum(np.maximum(u16(idx[1]), p0 + 1), 65533)
    else:
        p25 = p0 + 1
    if num_rows > 2:
        p75 = np.minimum(np.maximum(u16(idx[2]), p25 + 1), 65534)
    else:
        p75 = p25 + 1
    if num_rows > 3:
        p100 = np.maximum(u16(idx[3]), p75 + 1)
    else:
        p100 = p75 + 1
    return np.stack([p0, p25, p75, p100], axis=1).astype(np.uint16)


def _float_to_char(p0, p25, p75, p100, values):
    """Vectorised form of Kaldi's ``FloatToChar`` (per column)."""
    out = np.empty(values.shape, dtype=np.int64)

    low = values < p25
    mid = (~low) & (values < p75)
    high = ~(low | mid)

    with np.errstate(divide="ignore", invalid="ignore"):
        f_low = (values - p0) / (p25 - p0)
        f_mid = (values - p25) / (p75 - p25)
        f_high = (values - p75) / (p100 - p75)

    out[low] = np.clip((f_low[low] * 64 + 0.5).astype(np.int64), 0, 64)
    out[mid] = np.clip(64 + (f_mid[mid] * 128 + 0.5).astype(np.int64), 64, 192)
    out[high] = np.clip(192 + (f_high[high] * 63 + 0.5).astype(np.int64), 192, 255)
    return out.astype(np.uint8)


def _char_to_float(p0, p25, p75, p100, values):
    """Vectorised form of Kaldi's ``CharToFloat`` (per column)."""
    v = values.astype(np.float32)
    out = np.empty(v.shape, dtype=np.float32)

    low = values <= 64
    mid = (~low) & (values <= 192)
    high = ~(low | mid)

    out[low] = (p0 + (p25 - p0) * v * np.float32(1 / 64.0))[low]
    out[mid] = (p25 + (p75 - p25) * (v - 64) * np.float32(1 / 128.0))[mid]
    out[high] = (p75 + (p100 - p75) * (v - 192) * np.float32(1 / 63.0))[high]
    return out


def compress(matrix, method=kAutomaticMethod, endian="<"):
    """Compress ``matrix`` and return ``(token, payload_bytes)``."""
    matrix = np.ascontiguousarray(matrix, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(
            "Only 2-dim matrices can be written as a Kaldi CompressedMatrix, "
            "got ndim={}".format(matrix.ndim)
        )
    if matrix.size == 0:
        raise ValueError("Refusing to compress an empty matrix")

    fmt, min_value, range_ = _global_header(matrix, method)
    num_rows, num_cols = matrix.shape

    header = np.array(
        (min_value, range_, num_rows, num_cols),
        dtype=np.dtype(
            [
                ("min_value", endian + "f4"),
                ("range", endian + "f4"),
                ("num_rows", endian + "i4"),
                ("num_cols", endian + "i4"),
            ]
        ),
    ).tobytes()

    # ``min_value``/``range`` are stored as float32; quantise against the
    # rounded values so that a decoder reproduces exactly what we intended.
    min_value = float(np.float32(min_value))
    range_ = float(np.float32(range_))

    if fmt == 1:
        cols = _col_headers(matrix, min_value, range_)
        p = _uint16_to_float(min_value, range_, cols.astype(np.uint16))
        data = np.empty((num_cols, num_rows), dtype=np.uint8)
        for c in range(num_cols):
            data[c] = _float_to_char(p[c, 0], p[c, 1], p[c, 2], p[c, 3], matrix[:, c])
        body = cols.astype(_dt(endian, "u2")).tobytes() + data.tobytes()
    elif fmt == 2:
        body = _float_to_uint16(min_value, range_, matrix).astype(_dt(endian, "u2")).tobytes()
    else:
        body = _float_to_uint8(min_value, range_, matrix).tobytes()

    return FORMAT_TO_TOKEN[fmt], header + body


def data_size(fmt, num_rows, num_cols):
    """Byte size of the payload that follows the global header."""
    if fmt == 1:
        return num_cols * (8 + num_rows)
    if fmt == 2:
        return 2 * num_rows * num_cols
    return num_rows * num_cols


def decompress(payload, token, endian="<"):
    """Inverse of :func:`compress`; ``payload`` starts at the global header."""
    fmt = TOKEN_TO_FORMAT[token]
    head = np.frombuffer(
        payload,
        dtype=np.dtype(
            [
                ("min_value", endian + "f4"),
                ("range", endian + "f4"),
                ("num_rows", endian + "i4"),
                ("num_cols", endian + "i4"),
            ]
        ),
        count=1,
    )[0]
    min_value = float(head["min_value"])
    range_ = float(head["range"])
    num_rows = int(head["num_rows"])
    num_cols = int(head["num_cols"])

    body = payload[_GLOBAL_HEADER_SIZE:]
    if fmt == 1:
        n = num_cols * 4
        cols = np.frombuffer(body, dtype=_dt(endian, "u2"), count=n).reshape(num_cols, 4)
        data = np.frombuffer(body, dtype=np.uint8, count=num_cols * num_rows, offset=n * 2).reshape(
            num_cols, num_rows
        )
        p = _uint16_to_float(min_value, range_, cols)
        out = np.empty((num_cols, num_rows), dtype=np.float32)
        for c in range(num_cols):
            out[c] = _char_to_float(p[c, 0], p[c, 1], p[c, 2], p[c, 3], data[c])
        return out.T.copy()
    if fmt == 2:
        data = np.frombuffer(body, dtype=_dt(endian, "u2"), count=num_rows * num_cols).reshape(
            num_rows, num_cols
        )
        return _uint16_to_float(min_value, range_, data)
    data = np.frombuffer(body, dtype=np.uint8, count=num_rows * num_cols).reshape(
        num_rows, num_cols
    )
    return _uint8_to_float(min_value, range_, data)


def header_shape(payload, endian="<"):
    """``(num_rows, num_cols)`` read from a global header."""
    num_rows, num_cols = np.frombuffer(payload, dtype=_dt(endian, "i4"), count=2, offset=8)
    return int(num_rows), int(num_cols)
