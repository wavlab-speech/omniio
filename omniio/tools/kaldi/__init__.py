"""Kaldi ``ark``/``scp`` compatibility layer.

This subpackage lets omniio stand in for ``kaldiio`` wherever a project only
needs to read and write Kaldi archives::

    from omniio import kaldi

    with kaldi.ReadHelper("scp:feats.scp") as reader:
        for utt_id, feats in reader:
            ...

The exported names, their arguments and their return types mirror ``kaldiio``
so that ``import kaldiio`` can be swapped for ``from omniio import kaldi as
kaldiio``.  The archives written here are byte-identical to Kaldi's own, with
two deliberate exceptions that are documented in :mod:`omniio.kaldi.compression`.

Unlike ``kaldiio`` this code is licensed under the MIT license, and it is
written against the format description in Kaldi itself (Apache-2.0).
"""

from omniio.tools.kaldi.compression import (
    kAutomaticMethod,
    kOneByteAuto,
    kOneByteUnsignedInteger,
    kOneByteZeroOne,
    kSpeechFeature,
    kTwoByteAuto,
    kTwoByteSignedInteger,
)
from omniio.tools.kaldi.highlevel import ReadHelper, WriteHelper
from omniio.tools.kaldi.matio import (
    LazyLoader,
    ReadError,
    SegmentedLoader,
    load_ark,
    load_mat,
    load_scp,
    load_scp_sequential,
    load_segments,
    load_wav_scp,
    read_object,
    save_ark,
    save_mat,
    slice_segment,
)
from omniio.tools.kaldi.specifier import (
    parse_rspecifier,
    parse_specifier,
    parse_wspecifier,
)
from omniio.tools.kaldi.stream import open_like_kaldi, parse_extended_filename

__all__ = [
    "LazyLoader",
    "ReadError",
    "SegmentedLoader",
    "ReadHelper",
    "WriteHelper",
    "kAutomaticMethod",
    "kOneByteAuto",
    "kOneByteUnsignedInteger",
    "kOneByteZeroOne",
    "kSpeechFeature",
    "kTwoByteAuto",
    "kTwoByteSignedInteger",
    "load_ark",
    "load_mat",
    "load_scp",
    "load_scp_sequential",
    "load_segments",
    "load_wav_scp",
    "open_like_kaldi",
    "parse_extended_filename",
    "parse_rspecifier",
    "parse_specifier",
    "parse_wspecifier",
    "read_object",
    "save_ark",
    "save_mat",
    "slice_segment",
]
