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

from omniio.kaldi.compression import (
    kAutomaticMethod,
    kOneByteAuto,
    kOneByteUnsignedInteger,
    kOneByteZeroOne,
    kSpeechFeature,
    kTwoByteAuto,
    kTwoByteSignedInteger,
)
from omniio.kaldi.highlevel import (
    ReadHelper,
    WriteHelper,
    load_segments,
)
from omniio.kaldi.matio import (
    LazyLoader,
    ReadError,
    load_ark,
    load_mat,
    load_scp,
    load_scp_sequential,
    read_object,
    save_ark,
    save_mat,
)
from omniio.kaldi.specifier import parse_rspecifier, parse_wspecifier
from omniio.kaldi.stream import open_like_kaldi, parse_extended_filename

__all__ = [
    "LazyLoader",
    "ReadError",
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
    "open_like_kaldi",
    "parse_extended_filename",
    "parse_rspecifier",
    "parse_wspecifier",
    "read_object",
    "save_ark",
    "save_mat",
]
