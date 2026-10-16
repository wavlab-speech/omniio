"""Pack a lhotse corpus into omniio archives.

A corpus becomes two ordinary :class:`~omniio.blob.blob.Blob` archives that
share ``recording_id``::

    out_dir/
      audio/                one entry per physical audio source (file, URL, ...)
      supervisions_<name>/  one entry per supervision segment

Audio is stored once per *source*, not per recording: AMI's headset and array
microphones are separate mono WAVs, and keeping them separate lets a reader
fetch one channel without decoding the others (FLAC also stops at 8 channels,
and MDM has 12).  File sources are byte-copied unless a ``target_format`` is
asked for.  Several conversions into the same ``out_dir`` -- AMI's IHM, SDM and
MDM, say -- share the stored audio: a source already in ``audio/`` is not
written again.

Supervisions refer to the audio entries they cover by id (``audio_ids``), not
by lhotse channel index, because the same index names different microphones in
different recording sets (channel 0 is ``Headset-0`` in AMI IHM and
``Array1-01`` in MDM, under the same recording id).  Index columns -- times,
speaker, language, gender -- go into ``metadata.parquet`` so they can be
filtered without reading the archive; text, alignments and ``custom`` are the
entry's payload, zstd-compressed JSON written through the text modality.

Cuts are not stored.  A recording-level cut is the recording, a per-segment cut
is a supervision, and fixed windows are a rule over the two (see
:func:`omniio.lhotse.windows`).  Cuts given as input are reduced to their
recordings and supervisions.
"""

import dataclasses
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from omniio.blob.blob import Blob

# Non-file sources are decoded in memory before writing; this many at a time.
_DECODE_BATCH = 16


def load_lhotse():
    try:
        import lhotse
    except ImportError as e:  # pragma: no cover - exercised only without the dependency
        raise ImportError("lhotse conversion requires lhotse: pip install 'omniio[lhotse]'") from e
    return lhotse


def _manifest(obj, cls_name: str):
    """A lhotse manifest object, given the object itself or a path to one."""
    if obj is None:
        return None
    if isinstance(obj, (str, Path)):
        return getattr(load_lhotse(), cls_name).from_file(obj)
    return obj


def _source_id(rec, src) -> str:
    """Archive id of one audio source: the file stem when it has a name."""
    if src.type in ("file", "url"):
        stem = Path(str(src.source).split("?")[0]).stem
        if stem:
            return stem
    return f"{rec.id}_ch{'-'.join(str(c) for c in src.channels)}"


def _shift_alignment(sup, cut_start: float, cut_relative_start: float):
    """Alignment of a supervision taken from a cut, in recording time.

    lhotse keeps a cut's supervision times relative to the cut, but alignments
    are not always moved with them: AMI's 30 s cuts pair a cut-relative
    ``start`` with absolute word times.  An alignment that falls inside the
    supervision's cut-relative span is shifted; one that does not is taken to be
    in recording time already.
    """
    if not sup.alignment:
        return sup.alignment
    items = [it for tier in sup.alignment.values() for it in tier]
    if not items:
        return sup.alignment
    lo, hi = cut_relative_start - 0.5, cut_relative_start + sup.duration + 0.5
    if all(lo <= it.start <= hi for it in items):
        return {k: [it.with_offset(cut_start) for it in tier] for k, tier in sup.alignment.items()}
    return sup.alignment


def _from_cuts(cuts) -> Tuple[Dict[str, object], Dict[str, object]]:
    """Recordings and recording-time supervisions found in a CutSet."""
    recordings: Dict[str, object] = {}
    sups: Dict[str, object] = {}
    skipped: Dict[str, int] = {}
    dropped = set()
    for cut in cuts:
        rec = getattr(cut, "recording", None)
        if rec is None:  # MixedCut, PaddingCut, feature-only cuts
            name = type(cut).__name__
            skipped[name] = skipped.get(name, 0) + 1
            continue
        if getattr(cut, "has_features", False):
            dropped.add("features")
        if getattr(cut, "custom", None):
            dropped.add("cut custom fields")
        recordings.setdefault(rec.id, rec)
        for sup in cut.supervisions:
            if sup.id in sups:
                continue
            # with_offset moves start only; the alignment needs its own decision
            sups[sup.id] = dataclasses.replace(
                sup.with_offset(cut.start), alignment=_shift_alignment(sup, cut.start, sup.start)
            )
    if skipped:
        warnings.warn(
            "Skipped cuts without a single recording (not representable as recordings + "
            f"supervisions): {skipped}"
        )
    if dropped:
        warnings.warn(f"Cut-level data is not stored: {sorted(dropped)}")
    return recordings, sups


def _audio_ids_for(
    sup, rec, source_ids: List[Tuple[List[int], str]]
) -> Tuple[List[str], List[int]]:
    """Audio entries holding the channels a supervision covers, by channel, and
    the first lhotse channel of each.

    The channel numbers are what fixes the order of a variant's sources -- an
    array's geometry -- since the audio archive's own order is write order,
    shared between variants.
    """
    if sup.channel is None:
        wanted = set(rec.channel_ids)
    elif isinstance(sup.channel, int):
        wanted = {sup.channel}
    else:
        wanted = set(sup.channel)
    hits = sorted((min(chans), aid) for chans, aid in source_ids if wanted.intersection(chans))
    return [aid for _, aid in hits], [ch for ch, _ in hits]


def _payload(sup) -> str:
    alignment = None
    if sup.alignment:
        alignment = {
            tier: [
                [it.symbol, it.start, it.duration] + ([] if it.score is None else [it.score])
                for it in items
            ]
            for tier, items in sup.alignment.items()
        }
    return json.dumps(
        {"text": sup.text, "alignment": alignment, "custom": sup.custom},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def from_lhotse(
    out_dir: Union[str, Path],
    recordings=None,
    supervisions=None,
    cuts=None,
    *,
    name: str = "default",
    target_format: Optional[str] = None,
    target_bit_depth: Optional[int] = None,
    num_workers: int = 0,
    max_bin_size: int = 32 * 1024 * 1024 * 10,
    compression_level: int = 3,
    progress: bool = True,
) -> Tuple[Blob, Blob]:
    """Pack lhotse manifests into ``out_dir/audio`` and ``out_dir/supervisions_<name>``.

    Args:
        out_dir:          Directory holding the archives; created if missing.
        recordings:       ``RecordingSet`` or a path to one.
        supervisions:     ``SupervisionSet`` or a path to one.
        cuts:             ``CutSet`` or a path to one.  Its recordings and
                          supervisions (moved to recording time and
                          de-duplicated by id) are added to the above; a
                          supervision given in ``supervisions`` wins.
        name:             Suffix of the supervisions archive, e.g. ``"ihm_dev"``,
                          so several variants can share one ``audio/``.
        target_format:    ``None`` byte-copies file sources (and writes FLAC for
                          sources that have to be decoded); ``"flac"``,
                          ``"wav"`` or ``"webm"`` re-encodes everything.
        target_bit_depth: Forwarded to :func:`omniio.audio.write.audio_write`.
        num_workers:      Worker processes for the audio archive.  Supervisions
                          are always written in one process, in
                          ``(recording_id, start)`` order, so that the segments of
                          a time window sit next to each other in the bin.
        max_bin_size:     Forwarded to both :class:`Blob` archives.
        compression_level: zstd level of the supervision payloads.
        progress:         Show progress bars.

    Returns:
        ``(audio_blob, supervisions_blob)``.
    """
    recs = _manifest(recordings, "RecordingSet")
    sups = _manifest(supervisions, "SupervisionSet")
    cut_set = _manifest(cuts, "CutSet")
    if recs is None and cut_set is None:
        raise ValueError("Give recordings or cuts: there is no audio to convert.")

    rec_by_id: Dict[str, object] = {}
    sup_by_id: Dict[str, object] = {}
    if cut_set is not None:
        rec_by_id, sup_by_id = _from_cuts(cut_set)
    if recs is not None:
        rec_by_id.update((r.id, r) for r in recs)
    if sups is not None:
        sup_by_id.update((s.id, s) for s in sups)

    out_dir = Path(out_dir)
    audio = Blob(str(out_dir / "audio"), modality="audio", max_bin_size=max_bin_size)
    sup_blob = Blob(
        str(out_dir / f"supervisions_{name}"), modality="text", max_bin_size=max_bin_size
    )

    # --- Audio: one entry per source -------------------------------------
    known: Dict[str, str] = {}
    if len(audio):
        meta = audio.get_metadata()
        srcs = meta.column("source").to_pylist() if "source" in meta.column_names else None
        for i, aid in enumerate(meta.column("id").to_pylist()):
            known[aid] = srcs[i] if srcs else None

    by_path, by_array = [], []  # (item, id, extra) / (rec, src, id, extra)
    source_ids: Dict[str, List[Tuple[List[int], str]]] = {}
    for rec in rec_by_id.values():
        source_ids[rec.id] = []
        for src in rec.sources:
            aid = _source_id(rec, src)
            origin = str(src.source) if src.type in ("file", "url") else None
            source_ids[rec.id].append((list(src.channels), aid))
            if aid in known:
                if known[aid] is not None and origin is not None and known[aid] != origin:
                    raise ValueError(
                        f"Two different audio sources map to the archive id {aid!r}: "
                        f"{known[aid]} and {origin}"
                    )
                continue
            known[aid] = origin
            extra = {"recording_id": rec.id, "source": origin, "source_type": src.type}
            if src.type == "file" and not rec.transforms:
                by_path.append((src.source, aid, extra))
            else:
                by_array.append((rec, src, aid, extra))

    fmt_kwargs = {"target_format": target_format, "target_bit_depth": target_bit_depth}
    if by_path:
        items, ids, extras = (list(x) for x in zip(*by_path))
        audio.append(
            items, ids, num_workers=num_workers, progress=progress, metadata=extras, **fmt_kwargs
        )
    for i in range(0, len(by_array), _DECODE_BATCH):
        batch = by_array[i : i + _DECODE_BATCH]
        items = [
            (rec.load_audio(channels=src.channels).T, rec.sampling_rate) for rec, src, _, _ in batch
        ]
        audio.append(
            items,
            [aid for _, _, aid, _ in batch],
            num_workers=num_workers,
            progress=progress,
            metadata=[extra for _, _, _, extra in batch],
            **fmt_kwargs,
        )

    # --- Supervisions: one entry per segment, time-ordered ---------------
    existing = set(sup_blob.get_metadata().column("id").to_pylist()) if len(sup_blob) else set()
    ordered = sorted(sup_by_id.values(), key=lambda s: (s.recording_id, s.start, s.id))
    items, ids, extras, missing = [], [], [], set()
    for sup in ordered:
        if sup.id in existing:
            continue
        rec = rec_by_id.get(sup.recording_id)
        if rec is None:
            missing.add(sup.recording_id)
            continue
        items.append(_payload(sup))
        ids.append(sup.id)
        channel = sup.channel
        audio_ids, audio_channels = _audio_ids_for(sup, rec, source_ids[rec.id])
        extras.append(
            {
                "recording_id": sup.recording_id,
                "start": float(sup.start),
                "duration": float(sup.duration),
                "channel": (
                    None
                    if channel is None
                    else ([channel] if isinstance(channel, int) else list(channel))
                ),
                "audio_ids": audio_ids,
                "audio_channels": audio_channels,
                "speaker": sup.speaker,
                "language": sup.language,
                "gender": sup.gender,
            }
        )
    if missing:
        warnings.warn(
            f"Skipped supervisions of {len(missing)} recording(s) not in the input, "
            f"e.g. {sorted(missing)[:3]}"
        )
    if items:
        sup_blob.append(
            items,
            ids,
            num_workers=0,
            progress=progress,
            metadata=extras,
            is_path=False,
            compression_level=compression_level,
        )
    return audio, sup_blob
