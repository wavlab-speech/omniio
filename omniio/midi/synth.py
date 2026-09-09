"""Render a ``pretty_midi.PrettyMIDI`` to a waveform.

Two backends:

* ``fluidsynth`` — real GM instruments through a SoundFont. Needs ``pyfluidsynth`` plus
  the ``libfluidsynth`` shared library (``conda install -c conda-forge fluidsynth``).
  Unlike ``PrettyMIDI.fluidsynth`` this renders every instrument through one synthesizer
  (one MIDI channel per track, drums on channel 9) and keeps the synthesizer — with its
  loaded SoundFont — cached per process, so a large SoundFont is loaded once per worker
  instead of once per instrument per call.
* ``sine`` — ``PrettyMIDI.synthesize``: additive sine tones, no dependencies, drums are
  silent. Fine for tests and smoke checks, not for training data.

``backend="auto"`` uses fluidsynth when available and otherwise warns once and falls
back to sine.
"""
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

DEFAULT_SAMPLE_RATE = 44100
DRUM_CHANNEL = 9
MAX_CHANNELS = 256          # fluidsynth's hard limit on synth.midi-channels
SOUNDFONT_ENV = "OMNIIO_SOUNDFONT"
EFFECT_DRAIN_SEC = 2.0      # rendered and discarded after a render when effects are on
BLOCK = 64                  # fluidsynth renders (and applies events) in 64-sample blocks
DRAIN_SAMPLES = 1024        # always drained after a render, then padded to a block boundary

_synth_cache: Dict[Tuple[str, int, float, bool, bool], Tuple[Any, int, Dict[str, int]]] = {}
_warned_fallback = False


def fluidsynth_available() -> bool:
    """True when pyfluidsynth imports, i.e. the libfluidsynth shared library was found."""
    try:
        import fluidsynth
    except (ImportError, OSError):
        return False
    return fluidsynth is not None


def default_soundfont() -> str:
    """``$OMNIIO_SOUNDFONT`` if set, else the small GM SoundFont bundled with pretty_midi."""
    env = os.environ.get(SOUNDFONT_ENV)
    if env:
        return env
    import pretty_midi
    return os.path.join(os.path.dirname(pretty_midi.__file__), "TimGM6mb.sf2")


def resolve_backend(backend: str) -> str:
    global _warned_fallback
    if backend == "auto":
        if fluidsynth_available():
            return "fluidsynth"
        if not _warned_fallback:
            warnings.warn(
                "pyfluidsynth/libfluidsynth not available; MIDI synthesis falls back to "
                "additive sine tones (backend='sine'). Install fluidsynth for real "
                "instrument sounds.",
                stacklevel=3,
            )
            _warned_fallback = True
        return "sine"
    if backend not in ("fluidsynth", "sine"):
        raise ValueError(f"backend must be 'auto', 'fluidsynth' or 'sine', got {backend!r}")
    if backend == "fluidsynth" and not fluidsynth_available():
        raise ImportError(
            "backend='fluidsynth' requires pyfluidsynth and libfluidsynth "
            "(pip install pyfluidsynth; conda install -c conda-forge fluidsynth)"
        )
    return backend


def _get_synth(soundfont: str, sample_rate: int, gain: float, reverb: bool, chorus: bool):
    """Per-process cached ``fluidsynth.Synth`` with ``soundfont`` loaded."""
    import fluidsynth

    key = (os.path.abspath(soundfont), int(sample_rate), float(gain), bool(reverb),
           bool(chorus))
    hit = _synth_cache.get(key)
    if hit is not None:
        return hit
    if not os.path.exists(soundfont):
        raise FileNotFoundError(f"SoundFont not found: {soundfont}")
    settings = {"synth.reverb.active": int(bool(reverb)),
                "synth.chorus.active": int(bool(chorus))}
    synth = fluidsynth.Synth(gain=gain, samplerate=sample_rate, channels=MAX_CHANNELS,
                             **settings)
    sfid = synth.sfload(soundfont)
    if sfid < 0:
        synth.delete()
        raise RuntimeError(f"fluidsynth could not load SoundFont {soundfont}")
    entry = (synth, sfid, {"pulled": 0})
    _warm_up(*entry)
    _synth_cache[key] = entry
    return entry


def _warm_up(synth, sfid: int, state: Dict[str, int]):
    """Play every voice once and silence it again.

    A voice fluidsynth has never used renders the first block of a note's attack
    differently from one that has been used before (measured: up to ~25 % of peak on a
    percussive sample, confined to the first 64 samples, none of it audible). Without
    this, the first render in a process would differ from every later one. Touching all
    ``synth.polyphony`` voices up front puts the synthesizer in its steady state before
    the first real render.
    """
    chans = (0, 1)
    for ch in chans:
        synth.program_select(ch, sfid, 0, 0)
        for key in range(128):
            synth.noteon(ch, key, 100)
    synth.get_samples(BLOCK * 4)
    for ch in chans:
        synth.all_sounds_off(ch)
    synth.get_samples(BLOCK * 16)
    state["pulled"] += BLOCK * 20
    synth.system_reset()


def clear_synth_cache():
    for synth, _, _ in _synth_cache.values():
        try:
            synth.delete()
        except Exception:
            pass
    _synth_cache.clear()


def _assign_channels(pm) -> List[int]:
    """One channel per instrument; every drum track shares channel 9."""
    channels = []
    nxt = 0
    for inst in pm.instruments:
        if inst.is_drum:
            channels.append(DRUM_CHANNEL)
            continue
        if nxt == DRUM_CHANNEL:
            nxt += 1
        if nxt >= MAX_CHANNELS:
            warnings.warn(
                f"MIDI has more than {MAX_CHANNELS - 1} melodic tracks; wrapping channels")
            nxt = 0
        channels.append(nxt)
        nxt += 1
    return channels


def _render_fluidsynth(pm, sample_rate: int, n_samples: int, soundfont: Optional[str],
                       gain: float, reverb: bool, chorus: bool) -> np.ndarray:
    """(n_samples, 2) float32 stereo."""
    synth, sfid, state = _get_synth(soundfont or default_soundfont(), sample_rate, gain,
                                    reverb, chorus)
    synth.system_reset()

    channels = _assign_channels(pm)
    drum_program_set = False
    for inst, ch in zip(pm.instruments, channels):
        if inst.is_drum:
            if drum_program_set:
                continue
            if synth.program_select(ch, sfid, 128, inst.program) != 0:
                synth.program_select(ch, sfid, 128, 0)
            drum_program_set = True
        else:
            synth.program_select(ch, sfid, 0, inst.program)

    # (time, priority, kind, channel, a, b); note-offs first at equal times so a
    # re-struck pitch is released before it is re-triggered.
    events: List[Tuple[float, int, str, int, int, int]] = []
    for inst, ch in zip(pm.instruments, channels):
        for n in inst.notes:
            events.append((float(n.start), 1, "on", ch, int(n.pitch), int(n.velocity)))
            events.append((float(n.end), 0, "off", ch, int(n.pitch), 0))
        for b in inst.pitch_bends:
            events.append((float(b.time), 0, "bend", ch, int(b.pitch), 0))
        for cc in inst.control_changes:
            events.append((float(cc.time), 0, "cc", ch, int(cc.number), int(cc.value)))
    events.sort(key=lambda e: (e[0], e[1]))

    out = np.zeros((n_samples, 2), dtype=np.float32)
    cursor = 0

    def render_until(pos: int):
        nonlocal cursor
        pos = min(max(pos, cursor), n_samples)
        if pos > cursor:
            buf = synth.get_samples(pos - cursor)
            out[cursor:pos] = buf.reshape(-1, 2).astype(np.float32) / 32768.0
            state["pulled"] += pos - cursor
            cursor = pos

    for t, _, kind, ch, a, b in events:
        render_until(int(round(t * sample_rate)))
        if cursor >= n_samples and kind == "on":
            # Past the end of the requested window: nothing more can be heard.
            continue
        if kind == "on":
            synth.noteon(ch, a, b)
        elif kind == "off":
            synth.noteoff(ch, a)
        elif kind == "bend":
            synth.pitch_bend(ch, a)
        else:
            synth.cc(ch, a, b)
    render_until(n_samples)

    # Leave the cached synthesizer silent and block-aligned for the next call: kill every
    # voice outright (all_notes_off would only start their release), drain the partial
    # block fluidsynth is holding (and the effect delay lines if effects are on), and
    # stop on a block boundary. Events take effect at block boundaries, so starting a
    # render mid-block would shift every onset by up to BLOCK-1 samples and make the
    # output depend on what was rendered before.
    for ch in set(channels):
        synth.all_sounds_off(ch)
    drain = DRAIN_SAMPLES + (int(EFFECT_DRAIN_SEC * sample_rate) if (reverb or chorus) else 0)
    drain += (-(state["pulled"] + drain)) % BLOCK
    synth.get_samples(drain)
    state["pulled"] += drain
    return out


def _render_sine(pm, sample_rate: int, n_samples: int) -> np.ndarray:
    """(n_samples, 2) float32; both channels identical."""
    wave = pm.synthesize(fs=sample_rate)
    mono = np.zeros(n_samples, dtype=np.float32)
    n = min(n_samples, wave.shape[0])
    mono[:n] = wave[:n]
    return np.stack([mono, mono], axis=1)


def synthesize(
    pm,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = 1,
    duration: Optional[float] = None,
    tail_sec: float = 0.0,
    soundfont: Optional[str] = None,
    backend: str = "auto",
    gain: float = 0.2,
    reverb: bool = False,
    chorus: bool = False,
    normalize: bool = False,
) -> np.ndarray:
    """Render ``pm`` to a ``(frames, channels)`` float32 waveform.

    Args:
        pm:          ``pretty_midi.PrettyMIDI`` (typically already sliced, see
                     `omniio.midi.common.slice_midi`).
        sample_rate: Output rate in Hz.
        channels:    1 (stereo downmixed by averaging) or 2.
        duration:    Seconds to render. Default: the MIDI's end time, so the waveform
                     lines up sample-for-sample with a paired audio slice of the same
                     window. Notes past ``duration`` are cut, like the audio would be.
        tail_sec:    Extra seconds appended so releases of the final notes can ring out.
        soundfont:   ``.sf2`` path for the fluidsynth backend (default: ``$OMNIIO_SOUNDFONT``
                     or pretty_midi's bundled TimGM6mb).
        backend:     ``"auto"`` | ``"fluidsynth"`` | ``"sine"``.
        gain:        fluidsynth master gain (0.2 is fluidsynth's default; higher risks
                     clipping with dense polyphony).
        reverb / chorus: fluidsynth's built-in effects. Off by default so a render is a
                     pure function of the notes (and cheap); turn on for a wetter sound.
        normalize:   Peak-normalize the result to 1.0.
    """
    if channels not in (1, 2):
        raise ValueError(f"channels must be 1 or 2, got {channels}")
    sample_rate = int(sample_rate)
    total = (float(pm.get_end_time()) if duration is None else float(duration)) + float(tail_sec)
    n_samples = max(int(round(total * sample_rate)), 0)

    backend = resolve_backend(backend)
    if n_samples == 0 or not any(len(i.notes) for i in pm.instruments):
        stereo = np.zeros((n_samples, 2), dtype=np.float32)
    elif backend == "fluidsynth":
        stereo = _render_fluidsynth(pm, sample_rate, n_samples, soundfont, gain,
                                    reverb, chorus)
    else:
        stereo = _render_sine(pm, sample_rate, n_samples)

    audio = stereo if channels == 2 else stereo.mean(axis=1, keepdims=True)
    if normalize:
        peak = float(np.abs(audio).max()) if audio.size else 0.0
        if peak > 0:
            audio = audio / peak
    return np.ascontiguousarray(audio, dtype=np.float32)
