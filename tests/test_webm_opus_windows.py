"""Windowed (partial) reads of WebM/Opus.

WebM can only be seeked to a cluster boundary, so the seek lands at or *before* the
requested start. The reader used to keep the first `end - start` samples of whatever the
seek produced, which silently returned audio from the wrong part of the file (a request
for seconds 5-7 could return seconds 0-2). These tests pin the window to the timeline:
every window must be the same samples the full decode has at that offset.
"""

import io
from fractions import Fraction
from types import SimpleNamespace

import av
import numpy as np
import pytest

from omniio.audio.read import (
    PyAVCompatibilityWarning,
    _TESTED_LIBAVCODEC,
    _TESTED_PYAV,
    _gather_window,
    pyav_compatibility,
    _is_opus_frame_size,
    _read_webm,
    _snap_to_grid,
    audio_read_local,
    audio_read_remote,
)

SR = 48000          # Opus always decodes at 48 kHz
LADDER_SECONDS = 10


def tone_ladder(seconds=LADDER_SECONDS, rate=SR, offset=0):
    """Second k holds a steady tone at 500*(k+1+offset) Hz, so the content of any window
    names the part of the file it came from."""
    return np.concatenate([
        0.5 * np.sin(2 * np.pi * (500 * (k + 1 + offset)) * np.arange(rate) / rate)
        for k in range(seconds)
    ]).astype(np.float32)


def encode_webm(signal, rate=SR, mux="single", options=None):
    """Encode (samples,) or (samples, channels) float audio to WebM/Opus bytes.

    `mux` picks how the frames reach the muxer, which is what decides where the clusters
    (the only points a seek can land on) end up: "single" hands the encoder the whole
    signal at once, "chunked" feeds it 0.5 s at a time.
    """
    if signal.ndim == 1:
        signal = signal[:, None]
    channels = signal.shape[1]
    layout = {1: "mono", 2: "stereo"}[channels]
    pcm = (np.clip(signal, -1, 1) * 32767).astype(np.int16)

    buf = io.BytesIO()
    with av.open(buf, mode="w", format="webm", options=options or {}) as container:
        stream = container.add_stream("libopus", rate=rate)
        stream.layout = layout

        def send(block):
            frame = av.AudioFrame.from_ndarray(
                np.ascontiguousarray(block.reshape(1, -1)), format="s16", layout=layout
            )
            frame.rate = rate
            for packet in stream.encode(frame):
                container.mux(packet)

        if mux == "single":
            send(pcm)
        else:
            step = rate // 2
            for i in range(0, pcm.shape[0], step):
                send(pcm[i:i + step])

        for packet in stream.encode(None):
            container.mux(packet)
    return buf.getvalue()


def dominant_hz(x, rate=SR):
    """Loudest frequency in a mono signal."""
    n = min(x.size, 8192)
    w = x[:n] * np.hanning(n)
    return float(np.fft.rfftfreq(n, 1 / rate)[np.argmax(np.abs(np.fft.rfft(w)))])


def best_lag(a, b, max_lag=32):
    """Sample offset at which `a` best matches `b` (0 == correctly aligned).

    Normalised, over a region trimmed at both ends so every lag is scored on the same
    number of samples — an unnormalised score lets a window with a louder tail win at the
    wrong offset.
    """
    n = min(a.size, b.size)
    if n <= 4 * max_lag:
        return 0
    a, b = a[:n] - a[:n].mean(), b[:n] - b[:n].mean()
    ref = b[max_lag:n - max_lag]
    ref_norm = np.linalg.norm(ref) + 1e-12

    def score(lag):
        seg = a[max_lag + lag:n - max_lag + lag]
        return float(np.dot(seg, ref) / ((np.linalg.norm(seg) + 1e-12) * ref_norm))

    lags = range(-max_lag, max_lag + 1)
    return int(max(lags, key=score))


def rel_rms(a, b):
    """RMS difference between two windows, relative to the reference level."""
    n = min(a.size, b.size)
    return float(
        np.sqrt(np.mean((a[:n] - b[:n]) ** 2)) / (np.sqrt(np.mean(b[:n] ** 2)) + 1e-12)
    )


def frames(*spec):
    """Fake decoded frames — (first sample index, (channels, n) array) — for the unit
    tests of the trimming math. Each array is filled with its own absolute indices."""
    return [
        (pos, np.arange(pos, pos + n, dtype=np.float64)[None, :]) for pos, n in spec
    ]


# Windows worth checking: aligned, unaligned, sub-frame, spanning clusters, open-ended.
WINDOWS = [
    (0.0, 1.0), (0.0, 2.5), (0.5, 1.5), (1.0, 3.0), (2.5, 3.5), (3.0, 3.02),
    (4.9, 5.1), (5.0, 7.0), (6.25, 6.75), (7.0, None), (8.0, 9.0), (9.5, None),
    (0.0, None), (None, 4.0), (None, None), (0.137, 8.913),
]


@pytest.fixture(scope="module")
def mono_blob():
    return encode_webm(tone_ladder())


@pytest.fixture(scope="module")
def mono_full(mono_blob):
    return _read_webm(mono_blob, None, None).array


@pytest.fixture(scope="module")
def stereo_blob():
    # Different ladder per channel, so a swapped or collapsed channel is visible too.
    return encode_webm(np.stack([tone_ladder(), tone_ladder(offset=10)], axis=1))


@pytest.fixture(scope="module")
def stereo_full(stereo_blob):
    return _read_webm(stereo_blob, None, None).array


def expected_span(start, end, total):
    """The slice of the full decode a window is meant to reproduce."""
    lo = 0 if start is None else max(0, int(round(start * SR)))
    hi = total if end is None else min(total, max(lo, int(round(end * SR))))
    return lo, hi


class TestFullRead:
    """The unwindowed path stays exactly as it was."""

    def test_length_and_shape(self, mono_full):
        assert mono_full.shape == (LADDER_SECONDS * SR, 1)
        assert mono_full.dtype == np.float32

    def test_stereo_shape(self, stereo_full):
        assert stereo_full.shape == (LADDER_SECONDS * SR, 2)

    def test_sample_rate_and_type(self, mono_blob):
        result = _read_webm(mono_blob, None, None)
        assert result.sample_rate == SR
        assert result.file_type == "webm"
        assert result.modality == "audio"

    def test_channels_are_distinct(self, stereo_full):
        assert dominant_hz(stereo_full[:SR, 0]) == pytest.approx(500, abs=20)
        assert dominant_hz(stereo_full[:SR, 1]) == pytest.approx(5500, abs=20)


class TestWindowAlignment:
    """Every window is the slice of the full decode at that offset."""

    @pytest.mark.parametrize("start,end", WINDOWS)
    def test_window_length(self, mono_blob, mono_full, start, end):
        window = _read_webm(mono_blob, start, end).array
        lo, hi = expected_span(start, end, mono_full.shape[0])
        assert window.shape == (hi - lo, 1)

    @pytest.mark.parametrize("start,end", WINDOWS)
    def test_window_is_aligned(self, mono_blob, mono_full, start, end):
        window = _read_webm(mono_blob, start, end).array[:, 0]
        lo, hi = expected_span(start, end, mono_full.shape[0])
        reference = mono_full[lo:hi, 0]
        assert best_lag(window, reference) == 0
        assert rel_rms(window, reference) < 0.01

    @pytest.mark.parametrize("start,end", WINDOWS)
    def test_window_matches_stereo_reference(self, stereo_blob, stereo_full, start, end):
        window = _read_webm(stereo_blob, start, end).array
        lo, hi = expected_span(start, end, stereo_full.shape[0])
        assert window.shape == (hi - lo, 2)
        for ch in range(2):
            assert best_lag(window[:, ch], stereo_full[lo:hi, ch]) == 0
            assert rel_rms(window[:, ch], stereo_full[lo:hi, ch]) < 0.01

    @pytest.mark.parametrize("second", range(LADDER_SECONDS))
    def test_each_second_holds_its_own_tone(self, mono_blob, second):
        """The regression proper: a window must carry the audio at that timestamp, not
        the audio the seek happened to land on."""
        window = _read_webm(mono_blob, float(second), float(second) + 1.0).array[:, 0]
        assert window.size == SR
        assert dominant_hz(window) == pytest.approx(500 * (second + 1), abs=20)

    def test_mid_file_window_is_not_the_head_of_the_file(self, mono_blob, mono_full):
        """The exact shape of the old bug: [5s, 7s) came back as [0s, 2s)."""
        window = _read_webm(mono_blob, 5.0, 7.0).array[:, 0]
        head = mono_full[: 2 * SR, 0]
        assert rel_rms(window, head) > 0.5
        assert dominant_hz(window) == pytest.approx(3000, abs=20)

    def test_open_ended_window_runs_to_eof(self, mono_blob, mono_full):
        window = _read_webm(mono_blob, 7.0, None).array
        assert window.shape[0] == mono_full.shape[0] - 7 * SR

    def test_windows_tile_the_file(self, mono_blob, mono_full):
        """Consecutive windows reassemble the whole decode, nothing dropped or repeated."""
        pieces = [
            _read_webm(mono_blob, float(k), float(k) + 1.0).array
            for k in range(LADDER_SECONDS)
        ]
        stitched = np.concatenate(pieces, axis=0)
        assert stitched.shape == mono_full.shape
        assert best_lag(stitched[:, 0], mono_full[:, 0]) == 0
        assert rel_rms(stitched[:, 0], mono_full[:, 0]) < 0.01

    def test_overlapping_windows_agree(self, mono_blob):
        wide = _read_webm(mono_blob, 4.0, 6.0).array[:, 0]
        inner = _read_webm(mono_blob, 4.5, 5.5).array[:, 0]
        assert best_lag(inner, wide[SR // 2: SR + SR // 2]) == 0
        assert rel_rms(inner, wide[SR // 2: SR + SR // 2]) < 0.01

    def test_sub_frame_window(self, mono_blob, mono_full):
        """Shorter than one 20 ms Opus frame."""
        window = _read_webm(mono_blob, 6.0, 6.005).array
        assert window.shape == (240, 1)
        assert rel_rms(window[:, 0], mono_full[6 * SR: 6 * SR + 240, 0]) < 0.01


class TestSeekFidelity:
    """A window is not just aligned, it is the same audio: the seek margin has to be wide
    enough for the decoder to be warm before the window starts. At 80 ms of margin the
    first samples after a seek were out by ~0.09."""

    @pytest.mark.parametrize("start", [1.0, 2.5, 5.0, 7.0, 8.0, 8.5, 9.5])
    def test_window_matches_the_full_decode(self, mono_blob, mono_full, start):
        window = _read_webm(mono_blob, start, start + 0.4).array[:, 0]
        reference = mono_full[int(start * SR): int(start * SR) + window.size, 0]
        assert np.abs(window - reference).max() < 0.01

    @pytest.mark.parametrize("start", [3.0, 8.0])
    def test_first_samples_of_a_window_are_clean(self, mono_blob, mono_full, start):
        """The head of the window is where a cold decoder shows up first."""
        window = _read_webm(mono_blob, start, start + 0.4).array[:, 0]
        head = mono_full[int(start * SR): int(start * SR) + 480, 0]
        assert np.abs(window[:480] - head).max() < 0.01

    def test_seek_margin_covers_more_than_the_codec_preroll(self):
        """Opus asks for 80 ms; that only warms the decoder if the seek lands on an
        earlier cluster, which a wider margin makes likely."""
        from omniio.audio.read import _OPUS_SEEK_MARGIN_S

        assert _OPUS_SEEK_MARGIN_S >= 0.2


class TestMuxLayouts:
    """Cluster placement decides where a seek lands; none of it may leak into the result."""

    @pytest.mark.parametrize("mux,options", [
        ("single", None),
        ("chunked", None),
        ("single", {"cluster_time_limit": "500"}),
        ("chunked", {"cluster_time_limit": "2000"}),
    ])
    @pytest.mark.parametrize("start,end", [(0.0, 1.0), (2.5, 3.5), (5.0, 7.0), (9.0, None)])
    def test_window_survives_mux_layout(self, mux, options, start, end):
        blob = encode_webm(tone_ladder(), mux=mux, options=options)
        full = _read_webm(blob, None, None).array
        window = _read_webm(blob, start, end).array
        lo, hi = expected_span(start, end, full.shape[0])
        assert window.shape[0] == hi - lo
        assert best_lag(window[:, 0], full[lo:hi, 0]) == 0
        assert rel_rms(window[:, 0], full[lo:hi, 0]) < 0.01


class TestSourceRates:
    """Opus resamples to 48 kHz on the way in; windows are in seconds either way."""

    @pytest.mark.parametrize("rate", [16000, 24000, 48000])
    def test_window_alignment_at_source_rate(self, rate):
        blob = encode_webm(tone_ladder(rate=rate), rate=rate)
        result = _read_webm(blob, 5.0, 7.0)
        assert result.sample_rate == SR
        assert result.array.shape[0] == 2 * SR
        assert dominant_hz(result.array[:, 0]) == pytest.approx(3000, abs=25)


class TestEdgeCases:
    @pytest.mark.parametrize("start,end", [(11.0, 12.0), (LADDER_SECONDS + 0.5, None)])
    def test_window_past_eof_is_empty(self, mono_blob, start, end):
        window = _read_webm(mono_blob, start, end).array
        assert window.shape == (0, 1)
        assert window.dtype == np.float32

    def test_window_past_eof_keeps_channel_count(self, stereo_blob):
        assert _read_webm(stereo_blob, 30.0, 31.0).array.shape == (0, 2)

    def test_end_past_eof_is_clipped(self, mono_blob, mono_full):
        window = _read_webm(mono_blob, 9.0, 30.0).array
        assert window.shape[0] == mono_full.shape[0] - 9 * SR

    def test_empty_window(self, mono_blob):
        assert _read_webm(mono_blob, 4.0, 4.0).array.shape == (0, 1)

    def test_end_before_start_is_empty(self, mono_blob):
        assert _read_webm(mono_blob, 6.0, 5.0).array.shape == (0, 1)

    def test_negative_start_reads_from_zero(self, mono_blob, mono_full):
        window = _read_webm(mono_blob, -2.0, 1.0).array
        assert window.shape[0] == SR
        assert rel_rms(window[:, 0], mono_full[:SR, 0]) < 0.01

    def test_start_only_at_zero_matches_full_read(self, mono_blob, mono_full):
        assert _read_webm(mono_blob, 0.0, None).array.shape == mono_full.shape

    def test_very_short_file(self):
        blob = encode_webm(tone_ladder(seconds=1))
        assert _read_webm(blob, 0.25, 0.75).array.shape[0] == SR // 2


class TestArchiveEntryPoints:
    """The same window through the public readers, with the entry at a byte offset."""

    def test_local_read_window_at_offset(self, temp_dir, mono_blob):
        padding = b"\x00" * 1234
        archive = temp_dir / "archive.bin"
        archive.write_bytes(padding + mono_blob)

        result = audio_read_local(str(archive), len(padding), len(mono_blob), 5.0, 7.0)
        assert result.file_type == "webm"
        assert result.array.shape == (2 * SR, 1)
        assert dominant_hz(result.array[:, 0]) == pytest.approx(3000, abs=20)

    def test_local_read_second_entry(self, temp_dir, mono_blob, stereo_blob):
        archive = temp_dir / "two_entries.bin"
        archive.write_bytes(mono_blob + stereo_blob)

        result = audio_read_local(
            str(archive), len(mono_blob), len(stereo_blob), 3.0, 4.0
        )
        assert result.array.shape == (SR, 2)
        assert dominant_hz(result.array[:, 0]) == pytest.approx(2000, abs=20)
        assert dominant_hz(result.array[:, 1]) == pytest.approx(7000, abs=20)

    def test_remote_read_window(self, monkeypatch, mono_blob):
        class FakeResponse:
            content = mono_blob

            def raise_for_status(self):
                pass

        captured = {}

        def fake_get(url, headers=None, **kwargs):
            captured["headers"] = headers
            return FakeResponse()

        monkeypatch.setattr("omniio.audio.read.requests.get", fake_get)
        result = audio_read_remote("http://example/a.bin", 0, len(mono_blob), 5.0, 7.0)

        assert captured["headers"]["Range"] == f"bytes=0-{len(mono_blob) - 1}"
        assert result.array.shape == (2 * SR, 1)
        assert dominant_hz(result.array[:, 0]) == pytest.approx(3000, abs=20)


class TestGatherWindow:
    """The trimming math on its own: frame positions in, exact samples out."""

    def collected(self, spec, start, end):
        chunks, overshot = _gather_window(frames(*spec), start, end)
        assert not overshot
        return np.concatenate(chunks, axis=1)[0] if chunks else np.empty(0)

    def test_window_inside_one_frame(self):
        assert list(self.collected([(0, 100)], 10, 20)) == list(range(10, 20))

    def test_window_spanning_frames(self):
        got = self.collected([(0, 100), (100, 100), (200, 100)], 50, 250)
        assert list(got) == list(range(50, 250))

    def test_frames_before_the_window_are_dropped(self):
        got = self.collected([(0, 100), (100, 100)], 120, 180)
        assert list(got) == list(range(120, 180))

    def test_seek_landing_early_is_trimmed(self):
        """A seek that lands a whole cluster early must not shift the window."""
        got = self.collected([(4800, 960), (5760, 960)], 5000, 6000)
        assert list(got) == list(range(5000, 6000))

    def test_open_ended_window(self):
        got = self.collected([(0, 100), (100, 100)], 150, None)
        assert list(got) == list(range(150, 200))

    def test_window_from_zero(self):
        got = self.collected([(0, 100), (100, 100)], 0, None)
        assert list(got) == list(range(0, 200))

    def test_end_past_the_last_frame(self):
        got = self.collected([(0, 100)], 50, 10_000)
        assert list(got) == list(range(50, 100))

    def test_empty_window(self):
        assert self.collected([(0, 100)], 50, 50).size == 0

    def test_window_past_every_frame(self):
        assert self.collected([(0, 100)], 500, 600).size == 0

    def test_decoding_stops_at_the_window_end(self):
        """Frames after the window are never pulled from the decoder."""
        pulled = []

        def counting():
            for pos, arr in frames((0, 100), (100, 100), (200, 100), (300, 100)):
                pulled.append(pos)
                yield pos, arr

        chunks, _ = _gather_window(counting(), 50, 150)
        assert np.concatenate(chunks, axis=1).shape[1] == 100
        assert pulled == [0, 100]      # the frames at 200 and 300 are never decoded

    def test_overshot_seek_is_reported(self):
        """Nothing before the first decoded frame can be recovered — say so instead of
        returning the wrong samples."""
        chunks, overshot = _gather_window(frames((5000, 960)), 1000, 2000)
        assert overshot and chunks == []

    def test_overshoot_detection_is_off_for_a_run_from_the_top(self):
        """Nothing was seeked, so the first frame is the stream's first sample: it is the
        beginning of the file, not a seek that went too far."""
        chunks, overshot = _gather_window(
            frames((5000, 960)), 1000, 6000, detect_overshoot=False
        )
        assert not overshot
        assert np.concatenate(chunks, axis=1).shape[1] == 960

    def test_no_overshoot_when_frame_starts_exactly_at_the_window(self):
        chunks, overshot = _gather_window(frames((1000, 960)), 1000, 2000)
        assert not overshot and chunks

    def test_multichannel_is_trimmed_per_channel(self):
        arr = np.vstack([np.arange(0, 100), np.arange(1000, 1100)]).astype(np.float64)
        chunks, overshot = _gather_window([(0, arr)], 10, 20)
        got = np.concatenate(chunks, axis=1)
        assert not overshot
        assert got.shape == (2, 10)
        assert list(got[0]) == list(range(10, 20))
        assert list(got[1]) == list(range(1010, 1020))


class TestOvershootFallback:
    """If a seek lands after the requested start, the reader decodes from the top rather
    than handing back whatever it found there."""

    def test_reader_recovers_from_an_overshooting_seek(self, monkeypatch, mono_blob, mono_full):
        import omniio.audio.read as read_mod

        real_frames = read_mod._webm_frames
        calls = []

        def overshooting(container, stream, seek_sample):
            calls.append(seek_sample)
            if len(calls) == 1:                      # pretend the first seek went too far
                return real_frames(container, stream, int(8.5 * SR))
            return real_frames(container, stream, seek_sample)

        monkeypatch.setattr(read_mod, "_webm_frames", overshooting)
        window = read_mod._read_webm(mono_blob, 5.0, 7.0).array

        assert calls == [5 * SR, None]
        assert window.shape == (2 * SR, 1)
        assert best_lag(window[:, 0], mono_full[5 * SR: 7 * SR, 0]) == 0
        assert rel_rms(window[:, 0], mono_full[5 * SR: 7 * SR, 0]) < 0.01


# ---- streams that real ffmpeg will not easily produce -----------------------------------
# A fake container models the decode the reader has to cope with: a first frame shortened
# by the codec delay, frames handed back in cluster-sized runs, millisecond timestamps,
# and — after a seek — the decoder discarding the codec delay again, which is what puts a
# seeked run on a whole-frame boundary. Two things differ between PyAV/ffmpeg builds and
# the reader must not depend on either, so every synthetic test runs all four ways:
# whether the trimmed frame's pts stays equal to its packet's or is moved past the
# discarded samples (rounded to the tick), and whether `stream.start_time` is the first
# packet's pts (PyAV <= 16) or the first frame's presentation time, packet pts plus the
# codec delay (PyAV >= 17). The signal is a ramp, so every sample says which index it is
# and the assertions can be exact.

class FakeFormat:
    is_planar = True


class FakeFrame:
    format = FakeFormat()

    def __init__(self, pts, data):
        self.pts = pts
        self._data = data
        self.samples = data.shape[1]

    def to_ndarray(self):
        return self._data


class FakePacket:
    def __init__(self, pts, frames):
        self.pts = pts
        self._frames = frames

    def decode(self):
        return list(self._frames)


class FakeStream:
    def __init__(self, rate, time_base, start_time, channels):
        self.rate = rate
        self.time_base = time_base
        self.start_time = start_time
        self.channels = channels


class FakeContainer:
    """A decodable audio stream with controllable framing, timestamps and clusters."""

    # Build-dependent timestamp details, flipped for every synthetic test (see
    # `decoder_pts_behaviour`): whether a trimmed frame's pts is moved past the samples it
    # discarded, and whether stream.start_time is the first packet's pts or the first
    # frame's presentation time (packet pts + codec delay).
    advance_trimmed_pts = False
    start_time_is_first_frame = False

    def __init__(self, samples, rate=SR, frame=960, delay=312, cluster_frames=100,
                 start_time=7, tick=Fraction(1, 1000), gap_after=None, gap_frames=0):
        self.rate = rate
        self.frame = frame
        self.delay = delay
        self.cluster_frames = cluster_frames
        self.tick = tick
        self.gap_after = gap_after          # index of the frame the gap follows
        self.gap_frames = gap_frames        # frames dropped from the stream (DTX-like)
        self.signal = np.arange(samples, dtype=np.float64)[None, :]
        self.n_frames = (samples + delay + frame - 1) // frame
        self._base = start_time                 # pts of the first packet
        reported = start_time
        if start_time is not None and self.start_time_is_first_frame:
            reported = start_time + int(round(delay / rate / float(tick)))
        self.streams = SimpleNamespace(audio=[FakeStream(rate, tick, reported, 1)])
        self.seeks = []
        self._cursor = 0

    # --- the stream's own layout -------------------------------------------------------
    def _span(self, k):
        """(index of the first sample, index past the last) of frame k, index 0 being the
        stream's first decoded sample."""
        lo = max(0, k * self.frame - self.delay)
        hi = min(self.signal.shape[1], (k + 1) * self.frame - self.delay)
        return lo, hi

    def _pts(self, k):
        """Frame k's timestamp: the frame's position in time, quantised to the tick."""
        base = self._base or 0
        if k == 0:
            return base
        emitted = k - self.gap_frames if self.gap_after is not None and k > self.gap_after else k
        return base + int(round((emitted * self.frame / self.rate) / float(self.tick)))

    def _dropped(self, k):
        return (
            self.gap_after is not None
            and self.gap_after < k <= self.gap_after + self.gap_frames
        )

    # --- the container API the reader uses ---------------------------------------------
    def seek(self, timestamp, stream=None, **kwargs):
        self.seeks.append(timestamp)
        base = self._base or 0
        target = max(0, timestamp - base)
        frames_in = int(target * float(self.tick) * self.rate) // self.frame
        self._cursor = min(frames_in - frames_in % self.cluster_frames, self.n_frames)

    def demux(self, stream=None):
        """Packets from the cursor on, each decoding to at most one frame, then the flush
        packet PyAV appends. The packet's pts is the frame's position in the stream; the
        frame's pts is the same unless `advance_trimmed_pts` says the decoder moved it."""
        for k in range(self._cursor, self.n_frames):
            if self._dropped(k):
                continue
            pts = self._pts(k)
            lo, hi = self._span(k)
            if hi <= lo:
                yield FakePacket(pts, [])      # swallowed whole by the codec delay
                continue
            frame_pts = pts
            # After a seek onto frame k > 0, the decoder drops the codec delay again.
            if k == self._cursor and k > 0:
                trimmed = min(hi - lo, self.delay)
                lo += trimmed
                if self.advance_trimmed_pts:
                    frame_pts = pts + int(round(trimmed / self.rate / float(self.tick)))
            frames = [FakeFrame(frame_pts, self.signal[:, lo:hi].copy())] if hi > lo else []
            yield FakePacket(pts, frames)
        yield FakePacket(None, [])
        self._cursor = self.n_frames

    def decode(self, audio=0):
        for packet in self.demux():
            yield from packet.decode()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def fake_read(blob_key, factory, start_time, end_time, monkeypatch):
    """Run _read_webm against a fresh FakeContainer, recording the containers it opened."""
    import omniio.audio.read as read_mod

    opened = []

    def fake_open(buf, mode="r", **kwargs):
        container = factory()
        opened.append(container)
        return container

    monkeypatch.setattr(read_mod.av, "open", fake_open)
    result = read_mod._read_webm(blob_key, start_time, end_time)
    return result, opened


class TestSyntheticStreams:
    @pytest.fixture(
        autouse=True,
        params=[(False, False), (True, False), (False, True), (True, True)],
        ids=["pts=packet,start=packet", "pts=advanced,start=packet",
             "pts=packet,start=frame", "pts=advanced,start=frame"],
    )
    def decoder_pts_behaviour(self, request, monkeypatch):
        advance, start_is_frame = request.param
        monkeypatch.setattr(FakeContainer, "advance_trimmed_pts", advance)
        monkeypatch.setattr(FakeContainer, "start_time_is_first_frame", start_is_frame)

    """Position arithmetic on streams whose framing and timestamps are pinned by hand."""

    @pytest.mark.parametrize("start,end", [
        (1.0, 2.0), (2.5, 3.5), (0.9999, 1.9999), (5.0, None), (0.5, 0.75), (4.0, 4.001),
    ])
    def test_window_is_exact(self, monkeypatch, start, end):
        total = 10 * SR
        result, _ = fake_read(b"", lambda: FakeContainer(total), start, end, monkeypatch)
        lo = int(round(start * SR))
        hi = total if end is None else int(round(end * SR))
        assert list(result.array[:, 0]) == list(range(lo, hi))

    @pytest.mark.parametrize("delay", [0, 120, 312, 900])
    def test_any_codec_delay(self, monkeypatch, delay):
        result, _ = fake_read(
            b"", lambda: FakeContainer(10 * SR, delay=delay), 3.0, 4.0, monkeypatch
        )
        assert list(result.array[:, 0]) == list(range(3 * SR, 4 * SR))

    @pytest.mark.parametrize("frame", [120, 240, 480, 960, 1920, 2880])
    def test_any_opus_frame_size(self, monkeypatch, frame):
        result, _ = fake_read(
            b"",
            lambda: FakeContainer(10 * SR, frame=frame, delay=frame // 4,
                                  cluster_frames=max(1, 96000 // frame)),
            3.0, 4.0, monkeypatch,
        )
        assert list(result.array[:, 0]) == list(range(3 * SR, 4 * SR))

    @pytest.mark.parametrize("cluster_frames", [1, 7, 50, 100, 1000])
    def test_any_cluster_size(self, monkeypatch, cluster_frames):
        result, _ = fake_read(
            b"", lambda: FakeContainer(10 * SR, cluster_frames=cluster_frames),
            6.0, 7.0, monkeypatch,
        )
        assert list(result.array[:, 0]) == list(range(6 * SR, 7 * SR))

    @pytest.mark.parametrize("start_time", [0, 7, 5000, 123456])
    def test_stream_starting_at_any_timestamp(self, monkeypatch, start_time):
        result, _ = fake_read(
            b"", lambda: FakeContainer(10 * SR, start_time=start_time), 3.0, 4.0, monkeypatch
        )
        assert list(result.array[:, 0]) == list(range(3 * SR, 4 * SR))

    @pytest.mark.parametrize("tick", [Fraction(1, 1000), Fraction(1, 100), Fraction(1, 48000)])
    def test_any_timestamp_resolution(self, monkeypatch, tick):
        result, _ = fake_read(
            b"", lambda: FakeContainer(10 * SR, tick=tick), 3.0, 4.0, monkeypatch
        )
        assert list(result.array[:, 0]) == list(range(3 * SR, 4 * SR))

    def test_stream_without_a_start_timestamp_is_decoded_from_the_top(self, monkeypatch):
        """No timeline to seek against: decode from the first sample rather than guess."""
        result, opened = fake_read(
            b"", lambda: FakeContainer(10 * SR, start_time=None), 5.0, 6.0, monkeypatch
        )
        assert opened[0].seeks == []
        assert list(result.array[:, 0]) == list(range(5 * SR, 6 * SR))

    def test_window_inside_the_seek_margin_does_not_seek(self, monkeypatch):
        result, opened = fake_read(
            b"", lambda: FakeContainer(10 * SR), 0.1, 0.6, monkeypatch
        )
        assert opened[0].seeks == []
        assert list(result.array[:, 0]) == list(range(int(0.1 * SR), int(0.6 * SR)))

    def test_far_window_does_seek(self, monkeypatch):
        _, opened = fake_read(b"", lambda: FakeContainer(10 * SR), 5.0, 6.0, monkeypatch)
        assert opened[0].seeks and opened[0].seeks[0] < 5000

    def test_single_frame_stream(self, monkeypatch):
        result, _ = fake_read(b"", lambda: FakeContainer(600, frame=960), None, None, monkeypatch)
        assert list(result.array[:, 0]) == list(range(600))

    def test_frames_the_grid_does_not_describe_stay_within_the_tick(self, monkeypatch):
        """A non-Opus frame length (Vorbis in WebM, say) is not snapped: the position then
        carries the container's own timestamp error, and no more."""
        result, _ = fake_read(
            b"", lambda: FakeContainer(10 * SR, frame=1024, delay=0, cluster_frames=90),
            3.0, 4.0, monkeypatch,
        )
        got = result.array[:, 0]
        assert got.size == SR
        assert abs(int(got[0]) - 3 * SR) <= 24        # half a millisecond at 48 kHz

    @pytest.mark.parametrize("frame,cluster_frames", [(120, 7), (240, 3), (480, 13), (120, 31)])
    def test_cluster_boundaries_off_the_millisecond_grid(self, monkeypatch, frame, cluster_frames):
        """Short Opus frames put cluster boundaries on half-milliseconds, which is exactly
        where a timestamp-derived position rounds to the wrong sample."""
        result, _ = fake_read(
            b"",
            lambda: FakeContainer(10 * SR, frame=frame, delay=frame // 4,
                                  cluster_frames=cluster_frames),
            5.0, 5.25, monkeypatch,
        )
        assert list(result.array[:, 0]) == list(range(5 * SR, 5 * SR + SR // 4))

    def test_timestamps_with_a_gap_follow_the_timeline(self, monkeypatch):
        """Frames missing from the stream (DTX, dropped packets) shift what a timestamp
        points at. A seeked read follows the timestamps, so it returns the audio the
        container says is at that time — which is NOT the same index a full decode has,
        because a full decode just concatenates what it is given."""
        def factory():
            return FakeContainer(10 * SR, gap_after=200, gap_frames=50)

        windowed, _ = fake_read(b"", factory, 5.0, 5.5, monkeypatch)
        full, _ = fake_read(b"", factory, None, None, monkeypatch)

        gap = 50 * 960
        assert list(windowed.array[:, 0]) == list(range(5 * SR + gap, 5 * SR + gap + SR // 2))
        assert list(full.array[: SR, 0]) == list(range(SR))     # full read stays on samples


class TestSnapToGrid:
    def test_snaps_within_the_tick(self):
        assert _snap_to_grid(383976, 960, 25) == 384000
        assert _snap_to_grid(384024, 960, 25) == 384000

    def test_leaves_a_position_that_is_already_on_the_grid(self):
        assert _snap_to_grid(384000, 960, 25) == 384000

    def test_never_moves_further_than_the_tolerance(self):
        assert _snap_to_grid(384500, 960, 25) == 384500

    def test_zero_is_on_the_grid(self):
        assert _snap_to_grid(0, 960, 25) == 0
        assert _snap_to_grid(12, 960, 25) == 0

    def test_degenerate_frame_size(self):
        assert _snap_to_grid(1234, 0, 25) == 1234

    @pytest.mark.parametrize("n,expected", [
        (120, True), (240, True), (480, True), (960, True), (1920, True), (2880, True),
        (1024, False), (576, False), (0, False), (4096, False),
    ])
    def test_opus_frame_sizes(self, n, expected):
        assert _is_opus_frame_size(n, SR) is expected

    def test_opus_frame_sizes_at_other_rates(self):
        assert _is_opus_frame_size(320, 16000)       # 20 ms at 16 kHz
        assert not _is_opus_frame_size(1024, 16000)


class TestPyAVCompatibilityCheck:
    """The import-time check names a PyAV or libavcodec outside the swept range and is
    silent inside it."""

    def _with(self, monkeypatch, pyav, lavc):
        import omniio.audio.read as read_mod
        monkeypatch.setattr(read_mod.av, "__version__", pyav)
        monkeypatch.setattr(read_mod.av, "library_versions", {"libavcodec": lavc})
        return pyav_compatibility()

    def test_inside_the_range_is_silent(self, monkeypatch):
        assert self._with(monkeypatch, "17.1.0", (62, 28, 100)) is None
        assert self._with(monkeypatch, "9.2.0", (58, 134, 100)) is None

    def test_pyav_outside_the_range(self, monkeypatch):
        msg = self._with(monkeypatch, "19.0.0", (62, 28, 100))
        assert msg and "PyAV 19.0.0" in msg and "libavcodec 62.28" not in msg

    def test_libavcodec_outside_the_range(self, monkeypatch):
        msg = self._with(monkeypatch, "18.1.0", (63, 2, 100))
        assert msg and "libavcodec 63.2" in msg and "PyAV 18.1.0" not in msg

    def test_prerelease_version_string_is_parsed(self, monkeypatch):
        assert self._with(monkeypatch, "18.1.0rc1", (62, 28, 100)) is None

    def test_the_installed_build_is_in_the_range_or_warned_about(self):
        import av
        msg = pyav_compatibility()
        if msg is None:
            return
        pytest.skip(f"PyAV {av.__version__} is outside the swept range: {msg}")

    def test_warning_category_is_filterable(self):
        assert issubclass(PyAVCompatibilityWarning, UserWarning)
        assert _TESTED_PYAV[0] < _TESTED_PYAV[1] and _TESTED_LIBAVCODEC[0] < _TESTED_LIBAVCODEC[1]
