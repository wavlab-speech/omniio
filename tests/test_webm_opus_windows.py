"""Windowed (partial) reads of WebM/Opus.

WebM can only be seeked to a cluster boundary, so the seek lands at or *before* the
requested start. The reader used to keep the first `end - start` samples of whatever the
seek produced, which silently returned audio from the wrong part of the file (a request
for seconds 5-7 could return seconds 0-2). These tests pin the window to the timeline:
every window must be the same samples the full decode has at that offset.
"""

import io

import av
import numpy as np
import pytest

from omniio.modalities.audio.read import (
    _gather_window,
    _read_webm,
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
        from omniio.modalities.audio.read import _OPUS_SEEK_MARGIN_S

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

        monkeypatch.setattr("omniio.modalities.audio.read.requests.get", fake_get)
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
        import omniio.modalities.audio.read as read_mod

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
