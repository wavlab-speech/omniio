"""Windowed video reads must return the audio that is at the requested time.

`_decode_video` decodes audio in a pass of its own, over a second container that it seeks
to the start of the window. A seek lands at or before the target -- on a packet boundary
in MP4, on a cluster boundary in WebM, which can be seconds away -- so the run it starts
begins with a lead-in that is not part of the window. The pass used to keep the first
(end - start) samples of that run, lead-in included, which put the audio wherever the seek
happened to land: measured on the fixtures below, a WebM window asked for at 17 s came
back holding the audio at 23.7 s.

Position is asserted by the peak of a normalised cross-correlation against a full decode of
the same file, not by sample-wise equality. A seeked window is not bit-exact -- the decoder
starts cold -- so an RMS difference says nothing about where the window came from, while
the correlation peak pins it to a sample.

The fixtures are generated here and never committed.
"""
import fractions

import av
import numpy as np
import pytest

from omniio.video.read import _decode_video, video_read_local

SR = 48000
SECONDS = 40
FPS = 25
TOL_MS = 3.0            # a window must land within this of the time it was asked for


def _signal(seconds=SECONDS, rate=SR):
    """Broadband noise: every offset is unique, so the correlation peak is unambiguous.
    A tone, or a frame counter, would only localise to the length of one repeat."""
    rng = np.random.default_rng(20261019)
    return (0.3 * rng.standard_normal(rate * seconds)).astype(np.float32)


def _encode(path, signal, vcodec, acodec, gop):
    """A clip with a video and an audio stream. `gop` sets the keyframe spacing, which is
    what WebM puts its cluster boundaries on: wide clusters are what make a seek land far
    enough before the target for a missing lead-in trim to show."""
    with av.open(str(path), mode="w") as out:
        video = out.add_stream(vcodec, rate=FPS)
        video.width, video.height, video.pix_fmt = 64, 64, "yuv420p"
        video.options = {"g": str(gop)}
        audio = out.add_stream(acodec, rate=SR)
        audio.layout = "mono"

        for i in range(SECONDS * FPS):
            frame = av.VideoFrame.from_ndarray(
                np.full((64, 64, 3), i % 256, dtype=np.uint8), format="rgb24"
            )
            frame.pts = i
            frame.time_base = fractions.Fraction(1, FPS)
            out.mux(video.encode(frame))
        out.mux(video.encode(None))

        pcm = (np.clip(signal, -1, 1) * 32767).astype(np.int16)[None, :]
        pts = 0
        for i in range(0, pcm.shape[1], 1024):
            chunk = np.ascontiguousarray(pcm[:, i:i + 1024])
            frame = av.AudioFrame.from_ndarray(chunk, format="s16p", layout="mono")
            frame.sample_rate = SR
            frame.pts = pts
            frame.time_base = fractions.Fraction(1, SR)
            pts += chunk.shape[1]
            out.mux(audio.encode(frame))
        out.mux(audio.encode(None))


CLIPS = {
    # label: (suffix, video codec, audio codec, keyframe spacing in frames,
    #         how early a seek must land for the fixture to exercise the bug, in seconds)
    #
    # MP4 indexes AAC per packet, so a seek there lands at most one packet (1024 samples,
    # 21 ms) early: that is the whole of the lead-in this container can have, and it is
    # still several times TOL_MS. WebM can only be seeked to a cluster.
    "mp4": (".mp4", "libx264", "aac", 50, TOL_MS / 1000),
    "webm": (".webm", "libvpx", "libopus", 250, 0.5),
}


@pytest.fixture(scope="module", params=sorted(CLIPS), ids=sorted(CLIPS))
def clip(request, tmp_path_factory):
    """(path, size, full-decode reference, minimum seek lead-in) for one container."""
    label = request.param
    suffix, vcodec, acodec, gop, min_lead_in = CLIPS[label]
    path = tmp_path_factory.mktemp("omniio_video") / f"clip{suffix}"
    signal = _signal()
    try:
        _encode(path, signal, vcodec, acodec, gop)
    except Exception as exc:                        # pragma: no cover - build-dependent
        pytest.skip(f"cannot encode {vcodec}+{acodec} on this build: {exc}")

    size = path.stat().st_size
    full = video_read_local(str(path), 0, size)
    assert full.audio_array.shape[0] > SR * (SECONDS - 1), "fixture did not decode"
    return str(path), size, full, min_lead_in


def _peak_lag(probe, segment):
    """(offset of `probe` inside `segment`, in samples; normalised peak correlation)."""
    probe = np.asarray(probe, dtype=np.float64)
    segment = np.asarray(segment, dtype=np.float64)
    nfft = 1 << (len(segment) + len(probe) - 1).bit_length()
    corr = np.fft.irfft(
        np.fft.rfft(segment - segment.mean(), nfft)
        * np.conj(np.fft.rfft(probe - probe.mean(), nfft)),
        nfft,
    )[: len(segment) - len(probe) + 1]
    cumsq = np.concatenate([[0.0], np.cumsum(segment**2)])
    energy = cumsq[len(probe):len(probe) + len(corr)] - cumsq[:len(corr)]
    denom = np.sqrt(
        np.maximum(energy, 1e-30) * max(np.sum((probe - probe.mean()) ** 2), 1e-30)
    )
    peak = int(np.argmax(corr / denom))
    return peak, float((corr / denom)[peak])


def _landing(window, reference, start, sample_rate):
    """Where `window` actually sits in `reference`, in seconds, and how well it matches."""
    probe = window.mean(axis=1)[:sample_rate]
    lo = int(max(0.0, start - 15.0) * sample_rate)
    hi = int(min(len(reference) / sample_rate, start + 15.0 + 1.0) * sample_rate)
    peak, corr = _peak_lag(probe, reference[lo:hi])
    return (lo + peak) / sample_rate, corr


def _seek_lead_in(path, size):
    """How far before the target the audio stream's seeks actually land, in seconds.

    Measured by seeking, not read off keyframe flags: in Matroska every Opus packet is a
    sync point, so `is_keyframe` says nothing about where the clusters are, and a guard
    written that way would pass while measuring nothing.
    """
    lags = []
    with open(path, "rb") as f:
        blob = f.read(size)
    import io

    with av.open(io.BytesIO(blob)) as container:
        stream = container.streams.audio[0]
        for target in (12.0, 17.0, 23.0, 33.0):
            container.seek(int(target / stream.time_base), stream=stream)
            pts = next(
                (p.pts for p in container.demux(stream) if p.pts is not None), None
            )
            if pts is not None:
                lags.append(target - float(pts * stream.time_base))
    return lags


def test_seeks_land_before_the_target(clip):
    """Guard on the fixture itself.

    The bug is "the lead-in is never discarded", so it can only show where a seek lands
    before the target by more than the tolerance the alignment tests assert. If this
    build's encoder produced finer-grained clusters than the fixture asks for, every
    window would be accidentally right and the tests below would pass against the old
    reader too.
    """
    path, size, _, min_lead_in = clip
    lags = _seek_lead_in(path, size)
    assert lags, "no packet demuxed after a seek"
    assert max(lags) >= min_lead_in, (
        f"seeks land at most {max(lags) * 1000:.1f} ms before the target on this build, "
        f"under the {min_lead_in * 1000:.1f} ms this fixture needs to exercise the "
        "lead-in; the alignment assertions would not be sensitive to it"
    )


@pytest.mark.parametrize("start", [5.0, 17.0, 23.0, 33.0])
def test_audio_window_lands_at_the_requested_time(clip, start):
    """The audio of a window must be the audio a full decode has at that time."""
    path, size, full, _ = clip
    reference = full.audio_array.mean(axis=1)

    result = video_read_local(path, 0, size, start_time=start, end_time=start + 2.0)
    assert result.audio_array.shape[0] > full.sample_rate, "window too short to localise"

    landed, corr = _landing(result.audio_array, reference, start, full.sample_rate)
    assert corr > 0.3, (
        f"the audio returned for {start:.3f}s does not match the file anywhere near it "
        f"(best r={corr:.3f} at {landed:.3f}s)"
    )
    assert abs(landed - start) * 1000 <= TOL_MS, (
        f"asked for {start:.3f}s, got the audio at {landed:.3f}s "
        f"({(landed - start) * 1000:+.1f} ms, r={corr:.3f})"
    )


@pytest.mark.parametrize("start_frame", [125, 425, 825])
def test_audio_window_follows_start_frame(clip, start_frame):
    """A window given by frame index puts the audio at that frame's time."""
    path, size, full, _ = clip
    reference = full.audio_array.mean(axis=1)
    start = start_frame / full.fps

    result = video_read_local(
        path, 0, size, start_frame=start_frame, end_frame=start_frame + 2 * FPS
    )
    landed, corr = _landing(result.audio_array, reference, start, full.sample_rate)
    assert corr > 0.3 and abs(landed - start) * 1000 <= TOL_MS, (
        f"frame {start_frame} is {start:.3f}s, got the audio at {landed:.3f}s "
        f"({(landed - start) * 1000:+.1f} ms, r={corr:.3f})"
    )


def test_audio_is_cut_on_the_video_frame_grid(clip):
    """A start_time between two frames is resolved to the frame before it -- for the audio
    as well as the video, so the two arrays describe the same span.

    5.039 s is frame 125.975, so the window starts at frame 125, i.e. 5.000 s. Audio cut
    at the raw 5.039 s instead would sit 39 ms ahead of the video it is returned with.
    """
    path, size, full, _ = clip
    reference = full.audio_array.mean(axis=1)
    asked = 125.975 / FPS
    on_grid = 125 / FPS

    result = video_read_local(path, 0, size, start_time=asked, end_time=asked + 2.0)
    landed, corr = _landing(result.audio_array, reference, on_grid, full.sample_rate)
    assert corr > 0.3
    assert abs(landed - on_grid) * 1000 <= TOL_MS, (
        f"asked for {asked:.3f}s, which is frame 125 at {on_grid:.3f}s; "
        f"got the audio at {landed:.3f}s"
    )


def test_audio_and_video_cover_the_same_span(clip):
    """Whatever the window, the two arrays are the same length in seconds."""
    path, size, full, _ = clip
    result = video_read_local(path, 0, size, start_time=17.0, end_time=20.0)
    video_seconds = result.video_array.shape[0] / result.fps
    audio_seconds = result.audio_array.shape[0] / result.sample_rate
    assert abs(video_seconds - 3.0) < 0.1
    assert abs(audio_seconds - video_seconds) < 0.05


def test_window_contract_is_unchanged(clip):
    """Shapes, dtypes and the integer-normalised range the caller already relies on."""
    path, size, full, _ = clip
    result = video_read_local(path, 0, size, start_time=17.0, end_time=19.0)

    assert result.audio_array.ndim == 2
    assert result.audio_array.dtype == np.float32
    assert result.audio_array.shape[1] == full.audio_array.shape[1]
    assert abs(result.audio_array.shape[0] - 2 * full.sample_rate) <= full.sample_rate // 50
    # Around full scale, but not clamped to it: a lossy decoder overshoots (1.03 on this
    # fixture). What the reader owes the caller is the dtype scaling, which
    # test_whole_file_read_is_unchanged pins against a full decode sample for sample.
    assert np.isfinite(result.audio_array).all()
    assert 0.1 < np.abs(result.audio_array).max() < 2.0

    assert result.video_array.ndim == 4
    assert result.video_array.dtype == np.uint8
    assert result.video_array.shape[1:] == (64, 64, 3)
    assert abs(result.video_array.shape[0] - 2 * FPS) <= 2


def test_whole_file_read_is_unchanged(clip):
    """No window: the reader still returns the file from its first sample."""
    path, size, full, _ = clip
    head = _decode_video(open(path, "rb").read(size), end_time=2.0)
    assert head.audio_array.shape[0] > 0
    np.testing.assert_allclose(
        head.audio_array[: 2 * SR], full.audio_array[: 2 * SR], atol=1e-6
    )


def test_frame_indices_take_priority_over_time(clip):
    """Documented contract: when both are given, the frame indices win -- for the audio
    bounds too, which are derived from them."""
    path, size, full, _ = clip
    reference = full.audio_array.mean(axis=1)
    start_frame = 425
    start = start_frame / full.fps

    result = video_read_local(
        path, 0, size,
        start_frame=start_frame, end_frame=start_frame + 2 * FPS,
        start_time=3.0, end_time=5.0,          # must be ignored
    )
    landed, corr = _landing(result.audio_array, reference, start, full.sample_rate)
    assert corr > 0.3 and abs(landed - start) * 1000 <= TOL_MS, (
        f"start_time should have been ignored; audio landed at {landed:.3f}s, "
        f"expected {start:.3f}s (r={corr:.3f})"
    )
