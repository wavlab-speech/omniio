"""
Decode-only validation of windowed WebM/Opus reads on the installed PyAV.

For builds without an Opus *encoder* (the conda-forge PyAV 9.2 build for
linux-aarch64 links an ffmpeg with the native decoder only), the test suite's
clip encoder cannot run, but the property the sweep guards is about decoding:
that a window read through omniio lands on the same samples as the full
decode. This reads a blob encoded elsewhere (tests/tools/sweep_pyav_webm.sh or
any PyAV with libopus can produce one), takes 25 windows at spread offsets and
reports the largest deviation from the full decode in samples and in value.

    python tests/tools/webm_decode_check.py ladder.webm      # exit 1 if a window is misaligned

Make the blob with a PyAV that has libopus:
    python -c "import sys; sys.path.insert(0, 'tests'); from test_webm_opus_windows import *; \
               open('ladder.webm', 'wb').write(encode_webm(tone_ladder()))"
"""

import sys

import av
import numpy as np

from omniio.audio.read import _read_webm, pyav_compatibility

VALUE_TOL = 1e-3  # an off-by-one-frame misalignment of the tone ladder is ~0.1-1.0
N_WINDOWS = 25
WINDOW_S = 1.5


def main(path: str) -> int:
    blob = open(path, "rb").read()
    full = _read_webm(blob, None, None)
    samples, sr = full.array.shape[0], full.sample_rate
    worst_value, worst_len = 0.0, 0
    for t0 in np.linspace(0.3, samples / sr - WINDOW_S - 0.5, N_WINDOWS):
        win = _read_webm(blob, float(t0), float(t0) + WINDOW_S).array
        a = int(round(t0 * sr))
        ref = full.array[a : a + win.shape[0]]
        n = min(len(ref), len(win))
        worst_value = max(worst_value, float(np.abs(ref[:n] - win[:n]).max()))
        worst_len = max(worst_len, abs(len(ref) - len(win)))
    lv = getattr(av, "library_versions", {}).get("libavcodec", ("?", "?"))
    ok = worst_value <= VALUE_TOL and worst_len == 0
    version = getattr(av, "__version__", "?")
    verdict = "aligned" if ok else "MISALIGNED"
    compat = pyav_compatibility() or "inside the validated range"
    print(
        f"av {version} | libavcodec {lv[0]}.{lv[1]} | full decode {samples} samples | "
        f"{N_WINDOWS} windows vs full decode: max abs diff {worst_value:.2e}, "
        f"length mismatch {worst_len} | {verdict} | {compat}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
