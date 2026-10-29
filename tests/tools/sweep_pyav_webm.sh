#!/bin/bash
# Run the WebM/Opus window tests against several PyAV wheels (each bundles its own ffmpeg).
#
#   tests/tools/sweep_pyav_webm.sh [VERSION ...]    default: one wheel per ffmpeg generation
#
# Each wheel is installed into its own directory (pip --target) and put ahead of the
# interpreter's own PyAV on PYTHONPATH for that command only, so nothing in the
# environment is modified and no version leaks into the next iteration. Every row
# checks that the `av` it imported is the version it is labelled with (a directory left
# by an interrupted install is reinstalled, and a mismatch is reported instead of a
# result), then prints the bundled libavcodec, what the build reports for
# stream.start_time and for the first frame's pts after a seek, and the test result.
# Thread pools are pinned to one thread: on a shared node, N parallel unpinned runs can
# each take 50 CPU-minutes instead of ten seconds.
set -u
cd "$(dirname "$0")/../.."
PY=${PYTHON:-python}
VERSIONS=${*:-"12.3.0 13.1.0 14.2.0 15.1.0 16.1.0 17.1.0 18.1.0"}
WORK=${SWEEP_DIR:-/tmp/omniio-pyav-sweep}
mkdir -p "$WORK"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

version_of='import av; print(getattr(av, "__version__", ""))'
probe='
import io, sys, av
sys.path.insert(0, "tests")
from test_webm_opus_windows import tone_ladder, encode_webm
blob = encode_webm(tone_ladder())
with av.open(io.BytesIO(blob)) as c:
    s = c.streams.audio[0]
    c.seek(int(8.75 / s.time_base), stream=s)
    head = next((p.pts, f.pts, f.samples) for p in c.demux(s) for f in p.decode())
with av.open(io.BytesIO(blob)) as c:
    s = c.streams.audio[0]
    first = next(p.pts for p in c.demux(s) if p.pts is not None)
lavc = ".".join(map(str, getattr(av, "library_versions", {}).get("libavcodec", ("?",))[:2]))
print(f"libavcodec {lavc} | first packet pts {first} | stream.start_time {s.start_time} | "
      f"after seek: packet pts {head[0]}, frame pts {head[1]}, {head[2]} samples")
'
installed_version() {   # the av that PYTHONPATH=$1 actually imports, or empty
    PYTHONPATH="$1" $PY -c "$version_of" 2>/dev/null
}
for v in $VERSIONS; do
    dir="$WORK/av-$v"
    if [ "$(installed_version "$dir")" != "$v" ]; then
        rm -rf "$dir"
        $PY -m pip install -q --no-deps --only-binary=:all: --target "$dir" "av==$v" \
            || { echo "av $v | no wheel for this platform/interpreter"; continue; }
    fi
    got=$(installed_version "$dir")
    if [ "$got" != "$v" ]; then
        echo "av $v | MISMATCH: imported av '${got:-none}' from $dir; row not labelled"
        continue
    fi
    info=$(PYTHONPATH="$dir" $PY -c "$probe" 2>/dev/null | tail -1)
    result=$(PYTHONPATH="$dir" $PY -m pytest tests/test_webm_opus_windows.py -o addopts= -q -p no:cacheprovider 2>&1 | tail -1)
    echo "av $v | $info | $result"
done
