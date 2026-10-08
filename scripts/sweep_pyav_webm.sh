#!/bin/bash
# Run the WebM/Opus window tests against several PyAV wheels (each bundles its own ffmpeg).
#
#   scripts/sweep_pyav_webm.sh [VERSION ...]        default: one wheel per ffmpeg generation
#
# Each wheel is installed into its own directory (pip --target) and put ahead of the
# interpreter's own PyAV on PYTHONPATH, so nothing in the environment is modified. For
# every version it prints the bundled libavcodec, what the build reports for
# stream.start_time and for the first frame's pts after a seek, and the test result.
# Thread pools are pinned to one thread: on a shared node, N parallel unpinned runs can
# each take 50 CPU-minutes instead of ten seconds.
set -u
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
VERSIONS=${*:-"12.3.0 13.1.0 14.2.0 15.1.0 16.1.0 17.1.0 18.1.0"}
WORK=${SWEEP_DIR:-/tmp/omniio-pyav-sweep}
mkdir -p "$WORK"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

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
lavc = ".".join(map(str, av.library_versions["libavcodec"][:2]))
print(f"libavcodec {lavc} | first packet pts {first} | stream.start_time {s.start_time} | "
      f"after seek: packet pts {head[0]}, frame pts {head[1]}, {head[2]} samples")
'
for v in $VERSIONS; do
    dir="$WORK/av-$v"
    if [ ! -d "$dir/av" ]; then
        $PY -m pip install -q --no-deps --only-binary=:all: --target "$dir" "av==$v" \
            || { echo "av $v | no wheel for this platform/interpreter"; continue; }
    fi
    export PYTHONPATH="$dir${PYTHONPATH:+:$PYTHONPATH}"
    info=$($PY -c "$probe" 2>/dev/null | tail -1)
    result=$($PY -m pytest tests/test_webm_opus_windows.py -o addopts= -q -p no:cacheprovider 2>&1 | tail -1)
    echo "av $v | $info | $result"
done
