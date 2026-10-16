"""lhotse corpora as omniio archives.

    from omniio import lhotse as olh

    olh.from_lhotse("ami", recordings="ami-ihm_recordings_dev.jsonl.gz",
                    supervisions="ami-ihm_supervisions_dev.jsonl.gz", name="ihm_dev")

    reader = olh.SegmentReader("ami/audio", "ami/supervisions_ihm_dev")
    seg = reader.segment("IB4010-0-0")      # index row + text/alignment + audio

The converter needs lhotse (``pip install 'omniio[lhotse]'``); the reader does
not.  See :mod:`omniio.lhotse.convert` for the archive layout.
"""

from omniio.tools.lhotse.convert import from_lhotse
from omniio.tools.lhotse.read import SegmentReader, windows

__all__ = ["from_lhotse", "SegmentReader", "windows"]
