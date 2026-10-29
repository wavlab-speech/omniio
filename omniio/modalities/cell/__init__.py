"""Per-cell sparse feature rows (RNA counts, ATAC peaks, protein markers, dosages): one
entry per cell over a fixed panel of columns.

A codec, like ``discrete``: the entry is ``header + delta-varint column indices + values``
(``common.py``) and the library does not know what the columns are. The caller owns the
panel (gene ids, peak coordinates) the way it owns a tokenizer's vocabulary for a
``discrete`` archive; ``examples/cell_h5ad.py`` shows ingesting ``.h5ad`` files onto one
panel through ``Blob.append`` and mapping column numbers back to gene names on read.
``cell_write`` takes one row, or a `CellRows` item yielding many; ``cell_read`` /
``cell_read_batch`` fetch one byte range per cell.
"""
