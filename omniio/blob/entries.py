class Entries:
    """What a writer returns when one item becomes *several* archive entries.

    Wraps an iterable of ``(entry_id, raw_bytes, metadata)``; the Blob writer unrolls it
    into the worker's shard, entry by entry, checking each id against the archive. A
    slice of an .h5ad file is the first user (``omniio.cell``): the item is the slice,
    the entries are its cells.
    """

    def __init__(self, iterable):
        self._iterable = iterable

    def __iter__(self):
        return iter(self._iterable)
