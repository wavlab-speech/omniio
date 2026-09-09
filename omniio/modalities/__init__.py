"""The per-modality readers and writers: ``audio``, ``video``, ``text``, ``image``,
``midi``. Each subpackage exposes ``<modality>_read`` / ``<modality>_write`` (see
``omniio.interface`` for the public entry points).

The public import paths stay the short ones (``omniio.audio``, ``omniio.midi``, ...)
through the aliases in :mod:`omniio` (``_ALIASES``); this directory is where the code lives.
"""
