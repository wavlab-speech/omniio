"""Efficient multimedia I/O for binary archive blobs.

The source tree groups modules by what they are (``omniio/tools/`` for
format-specific helpers, and so on), but that layout is an implementation
detail.  The stable, public import paths are the short ones::

    from omniio import kaldi
    import omniio.kaldi
    from omniio.kaldi import ReadHelper

All three work regardless of where the code actually sits, so subpackages can
be reorganised without breaking importers.  To relocate one, move it and add a
line to :data:`_ALIASES` -- nothing else has to change.

The aliases resolve lazily: importing :mod:`omniio` does not pull in any of the
aliased subpackages or their dependencies.
"""

import importlib
import importlib.abc
import importlib.util
import sys

#: Public import path -> where the module actually lives.
_ALIASES = {
    "omniio.kaldi": "omniio.tools.kaldi",
    # e.g. when the modality packages move under omniio/modalities/:
    #   "omniio.audio": "omniio.modalities.audio",
}


def _resolve(fullname):
    """Map a public module name onto its real one, or ``None`` if unaliased."""
    for public, private in _ALIASES.items():
        if fullname == public:
            return private
        if fullname.startswith(public + "."):
            return private + fullname[len(public) :]
    return None


class _AliasLoader(importlib.abc.Loader):
    """Bind an already-importable module under a second name."""

    def __init__(self, target):
        self._target = target

    def create_module(self, spec):
        # The module is executed once, under its real name; both names then
        # refer to the same object, so patching or reloading either is
        # visible through the other.
        return importlib.import_module(self._target)

    def exec_module(self, module):
        pass


class _AliasFinder(importlib.abc.MetaPathFinder):
    """Resolve the public ``omniio.*`` paths in :data:`_ALIASES`."""

    def find_spec(self, fullname, path=None, target=None):
        real_name = _resolve(fullname)
        if real_name is None:
            return None
        # This finder runs before the normal one, so it must not claim a name
        # whose target does not exist: an _ALIASES entry added before its
        # package is actually moved would otherwise shadow the package that is
        # still there.
        try:
            if importlib.util.find_spec(real_name) is None:
                return None
        except (ImportError, AttributeError, ValueError):
            return None
        return importlib.util.spec_from_loader(fullname, _AliasLoader(real_name))


# Prepended, so that a submodule such as ``omniio.kaldi.compression`` is
# resolved through the alias too.  Left to the normal path finder it would be
# loaded a second time -- once under each name, with duplicate module state --
# because the aliased parent's ``__path__`` still points at the real directory.
# This is safe because the finder claims only the names in ``_ALIASES``.
if not any(isinstance(finder, _AliasFinder) for finder in sys.meta_path):
    sys.meta_path.insert(0, _AliasFinder())


def __getattr__(name):
    """Support ``from omniio import kaldi`` without importing it eagerly."""
    if _resolve("{}.{}".format(__name__, name)) is None:
        raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))
    module = importlib.import_module("{}.{}".format(__name__, name))
    globals()[name] = module
    return module


def __dir__():
    aliased = (name.split(".", 1)[1] for name in _ALIASES if name.count(".") == 1)
    return sorted(set(globals()) | set(aliased))
