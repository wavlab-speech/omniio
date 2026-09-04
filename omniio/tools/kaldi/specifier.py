"""Parsing of Kaldi ``rspecifier`` / ``wspecifier`` strings."""

_WRITE_OPTIONS = {"ark", "scp", "t", "f"}
# Read options that only affect Kaldi's internal buffering/ordering and have no
# meaning here; they are accepted and ignored.
_READ_OPTIONS = {"o", "no", "s", "ns", "cs", "ncs", "p", "np", "bg"}


def _split(specifier, kind):
    if not isinstance(specifier, str):
        raise TypeError("{} must be a string, got {}".format(kind, type(specifier).__name__))
    if ":" not in specifier:
        raise ValueError(
            "Invalid {}: {!r} (expected e.g. 'ark:file.ark' or "
            "'ark,scp:file.ark,file.scp')".format(kind, specifier)
        )
    types, files = specifier.split(":", 1)
    return [t.strip() for t in types.split(",")], files


def parse_wspecifier(wspecifier):
    """Split a wspecifier into ``{'ark': path, 'scp': path, 't': ..., 'f': ...}``.

    ``t`` selects the text format and ``f`` requests a flush after every write;
    both map to ``None`` because they take no argument.
    """
    types, files = _split(wspecifier, "wspecifier")

    unknown = [t for t in types if t not in _WRITE_OPTIONS]
    if unknown:
        raise ValueError(
            "Unsupported wspecifier option(s) {} in {!r}. "
            "Supported: {}".format(unknown, wspecifier, sorted(_WRITE_OPTIONS))
        )
    if "ark" not in types:
        raise ValueError("wspecifier must contain 'ark': {!r}".format(wspecifier))

    # A pipe destination may itself contain commas, so only split off as many
    # fields as there are file-valued options.
    file_types = [t for t in types if t in ("ark", "scp")]
    parts = files.split(",", len(file_types) - 1)
    if len(parts) != len(file_types):
        raise ValueError(
            "wspecifier {!r} declares {} file(s) but {} were given".format(
                wspecifier, len(file_types), len(parts)
            )
        )

    retval = dict(zip(file_types, parts))
    for t in types:
        if t not in ("ark", "scp"):
            retval[t] = None
    return retval


def parse_rspecifier(rspecifier):
    """Split an rspecifier into ``(kind, path)`` where kind is ``ark`` or ``scp``."""
    types, files = _split(rspecifier, "rspecifier")

    allowed = {"ark", "scp", "t", "b"} | _READ_OPTIONS
    unknown = [t for t in types if t not in allowed]
    if unknown:
        raise ValueError("Unsupported rspecifier option(s) {} in {!r}".format(unknown, rspecifier))
    kinds = [t for t in types if t in ("ark", "scp")]
    if len(kinds) != 1:
        raise ValueError(
            "rspecifier must contain exactly one of 'ark'/'scp': {!r}".format(rspecifier)
        )
    return kinds[0], files


#: Every flag ``parse_specifier`` reports, so callers can index the result
#: without checking for the key first.
_SPECIFIER_FLAGS = ("t", "o", "p", "f", "s", "cs")


def parse_specifier(specifier):
    """Parse either an r- or a wspecifier into a fully populated dict.

    ``ark`` and ``scp`` hold the file each names or ``None``; every flag is
    present as a bool whether or not it was given::

        >>> parse_specifier("ark,scp:a.ark,b.scp") == {
        ...     "ark": "a.ark", "scp": "b.scp",
        ...     "t": False, "o": False, "p": False,
        ...     "f": False, "s": False, "cs": False,
        ... }
        True

    Prefer :func:`parse_rspecifier` or :func:`parse_wspecifier` in new code:
    they know which side they are on, so they can reject a specifier that
    cannot work rather than returning a dict with ``ark`` set to ``None``.
    """
    types, files = _split(specifier, "specifier")
    file_types = [t for t in types if t in ("ark", "scp")]
    if not file_types:
        raise ValueError("specifier must contain 'ark' or 'scp': {!r}".format(specifier))
    parts = files.split(",", len(file_types) - 1)
    if len(parts) != len(file_types):
        raise ValueError(
            "specifier {!r} declares {} file(s) but {} were given".format(
                specifier, len(file_types), len(parts)
            )
        )

    retval = {"ark": None, "scp": None}
    retval.update(dict(zip(file_types, parts)))
    for flag in _SPECIFIER_FLAGS:
        retval[flag] = flag in types
    return retval
