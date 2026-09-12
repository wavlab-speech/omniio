"""Opening Kaldi-style extended filenames: plain paths, pipes and ``-``."""

import gzip
import io
import os
import subprocess
import sys


class _PipeStream(io.IOBase):
    """File-like wrapper around one end of a subprocess pipe.

    Closing waits for the child and raises if it exited non-zero, which is what
    makes a failing ``sox``/``ffmpeg`` in an scp entry a hard error rather than
    a silently truncated read.
    """

    def __init__(self, proc, stream, command):
        self._proc = proc
        self._stream = stream
        self._command = command

    def __getattr__(self, name):
        return getattr(self._stream, name)

    def readable(self):
        return self._stream.readable()

    def writable(self):
        return self._stream.writable()

    def seekable(self):
        return False

    def read(self, *args):
        return self._stream.read(*args)

    def write(self, data):
        return self._stream.write(data)

    def close(self):
        if self._stream.closed:
            return
        self._stream.close()
        returncode = self._proc.wait()
        if returncode != 0:
            raise IOError("Command exited with status {}: {}".format(returncode, self._command))

    @property
    def closed(self):
        return self._stream.closed

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.close()
        else:
            # Don't mask the original exception with a non-zero exit status.
            self._stream.close()
            self._proc.wait()


class _NonClosing(io.IOBase):
    """Wrap a caller-owned stream so ``with`` blocks don't close it."""

    def __init__(self, stream):
        self._stream = stream

    def __getattr__(self, name):
        return getattr(self._stream, name)

    def read(self, *args):
        return self._stream.read(*args)

    def write(self, data):
        return self._stream.write(data)

    def close(self):
        pass

    @property
    def closed(self):
        return self._stream.closed

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass


def open_like_kaldi(name, mode="r"):
    """Open a Kaldi extended filename.

    Recognised forms:

    ``-``           standard input / standard output
    ``some cmd |``  read the standard output of ``some cmd``
    ``| some cmd``  write to the standard input of ``some cmd``
    ``*.gz``        transparently gzip-compressed
    anything else   a plain path, opened with :func:`open`

    An already-open file object is returned wrapped so that leaving a ``with``
    block does not close it.
    """
    if not isinstance(name, str):
        return _NonClosing(name)

    binary = "b" in mode
    stripped = name.strip()

    if stripped == "-":
        if "r" in mode:
            return _NonClosing(sys.stdin.buffer if binary else sys.stdin)
        return _NonClosing(sys.stdout.buffer if binary else sys.stdout)

    if stripped.endswith("|"):
        command = stripped[:-1]
        proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
        stream = proc.stdout if binary else io.TextIOWrapper(proc.stdout)
        return _PipeStream(proc, stream, command)

    if stripped.startswith("|"):
        command = stripped[1:]
        proc = subprocess.Popen(command, shell=True, stdin=subprocess.PIPE)
        stream = proc.stdin if binary else io.TextIOWrapper(proc.stdin)
        return _PipeStream(proc, stream, command)

    if stripped.endswith(".gz"):
        return gzip.open(stripped, mode if binary else mode + "t")

    directory = os.path.dirname(stripped)
    if directory and ("w" in mode or "a" in mode):
        os.makedirs(directory, exist_ok=True)
    return open(stripped, mode)


def parse_extended_filename(name):
    """Split ``file.ark:1234`` into ``('file.ark', 1234)``.

    ``(name, None)`` is returned when there is no offset, including for pipes
    and for Windows-style drive letters.
    """
    if not isinstance(name, str):
        return name, None
    if name.strip().endswith("|"):
        return name, None
    head, sep, tail = name.rpartition(":")
    if sep and tail.isdigit():
        return head, int(tail)
    return name, None
