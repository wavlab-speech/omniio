"""The modality packages live under omniio/modalities/; the short public paths keep
working and resolve to the SAME module objects (no double import)."""
import importlib
import sys

import pytest

MODALITIES = ("audio", "video", "text", "image", "midi")


@pytest.mark.parametrize("name", MODALITIES)
def test_public_path_aliases_the_modalities_package(name):
    public = importlib.import_module(f"omniio.{name}")
    real = importlib.import_module(f"omniio.modalities.{name}")
    assert public is real
    assert sys.modules[f"omniio.{name}"] is sys.modules[f"omniio.modalities.{name}"]


@pytest.mark.parametrize("name", MODALITIES)
def test_submodules_resolve_through_the_alias_without_a_second_copy(name):
    pub_read = importlib.import_module(f"omniio.{name}.read")
    real_read = importlib.import_module(f"omniio.modalities.{name}.read")
    assert pub_read is real_read
    pub_write = importlib.import_module(f"omniio.{name}.write")
    assert pub_write is importlib.import_module(f"omniio.modalities.{name}.write")


def test_from_import_and_attribute_access():
    import omniio
    from omniio import audio, midi                   # lazy attribute access on the package
    from omniio.midi.read import midi_read_local     # submodule through the public path
    from omniio.audio.read import audio_read_local
    assert midi.read.midi_read_local is midi_read_local and audio.read.audio_read_local is audio_read_local
    assert "midi" in dir(omniio) and "audio" in dir(omniio)
    from omniio import interface                     # the documented entry points still resolve
    assert callable(interface.midi_read) and callable(interface.audio_read)


def test_package_imports_without_pretty_midi():
    """pretty_midi is an extra: the non-MIDI entry points must import without it, and
    the MIDI ones must fail with an ImportError that names the extra."""
    import subprocess
    import sys

    code = """
import sys
sys.modules["pretty_midi"] = None           # make `import pretty_midi` raise ImportError
import omniio.interface, omniio.blob.blob    # the modality registries import eagerly
import omniio.midi, omniio.midi.read, omniio.midi.write
from omniio.midi.common import midi_from_bytes
try:
    midi_from_bytes(b"MThd" + bytes(10))
except ImportError as e:
    assert "omniio[midi]" in str(e), str(e)
else:
    raise AssertionError("expected ImportError")
print("ok")
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"
