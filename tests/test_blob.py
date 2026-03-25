"""Tests for Blob class functionality."""

import pytest
import pyarrow as pa

from omniio.blob.blob import Blob


class TestBlobInit:
    """Test Blob initialization."""

    def test_create_new_blob(self, temp_dir):
        """Test creating a new blob archive."""
        archive_dir = temp_dir / "test_archive"

        blob = Blob(archive_dir=str(archive_dir), modality="audio")

        assert blob.modality == "audio"
        assert blob.archive_path.parent.exists()  # Archive directory exists
        assert blob.metadata_file.exists() or len(blob) == 0

    def test_create_blob_with_name(self, temp_dir):
        """Test creating blob with custom name."""
        archive_dir = temp_dir / "test_archive"

        blob = Blob(
            archive_dir=str(archive_dir),
            modality="text",
            name="custom_name"
        )

        assert blob.name == "custom_name"

    def test_invalid_modality(self, temp_dir):
        """Test creating blob with invalid modality raises error."""
        archive_dir = temp_dir / "test_archive"

        with pytest.raises(AssertionError):
            Blob(archive_dir=str(archive_dir), modality="invalid")

    def test_open_existing_blob(self, temp_dir, sample_audio_wav):
        """Test opening existing blob archive."""
        archive_dir = temp_dir / "test_archive"

        # Create and populate blob
        blob1 = Blob(archive_dir=str(archive_dir), modality="audio")
        audio_path, _, _ = sample_audio_wav
        blob1.append(
            items=[str(audio_path)],
            ids=["audio_001"],
            num_workers=0,
            target_format="wav"
        )

        # Reopen blob
        blob2 = Blob(archive_dir=str(archive_dir), modality="audio")

        assert len(blob2) == 1
        assert blob2.data is not None


class TestBlobAppend:
    """Test Blob append operations."""

    def test_append_single_audio(self, temp_dir, sample_audio_wav):
        """Test appending single audio file."""
        archive_dir = temp_dir / "test_archive"
        blob = Blob(archive_dir=str(archive_dir), modality="audio")

        audio_path, _, _ = sample_audio_wav
        blob.append(
            items=[str(audio_path)],
            ids=["audio_001"],
            num_workers=0,
            target_format="flac",
            target_bit_depth=16
        )

        assert len(blob) == 1
        metadata = blob.get_metadata()
        assert "audio_001" in metadata.column("id").to_pylist()

    def test_append_multiple_audio(self, temp_dir, sample_audio_wav, sample_audio_flac):
        """Test appending multiple audio files."""
        archive_dir = temp_dir / "test_archive"
        blob = Blob(archive_dir=str(archive_dir), modality="audio")

        wav_path, _, _ = sample_audio_wav
        flac_path, _, _ = sample_audio_flac

        blob.append(
            items=[str(wav_path), str(flac_path)],
            ids=["audio_001", "audio_002"],
            num_workers=0,
            target_format="flac"
        )

        assert len(blob) == 2
        metadata = blob.get_metadata()
        ids = metadata.column("id").to_pylist()
        assert "audio_001" in ids
        assert "audio_002" in ids

    def test_append_with_parallel_workers(self, temp_dir, sample_audio_wav, sample_audio_flac):
        """Test appending with multiple workers."""
        archive_dir = temp_dir / "test_archive"
        blob = Blob(archive_dir=str(archive_dir), modality="audio")

        wav_path, _, _ = sample_audio_wav
        flac_path, _, _ = sample_audio_flac

        blob.append(
            items=[str(wav_path), str(flac_path)],
            ids=["audio_001", "audio_002"],
            num_workers=2,
            target_format="wav"
        )

        assert len(blob) == 2

    def test_append_without_ids(self, temp_dir, sample_audio_wav):
        """Test appending without explicit IDs."""
        archive_dir = temp_dir / "test_archive"
        blob = Blob(archive_dir=str(archive_dir), modality="audio")

        audio_path, _, _ = sample_audio_wav

        blob.append(
            items=[str(audio_path), str(audio_path)],
            num_workers=0,
            target_format="wav"
        )

        assert len(blob) == 2
        metadata = blob.get_metadata()
        ids = metadata.column("id").to_pylist()
        assert f"{blob.name}_0" in ids
        assert f"{blob.name}_1" in ids

    def test_append_duplicate_id_error(self, temp_dir, sample_audio_wav):
        """Test appending duplicate ID without overwrite raises error."""
        archive_dir = temp_dir / "test_archive"
        blob = Blob(archive_dir=str(archive_dir), modality="audio")

        audio_path, _, _ = sample_audio_wav

        # First append
        blob.append(
            items=[str(audio_path)],
            ids=["audio_001"],
            num_workers=0,
            target_format="wav"
        )

        # Second append with same ID should fail
        with pytest.raises(RuntimeError, match="Duplicate id"):
            blob.append(
                items=[str(audio_path)],
                ids=["audio_001"],
                num_workers=0,
                overwrite=False,
                target_format="wav"
            )

    def test_append_duplicate_id_overwrite(self, temp_dir, sample_audio_wav):
        """Test appending duplicate ID with overwrite skips."""
        archive_dir = temp_dir / "test_archive"
        blob = Blob(archive_dir=str(archive_dir), modality="audio")

        audio_path, _, _ = sample_audio_wav

        # First append
        blob.append(
            items=[str(audio_path)],
            ids=["audio_001"],
            num_workers=0,
            target_format="wav"
        )

        initial_len = len(blob)

        # Second append with same ID and overwrite=True
        blob.append(
            items=[str(audio_path)],
            ids=["audio_001"],
            num_workers=0,
            overwrite=True,
            target_format="wav"
        )

        # Length should not change
        assert len(blob) == initial_len

    def test_append_text(self, temp_dir, sample_text):
        """Test appending text files."""
        archive_dir = temp_dir / "test_archive"
        blob = Blob(archive_dir=str(archive_dir), modality="text")

        text_path, _ = sample_text

        blob.append(
            items=[str(text_path)],
            ids=["text_001"],
            num_workers=0,
            is_path=True,
            compression_level=3
        )

        assert len(blob) == 1


class TestBlobMetadata:
    """Test Blob metadata operations."""

    def test_get_metadata(self, temp_dir, sample_audio_wav):
        """Test getting metadata table."""
        archive_dir = temp_dir / "test_archive"
        blob = Blob(archive_dir=str(archive_dir), modality="audio")

        audio_path, _, _ = sample_audio_wav
        blob.append(
            items=[str(audio_path)],
            ids=["audio_001"],
            num_workers=0,
            target_format="wav"
        )

        metadata = blob.get_metadata()

        assert isinstance(metadata, pa.Table)
        assert "id" in metadata.column_names
        assert "start_byte" in metadata.column_names
        assert "end_byte" in metadata.column_names
        assert "bin_index" in metadata.column_names

    def test_existing_ids(self, temp_dir, sample_audio_wav):
        """Test _existing_ids returns correct set."""
        archive_dir = temp_dir / "test_archive"
        blob = Blob(archive_dir=str(archive_dir), modality="audio")

        audio_path, _, _ = sample_audio_wav
        blob.append(
            items=[str(audio_path)],
            ids=["audio_001"],
            num_workers=0,
            target_format="wav"
        )

        existing = blob._existing_ids()

        assert "audio_001" in existing


class TestBlobUtilities:
    """Test Blob utility methods."""

    def test_summary_empty(self, temp_dir, capsys):
        """Test summary for empty blob."""
        archive_dir = temp_dir / "test_archive"
        blob = Blob(archive_dir=str(archive_dir), modality="audio")

        blob.summary()

        captured = capsys.readouterr()
        assert "empty" in captured.out.lower()

    def test_summary_with_data(self, temp_dir, sample_audio_wav, capsys):
        """Test summary with data."""
        archive_dir = temp_dir / "test_archive"
        blob = Blob(archive_dir=str(archive_dir), modality="audio")

        audio_path, _, _ = sample_audio_wav
        blob.append(
            items=[str(audio_path)],
            ids=["audio_001"],
            num_workers=0,
            target_format="wav"
        )

        blob.summary()

        captured = capsys.readouterr()
        assert "Total files: 1" in captured.out

    def test_clear_without_confirm(self, temp_dir):
        """Test clear without confirmation raises error."""
        archive_dir = temp_dir / "test_archive"
        blob = Blob(archive_dir=str(archive_dir), modality="audio")

        with pytest.raises(ValueError, match="requires confirmation"):
            blob.clear(confirm=False)

    def test_clear_with_confirm(self, temp_dir, sample_audio_wav):
        """Test clearing blob with confirmation."""
        archive_dir = temp_dir / "test_archive"
        blob = Blob(archive_dir=str(archive_dir), modality="audio")

        audio_path, _, _ = sample_audio_wav
        blob.append(
            items=[str(audio_path)],
            ids=["audio_001"],
            num_workers=0,
            target_format="wav"
        )

        assert len(blob) == 1

        blob.clear(confirm=True)

        assert len(blob) == 0
        assert not blob.metadata_file.exists()

    def test_len(self, temp_dir, sample_audio_wav):
        """Test __len__ method."""
        archive_dir = temp_dir / "test_archive"
        blob = Blob(archive_dir=str(archive_dir), modality="audio")

        assert len(blob) == 0

        audio_path, _, _ = sample_audio_wav
        blob.append(
            items=[str(audio_path)],
            ids=["audio_001"],
            num_workers=0,
            target_format="wav"
        )

        assert len(blob) == 1
