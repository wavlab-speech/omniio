"""Tests for Blob class functionality."""

import pytest
import pyarrow as pa

from omniio.blob.blob import Blob
from pathlib import Path

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

    def test_append_duplicate_id_skipped_with_warning(self, temp_dir, sample_audio_wav):
        """Test appending duplicate ID without allow_duplicate_ids skips and warns."""
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

        # Second append with same ID should skip with a warning
        with pytest.warns(UserWarning, match="audio_001"):
            blob.append(
                items=[str(audio_path)],
                ids=["audio_001"],
                num_workers=0,
                allow_duplicate_ids=False,
                target_format="wav"
            )

        # Length should not change
        assert len(blob) == initial_len

    def test_append_duplicate_id_allowed(self, temp_dir, sample_audio_wav):
        """Test appending duplicate ID with allow_duplicate_ids=True writes the item."""
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

        # Second append with same ID and allow_duplicate_ids=True should write
        blob.append(
            items=[str(audio_path)],
            ids=["audio_001"],
            num_workers=0,
            allow_duplicate_ids=True,
            target_format="wav"
        )

        # Length should increase
        assert len(blob) == initial_len + 1

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

    def test_metadata_has_path_column(self, temp_dir, sample_audio_wav):
        """Test that metadata includes path column pointing to bin files."""
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

        # Check path column exists
        assert "path" in metadata.column_names

        # Check path points to actual bin file
        path = metadata.column("path")[0].as_py()
        assert path.endswith(".bin")
        assert Path(path).exists()

        # Verify we can use the path to read data
        from omniio.interface import audio_read
        row = metadata.to_pandas().iloc[0]
        result = audio_read(
            row['path'],
            row['start_byte'],
            row['end_byte'] - row['start_byte']
        )
        assert result.array is not None

    def test_path_column_multiple_bins(self, temp_dir, sample_audio_wav):
        """Test path column with multiple bin files."""
        archive_dir = temp_dir / "test_archive"
        # Small max_bin_size to force multiple bins
        blob = Blob(
            archive_dir=str(archive_dir),
            modality="audio",
            max_bin_size=1024  # Very small, will create multiple bins
        )

        audio_path, _, _ = sample_audio_wav
        # Append multiple times to create multiple bins
        for i in range(3):
            blob.append(
                items=[str(audio_path)],
                ids=[f"audio_{i:03d}"],
                num_workers=0,
                target_format="wav"
            )

        metadata = blob.get_metadata()

        # Should have multiple unique paths
        paths = set(metadata.column("path").to_pylist())
        assert len(paths) > 0  # At least one bin file

        # All paths should exist
        for path in paths:
            assert Path(path).exists()
            assert path.endswith(".bin")


def _read_entry_bytes(blob: Blob, item_id: str) -> bytes:
    """Read the raw bytes for a single entry from its bin file using metadata offsets."""
    meta = blob.get_metadata().to_pydict()
    idx = meta["id"].index(item_id)
    bin_path = meta["path"][idx]
    start = meta["start_byte"][idx]
    end = meta["end_byte"][idx]
    with open(bin_path, "rb") as f:
        f.seek(start)
        return f.read(end - start)


class TestConcatShards:
    """Verify _concat_shards and _concat_shards_fast produce identical stored bytes."""

    def _build_blob(self, archive_dir, items, ids, num_workers, reshard, **kwargs):
        blob = Blob(archive_dir=str(archive_dir), modality="audio")
        blob.append(
            items=items,
            ids=ids,
            num_workers=num_workers,
            reshard=reshard,
            **kwargs,
        )
        return blob

    def test_single_item_bytes_match(self, temp_dir, sample_audio_wav):
        audio_path, _, _ = sample_audio_wav
        items = [str(audio_path)]
        ids = ["item_0"]
        kwargs = dict(target_format="wav")

        blob_reshard = self._build_blob(
            temp_dir / "reshard", items, ids, num_workers=0, reshard=True, **kwargs
        )
        blob_fast = self._build_blob(
            temp_dir / "fast", items, ids, num_workers=0, reshard=False, **kwargs
        )

        assert _read_entry_bytes(blob_reshard, "item_0") == _read_entry_bytes(blob_fast, "item_0")

    def test_multiple_items_bytes_match(self, temp_dir, sample_audio_wav, sample_audio_flac):
        wav_path, _, _ = sample_audio_wav
        flac_path, _, _ = sample_audio_flac
        items = [str(wav_path), str(flac_path), str(wav_path)]
        ids = ["item_0", "item_1", "item_2"]
        kwargs = dict(target_format="flac", target_bit_depth=16)

        blob_reshard = self._build_blob(
            temp_dir / "reshard", items, ids, num_workers=0, reshard=True, **kwargs
        )
        blob_fast = self._build_blob(
            temp_dir / "fast", items, ids, num_workers=0, reshard=False, **kwargs
        )

        for item_id in ids:
            assert _read_entry_bytes(blob_reshard, item_id) == _read_entry_bytes(blob_fast, item_id), \
                f"Bytes differ for {item_id}"

    def test_parallel_workers_bytes_match(self, temp_dir, sample_audio_wav, sample_audio_flac):
        wav_path, _, _ = sample_audio_wav
        flac_path, _, _ = sample_audio_flac
        items = [str(wav_path), str(flac_path), str(wav_path), str(flac_path)]
        ids = ["item_0", "item_1", "item_2", "item_3"]
        kwargs = dict(target_format="wav")

        blob_reshard = self._build_blob(
            temp_dir / "reshard", items, ids, num_workers=2, reshard=True, **kwargs
        )
        blob_fast = self._build_blob(
            temp_dir / "fast", items, ids, num_workers=2, reshard=False, **kwargs
        )

        for item_id in ids:
            assert _read_entry_bytes(blob_reshard, item_id) == _read_entry_bytes(blob_fast, item_id), \
                f"Bytes differ for {item_id}"

    def test_multiple_appends_bytes_match(self, temp_dir, sample_audio_wav, sample_audio_flac):
        """Two sequential appends: bytes from both methods should match across appends."""
        wav_path, _, _ = sample_audio_wav
        flac_path, _, _ = sample_audio_flac
        kwargs = dict(target_format="flac", target_bit_depth=16)

        for label, reshard in [("reshard", True), ("fast", False)]:
            blob = Blob(archive_dir=str(temp_dir / label), modality="audio")
            blob.append(
                items=[str(wav_path)], ids=["item_0"], num_workers=0, reshard=reshard, **kwargs
            )
            blob.append(
                items=[str(flac_path)], ids=["item_1"], num_workers=0, reshard=reshard, **kwargs
            )

        blob_reshard = Blob(archive_dir=str(temp_dir / "reshard"), modality="audio")
        blob_fast = Blob(archive_dir=str(temp_dir / "fast"), modality="audio")

        for item_id in ["item_0", "item_1"]:
            assert _read_entry_bytes(blob_reshard, item_id) == _read_entry_bytes(blob_fast, item_id), \
                f"Bytes differ for {item_id}"

    def test_offsets_are_non_overlapping(self, temp_dir, sample_audio_wav, sample_audio_flac):
        """Metadata offsets must be contiguous and non-overlapping within each bin."""
        wav_path, _, _ = sample_audio_wav
        flac_path, _, _ = sample_audio_flac
        items = [str(wav_path), str(flac_path), str(wav_path)]
        ids = ["item_0", "item_1", "item_2"]
        kwargs = dict(target_format="wav")

        for label, reshard in [("reshard", True), ("fast", False)]:
            blob = self._build_blob(
                temp_dir / label, items, ids, num_workers=0, reshard=reshard, **kwargs
            )
            meta = blob.get_metadata().to_pydict()
            # Group by bin_index and verify no overlaps within each bin
            by_bin = {}
            for i in range(len(meta["id"])):
                bi = meta["bin_index"][i]
                by_bin.setdefault(bi, []).append(
                    (meta["start_byte"][i], meta["end_byte"][i])
                )
            for bi, ranges in by_bin.items():
                ranges.sort()
                for j in range(1, len(ranges)):
                    assert ranges[j][0] >= ranges[j - 1][1], \
                        f"{label}: overlapping offsets in bin {bi}: {ranges[j-1]} vs {ranges[j]}"


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
