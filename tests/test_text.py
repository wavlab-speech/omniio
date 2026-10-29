"""Tests for text reading and writing functionality."""

import pytest
import zstandard as zstd

from omniio.text.read import _decompress_blob, text_read_local
from omniio.text.write import (
    _compress_text_data,
    _decompress_text_data,
    text_write,
)
from omniio.definitions import TextRead


class TestTextCompression:
    """Test text compression/decompression."""

    def test_compress_text_string(self):
        """Test compressing text string."""
        text = "This is a test string for compression." * 10
        compressed, original_size = _compress_text_data(text)

        assert original_size == len(text.encode('utf-8'))
        assert len(compressed) < original_size
        assert isinstance(compressed, bytes)

    def test_compress_text_bytes(self):
        """Test compressing text bytes."""
        text_bytes = b"This is a test string for compression." * 10
        compressed, original_size = _compress_text_data(text_bytes)

        assert original_size == len(text_bytes)
        assert len(compressed) < original_size

    def test_compress_with_different_levels(self):
        """Test compression with different levels."""
        text = "This is a test string." * 100

        compressed_low, _ = _compress_text_data(text, compression_level=1)
        compressed_high, _ = _compress_text_data(text, compression_level=19)

        # Higher compression should result in smaller size
        assert len(compressed_high) <= len(compressed_low)

    def test_decompress_text_data(self):
        """Test decompressing text data."""
        original_text = "This is a test string for compression." * 10
        compressed, _ = _compress_text_data(original_text)

        decompressed = _decompress_text_data(compressed)

        assert decompressed == original_text

    def test_compress_decompress_roundtrip(self):
        """Test compression/decompression roundtrip."""
        original = "Hello, World! " * 50 + "Testing 123. " * 30

        compressed, original_size = _compress_text_data(original)
        decompressed = _decompress_text_data(compressed)

        assert decompressed == original
        assert len(original.encode('utf-8')) == original_size


class TestTextWrite:
    """Test text writing functionality."""

    def test_write_from_file(self, sample_text):
        """Test writing text from file."""
        text_path, text_content = sample_text

        raw_bytes, metadata = text_write(
            str(text_path),
            "text_001",
            is_path=True,
            compression_level=3
        )

        assert isinstance(raw_bytes, bytes)
        assert metadata['original_path'] == str(text_path)
        assert metadata['original_size'] > 0
        assert metadata['compressed_size'] == len(raw_bytes)
        assert metadata['compressed_size'] < metadata['original_size']

    def test_write_from_string(self):
        """Test writing text from string."""
        text_content = "This is a direct string input." * 20

        raw_bytes, metadata = text_write(
            text_content,
            "text_002",
            is_path=False,
            compression_level=5
        )

        assert isinstance(raw_bytes, bytes)
        assert metadata['original_path'] == ""
        assert metadata['original_size'] == len(text_content.encode('utf-8'))
        assert metadata['compressed_size'] == len(raw_bytes)

    def test_write_empty_string(self):
        """Test writing empty string."""
        raw_bytes, metadata = text_write(
            "",
            "text_003",
            is_path=False,
            compression_level=3
        )

        assert metadata['original_size'] == 0
        assert len(raw_bytes) > 0  # Zstd header

    def test_write_unicode_text(self):
        """Test writing unicode text."""
        text = "Hello 世界! Привет мир! مرحبا بالعالم!" * 10

        raw_bytes, metadata = text_write(
            text,
            "text_004",
            is_path=False,
            compression_level=3
        )

        # Verify roundtrip
        decompressed = _decompress_text_data(raw_bytes)
        assert decompressed == text


class TestTextRead:
    """Test text reading functionality."""

    def test_read_compressed_text(self, temp_dir, sample_text):
        """Test reading compressed text from archive."""
        text_path, text_content = sample_text

        # Create archive with compressed text
        raw_bytes, _ = text_write(
            str(text_path),
            "text_001",
            is_path=True,
            compression_level=3
        )

        archive_path = temp_dir / "text_archive.bin"
        with open(archive_path, 'wb') as f:
            f.write(raw_bytes)

        # Read back
        result = text_read_local(archive_path, 0, len(raw_bytes))

        assert isinstance(result, TextRead)
        assert result.file_type == "text"
        assert result.modality == "text"
        assert result.text == text_content

    def test_read_with_offset(self, temp_dir):
        """Test reading from archive with offset."""
        text1 = "First text content." * 10
        text2 = "Second text content." * 10

        # Compress both texts
        compressed1, _ = _compress_text_data(text1)
        compressed2, _ = _compress_text_data(text2)

        # Create archive
        archive_path = temp_dir / "multi_text_archive.bin"
        with open(archive_path, 'wb') as f:
            f.write(compressed1)
            f.write(compressed2)

        # Read second text
        result = text_read_local(
            archive_path,
            start_offset=len(compressed1),
            file_size=len(compressed2)
        )

        assert result.text == text2

    def test_decompress_blob(self, compressed_text):
        """Test _decompress_blob helper."""
        compressed, original_text = compressed_text

        decompressed = _decompress_blob(compressed)

        assert decompressed == original_text
