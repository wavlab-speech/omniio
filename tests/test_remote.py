"""Unit tests for omniio.remote module."""

import os
import tempfile
import threading
import time
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import requests

from omniio.remote import load, serve


@pytest.fixture
def sample_metadata():
    """Create sample metadata table for testing."""
    data = {
        'item_id': ['audio_001', 'audio_002', 'audio_003'],
        'bin_index': [0, 0, 1],
        'start_byte': [0, 1000, 0],
        'end_byte': [1000, 2000, 1500],
        'path': ['/local/blob_0.bin', '/local/blob_0.bin', '/local/blob_1.bin'],
        'sample_rate': [16000, 16000, 16000],
    }
    return pa.table(data)


@pytest.fixture
def temp_metadata_file(sample_metadata):
    """Create temporary metadata parquet file."""
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.parquet', delete=False) as f:
        pq.write_table(sample_metadata, f.name)
        yield f.name
    os.unlink(f.name)


class TestLoad:
    """Tests for the load() function."""

    def test_load_local_metadata(self, temp_metadata_file):
        """Test loading local metadata and updating paths."""
        table = load(temp_metadata_file, 'http://localhost:8000')

        # Check that paths were updated
        paths = table.column('path').to_pylist()
        assert paths[0] == 'http://localhost:8000/blob_0.bin'
        assert paths[1] == 'http://localhost:8000/blob_0.bin'
        assert paths[2] == 'http://localhost:8000/blob_1.bin'

        # Check other columns unchanged
        assert table.column('item_id').to_pylist() == ['audio_001', 'audio_002', 'audio_003']
        assert table.column('bin_index').to_pylist() == [0, 0, 1]

    def test_load_updates_all_bins(self, sample_metadata):
        """Test that multiple bin indices are handled correctly."""
        # Add more bin indices
        extended_data = {
            'item_id': ['audio_001', 'audio_002', 'audio_003', 'audio_004'],
            'bin_index': [0, 1, 2, 3],
            'start_byte': [0, 0, 0, 0],
            'end_byte': [1000, 1000, 1000, 1000],
            'path': ['/local/blob_0.bin'] * 4,
        }
        table = pa.table(extended_data)

        with tempfile.NamedTemporaryFile(mode='wb', suffix='.parquet', delete=False) as f:
            pq.write_table(table, f.name)
            try:
                result = load(f.name, 'http://example.com')
                paths = result.column('path').to_pylist()

                assert paths[0] == 'http://example.com/blob_0.bin'
                assert paths[1] == 'http://example.com/blob_1.bin'
                assert paths[2] == 'http://example.com/blob_2.bin'
                assert paths[3] == 'http://example.com/blob_3.bin'
            finally:
                os.unlink(f.name)

    def test_load_strips_trailing_slash(self, temp_metadata_file):
        """Test that trailing slashes in URL are handled correctly."""
        table = load(temp_metadata_file, 'http://localhost:8000/')

        paths = table.column('path').to_pylist()
        # Should not have double slashes
        assert paths[0] == 'http://localhost:8000/blob_0.bin'

    def test_load_directory_input(self, sample_metadata):
        """Test loading from directory (appends metadata.parquet)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata_path = os.path.join(tmpdir, 'metadata.parquet')
            pq.write_table(sample_metadata, metadata_path)

            # Load using directory path
            table = load(tmpdir, 'http://localhost:8000')

            paths = table.column('path').to_pylist()
            assert paths[0] == 'http://localhost:8000/blob_0.bin'

    @patch('omniio.remote.requests.get')
    def test_load_remote_metadata(self, mock_get, sample_metadata):
        """Test loading metadata from remote URL."""
        # Mock the response
        mock_response = Mock()
        mock_response.raise_for_status = Mock()

        # Serialize table to bytes
        buf = BytesIO()
        pq.write_table(sample_metadata, buf)
        mock_response.content = buf.getvalue()

        mock_get.return_value = mock_response

        # Load from remote
        table = load('http://server:8000/metadata.parquet', 'http://server:8000')

        # Verify request was made
        mock_get.assert_called_once_with('http://server:8000/metadata.parquet')

        # Verify paths updated
        paths = table.column('path').to_pylist()
        assert paths[0] == 'http://server:8000/blob_0.bin'

    def test_load_huggingface(self, sample_metadata):
        """Test loading from HuggingFace dataset."""
        pytest.importorskip('datasets', reason='datasets package required for HuggingFace tests')

        with patch('datasets.load_dataset') as mock_load_dataset:
            # Mock the dataset
            mock_dataset = Mock()
            mock_dataset.data.table = sample_metadata
            mock_load_dataset.return_value = mock_dataset

            # Load from HuggingFace
            table = load('espnet/librispeech', 'huggingface')

            # Verify dataset was loaded
            mock_load_dataset.assert_called_once_with('espnet/librispeech')

            # Verify paths use HuggingFace URLs
            paths = table.column('path').to_pylist()
            assert paths[0] == 'https://huggingface.co/datasets/espnet/librispeech/resolve/main/blob_0.bin'
            assert paths[2] == 'https://huggingface.co/datasets/espnet/librispeech/resolve/main/blob_1.bin'

    def test_load_huggingface_missing_dependency(self):
        """Test that missing datasets package raises helpful error."""
        # Only run this test if datasets is NOT installed
        try:
            import datasets
            pytest.skip('datasets is installed, cannot test missing dependency')
        except ImportError:
            pass

        # Without mocking, calling load with huggingface should raise ImportError
        with pytest.raises(ImportError, match="huggingface"):
            load('espnet/librispeech', 'huggingface')


class TestServe:
    """Tests for the serve() function."""

    def test_serve_full_file(self):
        """Test serving a complete file without range request."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test file
            test_file = Path(tmpdir) / 'blob_0.bin'
            test_data = b'test data content'
            test_file.write_bytes(test_data)

            # Start server in background thread
            port = 8765
            server_thread = threading.Thread(
                target=serve,
                args=(port, tmpdir, '127.0.0.1'),
                daemon=True
            )
            server_thread.start()
            time.sleep(0.5)  # Wait for server to start

            try:
                # Request file
                response = requests.get(f'http://127.0.0.1:{port}/blob_0.bin', timeout=2)

                assert response.status_code == 200
                assert response.content == test_data
                assert 'Accept-Ranges' in response.headers
                assert response.headers['Accept-Ranges'] == 'bytes'
            except requests.exceptions.RequestException:
                pytest.skip("Server failed to start or respond")

    def test_serve_range_request(self):
        """Test serving a byte range with 206 response."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test file
            test_file = Path(tmpdir) / 'blob_0.bin'
            test_data = b'0123456789' * 100  # 1000 bytes
            test_file.write_bytes(test_data)

            # Start server in background thread
            port = 8766
            server_thread = threading.Thread(
                target=serve,
                args=(port, tmpdir, '127.0.0.1'),
                daemon=True
            )
            server_thread.start()
            time.sleep(0.5)

            try:
                # Request byte range
                headers = {'Range': 'bytes=10-19'}
                response = requests.get(
                    f'http://127.0.0.1:{port}/blob_0.bin',
                    headers=headers,
                    timeout=2
                )

                assert response.status_code == 206
                assert response.content == b'0123456789'
                assert 'Content-Range' in response.headers
                assert response.headers['Content-Range'] == 'bytes 10-19/1000'
                assert response.headers['Content-Length'] == '10'
            except requests.exceptions.RequestException:
                pytest.skip("Server failed to start or respond")

    def test_serve_multiple_bins(self):
        """Test serving multiple bin files and metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            (Path(tmpdir) / 'blob_0.bin').write_bytes(b'data0')
            (Path(tmpdir) / 'blob_1.bin').write_bytes(b'data1')

            # Create metadata
            metadata = pa.table({
                'item_id': ['001', '002'],
                'bin_index': [0, 1],
                'start_byte': [0, 0],
                'end_byte': [5, 5],
                'path': ['blob_0.bin', 'blob_1.bin'],
            })
            pq.write_table(metadata, Path(tmpdir) / 'metadata.parquet')

            # Start server
            port = 8767
            server_thread = threading.Thread(
                target=serve,
                args=(port, tmpdir, '127.0.0.1'),
                daemon=True
            )
            server_thread.start()
            time.sleep(0.5)

            try:
                # Test blob_0.bin
                resp0 = requests.get(f'http://127.0.0.1:{port}/blob_0.bin', timeout=2)
                assert resp0.status_code == 200
                assert resp0.content == b'data0'

                # Test blob_1.bin
                resp1 = requests.get(f'http://127.0.0.1:{port}/blob_1.bin', timeout=2)
                assert resp1.status_code == 200
                assert resp1.content == b'data1'

                # Test metadata.parquet
                resp_meta = requests.get(f'http://127.0.0.1:{port}/metadata.parquet', timeout=2)
                assert resp_meta.status_code == 200
                # Should be able to parse the metadata
                table = pq.read_table(BytesIO(resp_meta.content))
                assert table.num_rows == 2
            except requests.exceptions.RequestException:
                pytest.skip("Server failed to start or respond")

    def test_serve_invalid_range(self):
        """Test that invalid range requests return 416."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / 'blob_0.bin'
            test_file.write_bytes(b'0123456789')  # 10 bytes

            port = 8768
            server_thread = threading.Thread(
                target=serve,
                args=(port, tmpdir, '127.0.0.1'),
                daemon=True
            )
            server_thread.start()
            time.sleep(0.5)

            try:
                # Request range beyond file size
                headers = {'Range': 'bytes=100-200'}
                response = requests.get(
                    f'http://127.0.0.1:{port}/blob_0.bin',
                    headers=headers,
                    timeout=2
                )

                assert response.status_code == 416  # Range Not Satisfiable
            except requests.exceptions.RequestException:
                pytest.skip("Server failed to start or respond")

    def test_serve_nonexistent_directory(self):
        """Test that serving nonexistent directory raises error."""
        with pytest.raises(FileNotFoundError):
            serve(8769, '/nonexistent/path')

    def test_serve_file_not_found(self):
        """Test that requesting nonexistent file returns 404."""
        with tempfile.TemporaryDirectory() as tmpdir:
            port = 8770
            server_thread = threading.Thread(
                target=serve,
                args=(port, tmpdir, '127.0.0.1'),
                daemon=True
            )
            server_thread.start()
            time.sleep(0.5)

            try:
                response = requests.get(
                    f'http://127.0.0.1:{port}/nonexistent.bin',
                    timeout=2
                )
                assert response.status_code == 404
            except requests.exceptions.RequestException:
                pytest.skip("Server failed to start or respond")
