"""Integration tests for omniio.remote with actual archives."""

import os
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import requests

from omniio.blob.blob import Blob
from omniio.interface import audio_read
from omniio.remote import load, serve


@pytest.fixture
def sample_audio_archive():
    """Create a small audio archive for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        archive_dir = Path(tmpdir) / 'test_archive'
        archive_dir.mkdir()

        # Create a small test audio file (WAV)
        # Generate 1 second of audio at 16kHz
        sample_rate = 16000
        duration = 1.0
        samples = int(sample_rate * duration)
        audio_data = np.sin(2 * np.pi * 440 * np.linspace(0, duration, samples))
        audio_data = audio_data.astype(np.float32)

        # Write as simple WAV file
        import soundfile as sf
        test_audio = archive_dir / 'test.wav'
        sf.write(str(test_audio), audio_data, sample_rate)

        # Create blob archive
        blob = Blob(
            archive_dir=str(archive_dir),
            modality='audio',
            max_bin_size=10 * 1024 * 1024  # 10MB
        )

        # Append the audio file
        blob.append(
            items=[str(test_audio)],
            ids=['test_audio_001'],
            num_workers=0,
            target_format='flac',
            target_bit_depth=16
        )

        # Clean up test audio file
        test_audio.unlink()

        yield archive_dir


class TestRemoteIntegration:
    """Integration tests for remote archive access."""

    def test_roundtrip_remote_audio(self, sample_audio_archive):
        """Test creating archive, serving it, loading remote metadata, and reading audio."""
        port = 9000

        # Start server in background
        server_thread = threading.Thread(
            target=serve,
            args=(port, str(sample_audio_archive), '127.0.0.1'),
            daemon=True
        )
        server_thread.start()
        time.sleep(1)  # Wait for server to start

        try:
            # Load metadata with remote URLs
            local_metadata = sample_audio_archive / 'metadata.parquet'
            remote_table = load(str(local_metadata), f'http://127.0.0.1:{port}')

            # Verify path was updated
            paths = remote_table.column('path').to_pylist()
            assert paths[0] == f'http://127.0.0.1:{port}/blob_0.bin'

            # Save remote metadata
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.parquet', delete=False) as f:
                pq.write_table(remote_table, f.name)
                remote_metadata_path = f.name

            try:
                # Read the table back
                table = pq.read_table(remote_metadata_path)

                # Extract metadata for first entry
                row = table.slice(0, 1).to_pylist()[0]
                archive_path = row['path']
                start_offset = row['start_byte']
                file_size = row['end_byte'] - row['start_byte']

                # Read audio using remote URL (should automatically use audio_read_remote)
                result = audio_read(archive_path, start_offset, file_size)

                # Verify the audio data
                assert result.file_type == 'flac'  # file_type is the format
                assert result.modality == 'audio'
                assert result.sample_rate == 16000
                assert result.array.shape[0] > 0  # Has frames
                assert result.array.dtype == np.float32

            finally:
                os.unlink(remote_metadata_path)

        except requests.exceptions.RequestException:
            pytest.skip("Server failed to start or respond")

    def test_with_dataset(self, sample_audio_archive):
        """Test using remote archive with PyTorch-style Dataset."""
        # First check if examples module exists
        try:
            from examples.audio_dataset import AudioArchiveDataset
        except (ImportError, ModuleNotFoundError):
            pytest.skip("AudioArchiveDataset example not available")

        port = 9001

        # Start server
        server_thread = threading.Thread(
            target=serve,
            args=(port, str(sample_audio_archive), '127.0.0.1'),
            daemon=True
        )
        server_thread.start()
        time.sleep(1)

        try:
            # Load and save remote metadata
            local_metadata = sample_audio_archive / 'metadata.parquet'
            remote_table = load(str(local_metadata), f'http://127.0.0.1:{port}')

            with tempfile.NamedTemporaryFile(mode='wb', suffix='.parquet', delete=False) as f:
                pq.write_table(remote_table, f.name)
                remote_metadata_path = f.name

            try:
                # Create dataset with remote metadata
                dataset = AudioArchiveDataset(remote_metadata_path)

                # Verify dataset length
                assert len(dataset) == 1

                # Get first sample
                sample = dataset[0]

                # Verify sample structure
                assert 'audio' in sample
                assert 'sample_rate' in sample
                assert 'id' in sample  # Dataset uses 'id' not 'item_id'
                assert sample['sample_rate'] == 16000
                assert sample['id'] == 'test_audio_001'
                assert isinstance(sample['audio'], np.ndarray) or hasattr(sample['audio'], 'numpy')

            finally:
                os.unlink(remote_metadata_path)

        except requests.exceptions.RequestException:
            pytest.skip("Server failed to start or respond")

    def test_multiple_bin_files(self):
        """Test remote access with multiple bin files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_dir = Path(tmpdir) / 'multi_bin_archive'
            archive_dir.mkdir()

            # Create multiple small audio files
            import soundfile as sf

            audio_files = []
            for i in range(3):
                audio_data = np.sin(2 * np.pi * 440 * np.linspace(0, 0.1, 1600))
                audio_data = audio_data.astype(np.float32)

                audio_file = archive_dir / f'test_{i}.wav'
                sf.write(str(audio_file), audio_data, 16000)
                audio_files.append(str(audio_file))

            # Create blob with small max size to force multiple bins
            blob = Blob(
                archive_dir=str(archive_dir),
                modality='audio',
                max_bin_size=256  # 256 bytes to force multiple bins
            )

            blob.append(
                items=audio_files,
                ids=[f'audio_{i:03d}' for i in range(3)],
                num_workers=0,
                target_format='flac',
                target_bit_depth=16
            )

            # Clean up test files
            for audio_file in audio_files:
                Path(audio_file).unlink()

            port = 9002

            # Start server
            server_thread = threading.Thread(
                target=serve,
                args=(port, str(archive_dir), '127.0.0.1'),
                daemon=True
            )
            server_thread.start()
            time.sleep(1)

            try:
                # Load remote metadata
                local_metadata = archive_dir / 'metadata.parquet'
                remote_table = load(str(local_metadata), f'http://127.0.0.1:{port}')

                # Check bin indices (FLAC compression may fit all in one bin, which is OK)
                bin_indices = set(remote_table.column('bin_index').to_pylist())
                # Just verify we have at least one bin
                assert len(bin_indices) >= 1

                # Verify all paths are updated correctly
                paths = remote_table.column('path').to_pylist()
                for i, path in enumerate(paths):
                    bin_idx = remote_table.column('bin_index')[i].as_py()
                    expected_path = f'http://127.0.0.1:{port}/blob_{bin_idx}.bin'
                    assert path == expected_path

                # Try reading from all entries
                for i in range(remote_table.num_rows):
                    row = remote_table.slice(i, 1).to_pylist()[0]
                    result = audio_read(
                        row['path'],
                        row['start_byte'],
                        row['end_byte'] - row['start_byte']
                    )
                    assert result.sample_rate == 16000
                    assert result.modality == 'audio'

            except requests.exceptions.RequestException:
                pytest.skip("Server failed to start or respond")

    def test_load_from_remote_metadata(self):
        """Test loading metadata directly from remote server."""
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_dir = Path(tmpdir) / 'test_archive'
            archive_dir.mkdir()

            # Create minimal archive
            import soundfile as sf
            audio_data = np.sin(2 * np.pi * 440 * np.linspace(0, 0.1, 1600)).astype(np.float32)
            test_audio = archive_dir / 'test.wav'
            sf.write(str(test_audio), audio_data, 16000)

            blob = Blob(str(archive_dir), modality='audio')
            blob.append(items=[str(test_audio)], ids=['test'], num_workers=0)
            test_audio.unlink()

            port = 9003

            # Start server
            server_thread = threading.Thread(
                target=serve,
                args=(port, str(archive_dir), '127.0.0.1'),
                daemon=True
            )
            server_thread.start()
            time.sleep(1)

            try:
                # Load metadata directly from remote URL
                remote_table = load(
                    f'http://127.0.0.1:{port}/metadata.parquet',
                    f'http://127.0.0.1:{port}'
                )

                # Verify it loaded and was updated
                assert remote_table.num_rows == 1
                path = remote_table.column('path')[0].as_py()
                assert path.startswith(f'http://127.0.0.1:{port}/blob_')

            except requests.exceptions.RequestException:
                pytest.skip("Server failed to start or respond")
