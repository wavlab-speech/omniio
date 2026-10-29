#!/usr/bin/env python
"""
Example demonstrating remote archive access with omniio.

This example shows:
1. Creating a local audio archive
2. Serving it over HTTP
3. Loading remote metadata with updated URLs
4. Reading audio from the remote archive
"""

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from omniio.blob.blob import Blob
from omniio.interface import audio_read
from omniio.remote import load, serve


def create_sample_archive(archive_dir: str):
    """Create a sample audio archive."""
    print(f"Creating sample archive in {archive_dir}...")

    # Generate sample audio files
    sample_rate = 16000
    duration = 0.5  # seconds

    audio_files = []
    for i, freq in enumerate([440, 523, 659]):  # A4, C5, E5
        # Generate sine wave
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = (np.sin(2 * np.pi * freq * t) * 0.5).astype(np.float32)

        # Save as WAV
        audio_file = Path(archive_dir) / f'audio_{i}.wav'
        sf.write(str(audio_file), audio_data, sample_rate)
        audio_files.append(str(audio_file))

    # Create blob archive
    blob = Blob(
        archive_dir=archive_dir,
        modality='audio',
        max_bin_size=10 * 1024 * 1024  # 10MB
    )

    blob.append(
        items=audio_files,
        ids=[f'sample_{i:03d}' for i in range(len(audio_files))],
        num_workers=0,
        target_format='flac',
        target_bit_depth=16
    )

    # Clean up temp files
    for audio_file in audio_files:
        Path(audio_file).unlink()

    print("Archive created successfully!")
    blob.summary()


def demonstrate_local_access(archive_dir: str):
    """Demonstrate reading from local archive."""
    print("\n=== Local Access ===")

    import pyarrow.parquet as pq

    # Load metadata
    metadata_path = Path(archive_dir) / 'metadata.parquet'
    table = pq.read_table(str(metadata_path))

    # Read first audio entry
    row = table.slice(0, 1).to_pylist()[0]
    result = audio_read(
        row['path'],
        row['start_byte'],
        row['end_byte'] - row['start_byte']
    )

    print(f"Read audio locally: {result.array.shape} @ {result.sample_rate}Hz")


def demonstrate_remote_access(archive_dir: str):
    """Demonstrate serving and remote access."""
    print("\n=== Remote Access ===")
    print("In a production scenario, you would:")
    print("1. On server: python -c \"from omniio.remote import serve; serve(8000, './archive')\"")
    print("2. On client: Load metadata with remote URLs")
    print()

    import pyarrow.parquet as pq

    # Simulate loading metadata with remote URLs
    metadata_path = Path(archive_dir) / 'metadata.parquet'
    remote_table = load(str(metadata_path), 'http://example.com:8000')

    # Show updated paths
    print("Updated paths in metadata:")
    paths = remote_table.column('path').to_pylist()
    for i, path in enumerate(paths):
        print(f"  Entry {i}: {path}")

    # Save remote metadata
    remote_metadata_path = Path(archive_dir) / 'remote_metadata.parquet'
    pq.write_table(remote_table, str(remote_metadata_path))
    print(f"\nRemote metadata saved to: {remote_metadata_path}")
    print("This metadata can be used with the existing Dataset classes,")
    print("and omniio will automatically use HTTP range requests.")


def main():
    """Run the example."""
    print("=== omniio Remote Access Example ===\n")

    # Create temporary archive
    with tempfile.TemporaryDirectory() as tmpdir:
        archive_dir = Path(tmpdir) / 'audio_archive'
        archive_dir.mkdir()

        # Create sample archive
        create_sample_archive(str(archive_dir))

        # Demonstrate local access
        demonstrate_local_access(str(archive_dir))

        # Demonstrate remote access concept
        demonstrate_remote_access(str(archive_dir))

    print("\n=== Example complete! ===")
    print("\nTo serve an archive:")
    print("  python -c \"from omniio.remote import serve; serve(8000, './my_archive')\"")
    print("\nTo load remote metadata:")
    print("  from omniio.remote import load")
    print("  table = load('http://server:8000/metadata.parquet', 'http://server:8000')")


if __name__ == '__main__':
    main()
