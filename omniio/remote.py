"""Remote access utilities for omniio archives."""

import http.server
import os
import socketserver
from io import BytesIO
from pathlib import Path
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq
import requests


def load(path: str, url: str) -> pa.Table:
    """
    Load parquet metadata and update path column for remote access.

    Args:
        path: Path to metadata.parquet (local/URL/directory) or HF repo name
        url: Base URL for HTTP server, or "huggingface" for HF datasets

    Returns:
        PyArrow Table with updated path column pointing to remote URLs

    Examples:
        # Load from local metadata with remote URL
        >>> table = load('./archive/metadata.parquet', 'http://server:8000')

        # Load from remote metadata
        >>> table = load('http://server:8000/metadata.parquet', 'http://server:8000')

        # Load from HuggingFace dataset
        >>> table = load('espnet/librispeech', 'huggingface')
    """
    # Handle directory input
    if path.endswith('/') or (not path.startswith('http') and os.path.isdir(path)):
        path = os.path.join(path, 'metadata.parquet')

    # Load metadata based on source type
    if url.lower() == "huggingface":
        table = _load_huggingface(path)
        # Construct HuggingFace URLs
        base_url = f"https://huggingface.co/datasets/{path}/resolve/main"
    else:
        # Load from local or remote
        if path.startswith('http://') or path.startswith('https://'):
            # Remote metadata
            resp = requests.get(path)
            resp.raise_for_status()
            table = pq.read_table(BytesIO(resp.content))
        else:
            # Local metadata
            table = pq.read_table(path)

        # Use provided URL as base
        base_url = url.rstrip('/')

    # Update path column with remote URLs
    bin_indices = table.column('bin_index').to_pylist()
    new_paths = [f"{base_url}/blob_{idx}.bin" for idx in bin_indices]
    new_path_col = pa.array(new_paths, type=pa.string())

    # Replace path column
    path_idx = table.schema.get_field_index('path')
    table = table.set_column(path_idx, 'path', new_path_col)

    return table


def _load_huggingface(repo_name: str) -> pa.Table:
    """
    Load metadata from HuggingFace dataset.

    Args:
        repo_name: HuggingFace dataset repository name (e.g., 'espnet/librispeech')

    Returns:
        PyArrow Table from the dataset
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "HuggingFace datasets support requires the 'huggingface' extra. "
            "Install with: pip install omniio[huggingface]"
        )

    dataset = load_dataset(repo_name)
    # Access the underlying PyArrow table
    return dataset.data.table


def serve(port: int, archive_dir: str = "./", host: str = "0.0.0.0"):
    """
    Start HTTP server to serve binary blobs and metadata with range request support.

    Args:
        port: Port number to listen on
        archive_dir: Directory containing blob_*.bin and metadata.parquet
        host: Host address to bind to (default: all interfaces)

    Examples:
        # Serve archive on port 8000
        >>> serve(8000, './my_archive')

        # Serve on localhost only
        >>> serve(8000, './my_archive', host='127.0.0.1')
    """
    archive_path = Path(archive_dir).resolve()

    if not archive_path.exists():
        raise FileNotFoundError(f"Archive directory not found: {archive_dir}")

    class RangeRequestHandler(http.server.SimpleHTTPRequestHandler):
        """HTTP handler with range request support for omniio archives."""

        def __init__(self, *args, **kwargs):
            # Set directory to serve from
            super().__init__(*args, directory=str(archive_path), **kwargs)

        def do_GET(self):
            """Handle GET request with optional Range header."""
            # Validate path stays within archive_dir
            requested_path = self.translate_path(self.path)
            try:
                requested_path = Path(requested_path).resolve()
                if not str(requested_path).startswith(str(archive_path)):
                    self.send_error(403, "Forbidden")
                    return
            except Exception:
                self.send_error(400, "Bad Request")
                return

            # Check if file exists
            if not requested_path.is_file():
                self.send_error(404, "File not found")
                return

            # Get file size
            file_size = requested_path.stat().st_size

            # Check for Range header
            range_header = self.headers.get('Range')

            if range_header:
                # Parse range header (format: "bytes=start-end")
                try:
                    range_spec = range_header.replace('bytes=', '')
                    start, end = range_spec.split('-')
                    start = int(start) if start else 0
                    end = int(end) if end else file_size - 1

                    # Validate range
                    if start >= file_size or end >= file_size or start > end:
                        self.send_error(416, "Range Not Satisfiable")
                        return

                    # Send 206 Partial Content
                    content_length = end - start + 1
                    self.send_response(206)
                    self.send_header('Content-Type', 'application/octet-stream')
                    self.send_header('Content-Length', str(content_length))
                    self.send_header('Content-Range', f'bytes {start}-{end}/{file_size}')
                    self.send_header('Accept-Ranges', 'bytes')
                    self.end_headers()

                    # Send requested byte range
                    with open(requested_path, 'rb') as f:
                        f.seek(start)
                        chunk = f.read(content_length)
                        self.wfile.write(chunk)

                except (ValueError, IndexError):
                    self.send_error(400, "Invalid Range header")
                    return
            else:
                # No range request - serve full file
                self.send_response(200)
                self.send_header('Content-Type', 'application/octet-stream')
                self.send_header('Content-Length', str(file_size))
                self.send_header('Accept-Ranges', 'bytes')
                self.end_headers()

                with open(requested_path, 'rb') as f:
                    self.copyfile(f, self.wfile)

        def log_message(self, format, *args):
            """Override to customize logging."""
            # Format: "GET /blob_0.bin bytes=0-1023" or "GET /metadata.parquet"
            print(f"[omniio] {self.address_string()} - {format % args}")

    # Start server
    try:
        with socketserver.TCPServer((host, port), RangeRequestHandler) as httpd:
            print(f"Serving omniio archive at http://{host}:{port}")
            print(f"Archive directory: {archive_path}")
            print("Press Ctrl+C to stop the server")
            httpd.serve_forever()
    except OSError as e:
        if e.errno == 98:  # Address already in use
            raise OSError(f"Port {port} is already in use. Choose a different port.") from e
        raise
