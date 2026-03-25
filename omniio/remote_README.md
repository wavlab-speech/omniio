# omniio.remote - Remote Archive Access

The `omniio.remote` module enables distributed access to multimedia archives over HTTP without copying large binary files.

## Features

- **HTTP Range Requests**: Efficient streaming of archive data via byte-range requests
- **Metadata Transformation**: Convert local paths to remote URLs
- **Built-in HTTP Server**: Serve archives with minimal configuration
- **HuggingFace Support**: Load archives from HuggingFace datasets (optional)
- **Zero Code Changes**: Works with existing Dataset classes automatically

## Installation

Basic installation:
```bash
pip install omniio
```

With HuggingFace support:
```bash
pip install omniio[huggingface]
```

## Quick Start

### Server Side

Start an HTTP server to serve your archive:

```python
from omniio.remote import serve

# Serve archive on port 8000
serve(port=8000, archive_dir='./my_audio_archive')
# Output: Serving omniio archive at http://0.0.0.0:8000
```

Or from the command line:
```bash
python -c "from omniio.remote import serve; serve(8000, './my_audio_archive')"
```

### Client Side

Load metadata with remote URLs:

```python
import pyarrow.parquet as pq
from omniio.remote import load

# Load from local metadata file with remote URL base
table = load(
    path='http://server:8000/metadata.parquet',  # or local path
    url='http://server:8000'
)

# Save locally
pq.write_table(table, 'remote_metadata.parquet')

# Use with existing Dataset classes - works automatically!
from examples.audio_dataset import AudioArchiveDataset
dataset = AudioArchiveDataset('remote_metadata.parquet')

# Access works exactly the same - omniio handles HTTP requests transparently
sample = dataset[0]
print(sample['audio'].shape, sample['sample_rate'])
```

## API Reference

### `load(path, url)`

Load parquet metadata and update path column for remote access.

**Parameters:**
- `path` (str): Path to metadata.parquet (local/URL/directory) or HF repo name
- `url` (str): Base URL for HTTP server, or "huggingface" for HF datasets

**Returns:**
- `pa.Table`: PyArrow Table with updated path column pointing to remote URLs

**Examples:**

```python
# From local metadata
table = load('./archive/metadata.parquet', 'http://server:8000')

# From remote metadata
table = load('http://server:8000/metadata.parquet', 'http://server:8000')

# From directory (appends /metadata.parquet)
table = load('./archive/', 'http://server:8000')

# From HuggingFace dataset (requires omniio[huggingface])
table = load('espnet/librispeech', 'huggingface')
```

### `serve(port, archive_dir, host)`

Start HTTP server to serve binary blobs and metadata with range request support.

**Parameters:**
- `port` (int): Port number to listen on
- `archive_dir` (str): Directory containing blob_*.bin and metadata.parquet (default: './')
- `host` (str): Host address to bind to (default: '0.0.0.0' for all interfaces)

**Examples:**

```python
# Serve on all interfaces
serve(8000, './my_archive')

# Serve on localhost only
serve(8000, './my_archive', host='127.0.0.1')

# Serve on specific interface
serve(8000, './my_archive', host='192.168.1.100')
```

**Server Features:**
- Supports HTTP range requests (RFC 7233) for efficient streaming
- Returns 206 Partial Content for range requests
- Returns 200 OK for full file requests
- Validates paths to prevent directory traversal
- Logs all requests with file and byte range information

## How It Works

### Automatic Routing

The existing `omniio.interface` module automatically routes between local and remote:

```python
from omniio.interface import audio_read

# If path exists locally, reads from local file
result = audio_read('/local/blob_0.bin', start_offset, file_size)

# If path doesn't exist locally, uses HTTP range requests
result = audio_read('http://server:8000/blob_0.bin', start_offset, file_size)
```

No code changes needed - just update the metadata paths!

### Metadata Transformation

The `load()` function updates the `path` column in metadata:

**Before:**
```
path: /local/archive/blob_0.bin
bin_index: 0
```

**After:**
```
path: http://server:8000/blob_0.bin
bin_index: 0
```

### HTTP Range Requests

Remote reading uses efficient byte-range requests:

```
GET /blob_0.bin HTTP/1.1
Range: bytes=1000-2999
```

Returns only the requested 2000 bytes, not the entire file.

## Use Cases

### 1. Distributed Training

Train on remote archives without downloading:

```python
# On training machines (clients)
table = load('http://data-server:8000/metadata.parquet', 'http://data-server:8000')
pq.write_table(table, 'remote_metadata.parquet')

dataset = AudioArchiveDataset('remote_metadata.parquet')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in dataloader:
    # Data streamed on-demand from remote server
    train_step(batch)
```

### 2. HuggingFace Datasets

Share archives via HuggingFace:

```python
# Load from HF dataset
table = load('username/my_audio_dataset', 'huggingface')
pq.write_table(table, 'hf_metadata.parquet')

# Use normally
dataset = AudioArchiveDataset('hf_metadata.parquet')
```

### 3. Cloud Storage

Serve from cloud storage URLs:

```python
# Point to S3/GCS URLs
table = load('./local_metadata.parquet', 'https://my-bucket.s3.amazonaws.com/archives')
# Paths become: https://my-bucket.s3.amazonaws.com/archives/blob_0.bin
```

## Performance Considerations

- **Network Bandwidth**: Reading speed limited by network connection
- **Latency**: Each sample requires HTTP request - use larger batch sizes
- **Caching**: Consider implementing client-side caching for frequently accessed data
- **Parallel Workers**: Use multiple DataLoader workers to overlap network I/O

## Security Notes

- The built-in server validates paths to prevent directory traversal
- Only files within `archive_dir` can be accessed
- For production use, consider:
  - Using HTTPS with proper certificates
  - Adding authentication (e.g., nginx with basic auth)
  - Rate limiting to prevent abuse
  - Firewall rules to restrict access

## Examples

See `examples/remote_example.py` for a complete working example.

## Troubleshooting

**Port already in use:**
```python
OSError: Port 8000 is already in use. Choose a different port.
```
Solution: Use a different port number or stop the process using the port.

**Connection errors:**
```python
requests.exceptions.ConnectionError: Failed to establish connection
```
Solution: Ensure server is running and accessible from client machine.

**Missing HuggingFace dependency:**
```python
ImportError: HuggingFace datasets support requires the 'huggingface' extra
```
Solution: Install with `pip install omniio[huggingface]`
