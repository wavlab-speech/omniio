#!/usr/bin/env python3
"""
Generic binary blob
"""

import collections
import os
import shutil
import threading
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union

from tqdm import tqdm

import pyarrow as pa
import pyarrow.parquet as pq

from omniio.blob.write import modality_writer


def _worker_process(
    worker_id: int,
    items_with_ids: List[Tuple[Any, str]],
    modality: str,
    shard_dir: str,
    existing_ids: Set[str],
    allow_duplicate_ids: bool,
    modality_kwargs: dict,
    max_bin_size: int,
    progress_file: Optional[str] = None,
) -> Tuple[int, List[Tuple[str, str, int]]]:
    """
    Worker function that runs in a separate process.
    Writes items to one or more shard bin files in shard_dir, splitting into a
    new file when max_bin_size is exceeded.  Files are named
    shard_<worker_id>_<bin_id>.bin / .parquet.

    Worker 0 optionally writes its per-item count to progress_file so the
    main process can approximate overall progress.

    Returns:
        (worker_id, [(bin_path, meta_path, n_rows), ...])
    """
    import warnings

    write_fn = modality_writer[modality]

    def _make_paths(bid: int) -> Tuple[str, str]:
        bp = os.path.join(shard_dir, f"shard_{worker_id}_{bid}.bin")
        mp = os.path.join(shard_dir, f"shard_{worker_id}_{bid}.parquet")
        return bp, mp

    sub_results: List[Tuple[str, str, int]] = []
    bin_id = 0
    bin_path, meta_path = _make_paths(bin_id)

    metadata_rows: list = []
    offset = 0
    cur_bin_size = 0
    n_written = 0
    _last_progress_write = 0.0

    f = open(bin_path, "wb")
    try:
        for item, item_id in items_with_ids:
            if item_id in existing_ids and not allow_duplicate_ids:
                warnings.warn(
                    f"Skipping duplicate id already in archive: {item_id}. "
                    "Pass allow_duplicate_ids=True to write it anyway."
                )
                continue

            raw_bytes, meta_dict = write_fn(item, item_id, **modality_kwargs)
            n_bytes = len(raw_bytes)

            # Roll over to a new bin when the current one is non-empty and full
            if cur_bin_size > 0 and cur_bin_size + n_bytes > max_bin_size:
                f.close()
                if metadata_rows:
                    table = pa.Table.from_pylist(metadata_rows)
                    pq.write_table(table, meta_path)
                    sub_results.append((bin_path, meta_path, len(metadata_rows)))
                else:
                    os.remove(bin_path)
                bin_id += 1
                bin_path, meta_path = _make_paths(bin_id)
                f = open(bin_path, "wb")
                metadata_rows = []
                offset = 0
                cur_bin_size = 0

            f.write(raw_bytes)
            meta_dict["id"] = item_id
            meta_dict["start_byte"] = offset
            meta_dict["end_byte"] = offset + n_bytes
            meta_dict["bin_index"] = -1  # placeholder, resolved during concat
            metadata_rows.append(meta_dict)
            offset += n_bytes
            cur_bin_size += n_bytes
            n_written += 1

            if progress_file is not None:
                now = time.time()
                if now - _last_progress_write >= 0.5:
                    with open(progress_file, "w") as pf:
                        pf.write(str(n_written))
                    _last_progress_write = now
    finally:
        f.close()

    # Flush final bin
    if metadata_rows:
        table = pa.Table.from_pylist(metadata_rows)
        pq.write_table(table, meta_path)
        sub_results.append((bin_path, meta_path, len(metadata_rows)))
    elif os.path.exists(bin_path):
        os.remove(bin_path)

    return worker_id, sub_results


class Blob:
    """Creates and manages binary blobs"""

    def __init__(
        self,
        archive_dir: str,
        modality: str,
        name: Optional[str] = None,
        max_bin_size: int = 32 * 1024 * 1024 * 10,
    ):
        """
        Initialize archive.

        Args:
            archive_dir: Path to the archive directory
                archive_dir/
                    blob_0.bin
                    blob_1.bin (if over max_bin_size)
                    ...
                    metadata.parquet
        """

        supported_modalities = modality_writer.keys()
        assert modality in supported_modalities, f"{modality} not in {supported_modalities}"

        self.modality = modality
        self.max_bin_size = max_bin_size

        base_path = Path(archive_dir)
        base_path.mkdir(parents=True, exist_ok=True)
        self.archive_path = base_path / "blob"
        self.metadata_file = base_path / "metadata.parquet"

        self.data: Optional[pa.Table] = None
        if self.metadata_file.exists():
            self.data = self._read_metadata()

        if name is not None:
            self.name = name
        else:
            archive_dir = str(archive_dir)
            self.name = archive_dir.rstrip("/").split("/")[-1]

    # ------------------------------------------------------------------ #
    # Metadata helpers (pyarrow)
    # ------------------------------------------------------------------ #

    def _read_metadata(self) -> pa.Table:
        if not self.metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_file}")
        return pq.read_table(self.metadata_file)

    def get_metadata(self) -> pa.Table:
        """Return the metadata as a PyArrow Table."""
        if self.data is None:
            self.data = self._read_metadata()
        return self.data

    def _existing_ids(self) -> Set[str]:
        """Return set of ids already in the archive."""
        if self.data is None:
            return set()
        return set(self.data.column("id").to_pylist())

    # ------------------------------------------------------------------ #
    # Bin file helpers
    # ------------------------------------------------------------------ #

    def __len__(self):
        if self.data is None:
            return 0
        return self.data.num_rows

    def _get_bin_file_path(self, bin_index: int) -> Path:
        return self.archive_path.with_name(f"{self.archive_path.stem}_{bin_index}.bin")

    def _get_current_bin_info(self) -> Tuple[int, int]:
        bin_index = 0
        while True:
            bin_path = self._get_bin_file_path(bin_index)
            if not bin_path.exists():
                if bin_index == 0:
                    return 0, 0
                else:
                    prev = self._get_bin_file_path(bin_index - 1)
                    return bin_index - 1, prev.stat().st_size
            bin_index += 1

    # ------------------------------------------------------------------ #
    # Summary / clear
    # ------------------------------------------------------------------ #

    def summary(self):
        if self.data is None:
            print(f"\nArchive: {self.archive_path}")
            print("Archive is empty (no metadata file found).")
            return

        table = self.data
        num = table.num_rows
        print(f"\nArchive: {self.archive_path}")
        print(f"Total files: {num}")

        if num == 0:
            return

        if "data_type" in table.column_names:
            counts = collections.Counter(table.column("data_type").to_pylist())
            for dtype, count in counts.items():
                print(f"  {dtype}: {count} files")

        bin_indices = set(table.column("bin_index").to_pylist())
        total_size = 0
        for idx in sorted(bin_indices):
            p = self._get_bin_file_path(idx)
            if p.exists():
                total_size += p.stat().st_size
        print(f"Total size: {total_size / (1024**3):.2f} GB")
        print(f"Number of bin files: {len(bin_indices)}")

    def clear(self, confirm: bool = False):
        if not confirm:
            raise ValueError(
                "Archive clearing requires confirmation. "
                "Call clear(confirm=True) to proceed. "
                "This will delete all bin files and metadata!"
            )

        files_deleted = 0
        if self.metadata_file.exists():
            self.metadata_file.unlink()
            files_deleted += 1

        bin_index = 0
        while True:
            bin_path = self._get_bin_file_path(bin_index)
            if bin_path.exists():
                bin_path.unlink()
                files_deleted += 1
                bin_index += 1
            else:
                break

        self.data = None
        print(f"Archive '{self.archive_path.stem}' cleared successfully!")
        print(f"Deleted {files_deleted} file(s) ({bin_index} bin file(s) + metadata).")

    # ------------------------------------------------------------------ #
    # Append (parallelized)
    # ------------------------------------------------------------------ #

    def append(
        self,
        items: List[Any],
        ids: List[str] = None,
        num_workers: int = 0,
        allow_duplicate_ids: bool = False,
        progress: bool = True,
        reshard: bool = False,
        **modality_kwargs,
    ):
        """
        Append items to the archive in parallel.

        Each worker writes to its own temp bin + metadata shard.  After all
        workers finish (or on error, after all workers finish their current
        item), the shards are merged into the main archive.

        Duplicate-id checking against the existing archive happens inside
        each worker process, so the full id set is serialized once to each
        worker rather than requiring a synchronization step.

        Args:
            items:       List of raw items matching the modality.
            ids:         Unique string id per item.  Must be same length as items.
            num_workers: 0 means single-process (no pool). >0 spawns a pool.
            allow_duplicate_ids: If False (default), items whose id already
                         exists in the archive are skipped with a warning.
                         If True, the item is written regardless.
            reshard:     If True, shard bins are streamed and concatenated into
                         the existing blob files (respecting max_bin_size).
                         If False (default), each shard bin is simply moved into
                         the archive as its own blob file — much faster, at the
                         cost of producing more bin files.
            **modality_kwargs: Forwarded to the modality write_fn.
        """
        if ids is None:
            ids = [f"{self.name}_{i}" for i in range(len(self), len(self) + len(items))]
        else:
            assert len(items) == len(ids), "items and ids must be the same length"

        # --- Check uniqueness of incoming ids -------------------------
        seen = set(ids)
        if len(seen) != len(ids):
            raise ValueError(f"Duplicate id in input")

        # --- Snapshot existing ids for workers ------------------------
        existing_ids = self._existing_ids()

        # --- Partition work -------------------------------------------
        effective_workers = max(num_workers, 1)
        chunks: List[List[Tuple[Any, str]]] = [[] for _ in range(effective_workers)]
        for i, (item, item_id) in enumerate(zip(items, ids)):
            chunks[i % effective_workers].append((item, item_id))
        chunks = [c for c in chunks if c]

        log_dir = self.archive_path.parent / "logs"
        log_dir.mkdir(exist_ok=True)

        try:
            # Each entry: (worker_id, [(bin_path, meta_path, n_rows), ...])
            completed_results: List[Tuple[int, List[Tuple[str, str, int]]]] = []
            first_error: Optional[Exception] = None

            pbar = tqdm(total=len(items), desc="appending", disable=not progress)

            if num_workers <= 0:
                # ---- single-process path (per-item progress) ----
                import warnings as _warnings

                write_fn = modality_writer[self.modality]

                def _sp_paths(bid: int) -> Tuple[str, str]:
                    return (
                        os.path.join(str(log_dir), f"shard_0_{bid}.bin"),
                        os.path.join(str(log_dir), f"shard_0_{bid}.parquet"),
                    )

                sub_results: List[Tuple[str, str, int]] = []
                sp_bin_id = 0
                bin_path, meta_path = _sp_paths(0)
                metadata_rows: list = []
                offset = 0
                cur_bin_size = 0

                try:
                    f = open(bin_path, "wb")
                    try:
                        for item, item_id in zip(items, ids):
                            if item_id in existing_ids and not allow_duplicate_ids:
                                _warnings.warn(
                                    f"Skipping duplicate id already in archive: {item_id}. "
                                    "Pass allow_duplicate_ids=True to write it anyway."
                                )
                                pbar.update(1)
                                continue
                            raw_bytes, meta_dict = write_fn(item, item_id, **modality_kwargs)
                            n_bytes = len(raw_bytes)

                            if cur_bin_size > 0 and cur_bin_size + n_bytes > self.max_bin_size:
                                f.close()
                                if metadata_rows:
                                    table = pa.Table.from_pylist(metadata_rows)
                                    pq.write_table(table, meta_path)
                                    sub_results.append((bin_path, meta_path, len(metadata_rows)))
                                else:
                                    os.remove(bin_path)
                                sp_bin_id += 1
                                bin_path, meta_path = _sp_paths(sp_bin_id)
                                f = open(bin_path, "wb")
                                metadata_rows = []
                                offset = 0
                                cur_bin_size = 0

                            f.write(raw_bytes)
                            meta_dict["id"] = item_id
                            meta_dict["start_byte"] = offset
                            meta_dict["end_byte"] = offset + n_bytes
                            meta_dict["bin_index"] = -1
                            metadata_rows.append(meta_dict)
                            offset += n_bytes
                            cur_bin_size += n_bytes
                            pbar.update(1)
                    finally:
                        f.close()

                    if metadata_rows:
                        table = pa.Table.from_pylist(metadata_rows)
                        pq.write_table(table, meta_path)
                        sub_results.append((bin_path, meta_path, len(metadata_rows)))
                    elif os.path.exists(bin_path):
                        os.remove(bin_path)

                    completed_results.append((0, sub_results))
                except Exception as exc:
                    first_error = exc
            else:
                # ---- multi-process path (worker-0 progress approximation) ----
                progress_file = str(log_dir / "worker0_progress.txt")
                num_chunks = len(chunks)

                # Background thread polls worker 0's progress file and scales
                # by num_chunks to approximate total progress.
                stop_poll = threading.Event()
                last_reported = [0]

                def _poll_progress():
                    while not stop_poll.is_set():
                        try:
                            with open(progress_file) as _pf:
                                w0_count = int(_pf.read().strip())
                            estimated = min(w0_count * num_chunks, len(items))
                            delta = estimated - last_reported[0]
                            if delta > 0:
                                pbar.update(delta)
                                last_reported[0] = estimated
                        except Exception:
                            pass
                        time.sleep(0.5)

                poll_thread = threading.Thread(target=_poll_progress, daemon=True)
                poll_thread.start()

                futures = {}
                with ProcessPoolExecutor(max_workers=num_workers) as pool:
                    for wid, chunk in enumerate(chunks):
                        fut = pool.submit(
                            _worker_process,
                            wid, chunk, self.modality, str(log_dir),
                            existing_ids, allow_duplicate_ids, modality_kwargs,
                            self.max_bin_size,
                            progress_file if wid == 0 else None,
                        )
                        futures[fut] = wid

                    for fut in as_completed(futures):
                        try:
                            result = fut.result()
                            completed_results.append(result)
                        except Exception as exc:
                            if first_error is None:
                                first_error = exc

                stop_poll.set()
                poll_thread.join()

                # Bring pbar to 100% (approximation may not reach exactly total)
                remaining = len(items) - last_reported[0]
                if remaining > 0:
                    pbar.update(remaining)

            pbar.close()

            # --- Concatenate shards onto the archive ------------------
            if reshard:
                self._concat_shards(completed_results)
            else:
                self._concat_shards_fast(completed_results)

            if first_error is not None:
                raise RuntimeError(
                    "One or more workers failed. Successfully processed shards "
                    "have been concatenated into the archive. Original error:\n"
                    f"{first_error}"
                ) from first_error

        finally:
            shutil.rmtree(log_dir, ignore_errors=True)

    # ------------------------------------------------------------------ #
    # Shard concatenation
    # ------------------------------------------------------------------ #

    def _concat_shards(
        self,
        results: List[Tuple[int, List[Tuple[str, str, int]]]],
    ):
        """
        Merge worker shard files into the main archive.

        Bin data is concatenated via streaming copy (never fully in memory).
        Metadata byte offsets are shifted by reading file sizes only.
        """
        if not results:
            return

        results.sort(key=lambda r: r[0])
        # Flatten sub-bins in worker order
        flat: List[Tuple[str, str, int]] = []
        for _wid, sub_bins in results:
            flat.extend(sub_bins)

        cur_bin_index, cur_bin_size = self._get_current_bin_info()

        meta_tables: List[pa.Table] = []

        if self.data is not None and self.data.num_rows > 0:
            meta_tables.append(self.data)

        for shard_bin_path, shard_meta_path, n_rows in flat:
            if n_rows == 0:
                continue

            shard_bin_size = os.path.getsize(shard_bin_path)

            if cur_bin_size > 0 and (cur_bin_size + shard_bin_size) > self.max_bin_size:
                cur_bin_index += 1
                cur_bin_size = 0

            byte_offset = cur_bin_size

            dest_bin = self._get_bin_file_path(cur_bin_index)
            with open(dest_bin, "ab") as dst, open(shard_bin_path, "rb") as src:
                shutil.copyfileobj(src, dst)

            shard_meta = pq.read_table(shard_meta_path)

            start_col = shard_meta.column("start_byte")
            end_col = shard_meta.column("end_byte")

            shifted_start = pa.compute.add(start_col, byte_offset)
            shifted_end = pa.compute.add(end_col, byte_offset)
            bin_index_col = pa.array(
                [cur_bin_index] * shard_meta.num_rows, type=pa.int64()
            )

            # Add path column with absolute path to bin file
            bin_path = str(dest_bin.absolute())
            path_col = pa.array(
                [bin_path] * shard_meta.num_rows, type=pa.string()
            )

            shard_meta = shard_meta.set_column(
                shard_meta.schema.get_field_index("start_byte"), "start_byte", shifted_start
            )
            shard_meta = shard_meta.set_column(
                shard_meta.schema.get_field_index("end_byte"), "end_byte", shifted_end
            )
            shard_meta = shard_meta.set_column(
                shard_meta.schema.get_field_index("bin_index"), "bin_index", bin_index_col
            )

            # Add or update path column
            if "path" in shard_meta.column_names:
                shard_meta = shard_meta.set_column(
                    shard_meta.schema.get_field_index("path"), "path", path_col
                )
            else:
                shard_meta = shard_meta.append_column("path", path_col)

            meta_tables.append(shard_meta)
            cur_bin_size += shard_bin_size

        if meta_tables:
            combined = pa.concat_tables(meta_tables, promote_options="default")
            pq.write_table(combined, self.metadata_file)
            self.data = combined

    def _concat_shards_fast(
        self,
        results: List[Tuple[int, List[Tuple[str, str, int]]]],
    ):
        """
        Merge worker shard files into the archive without resharding.

        Each shard bin is moved (renamed) into the archive directory as its
        own blob file.  Byte offsets within each shard are already correct
        (0-based from the start of that shard), so no shifting is needed.
        Only the parquet metadata tables are concatenated.
        """
        if not results:
            return

        results.sort(key=lambda r: r[0])
        # Flatten sub-bins in worker order
        flat: List[Tuple[str, str, int]] = []
        for _wid, sub_bins in results:
            flat.extend(sub_bins)

        # Find the next available bin index
        cur_bin_index, _ = self._get_current_bin_info()
        # If a bin file already exists at cur_bin_index, next shard goes after it
        if self._get_bin_file_path(cur_bin_index).exists():
            cur_bin_index += 1

        meta_tables: List[pa.Table] = []

        if self.data is not None and self.data.num_rows > 0:
            meta_tables.append(self.data)

        for shard_bin_path, shard_meta_path, n_rows in flat:
            if n_rows == 0:
                continue

            dest_bin = self._get_bin_file_path(cur_bin_index)
            shutil.move(shard_bin_path, dest_bin)

            shard_meta = pq.read_table(shard_meta_path)

            bin_index_col = pa.array(
                [cur_bin_index] * shard_meta.num_rows, type=pa.int64()
            )
            bin_path = str(dest_bin.absolute())
            path_col = pa.array(
                [bin_path] * shard_meta.num_rows, type=pa.string()
            )

            shard_meta = shard_meta.set_column(
                shard_meta.schema.get_field_index("bin_index"), "bin_index", bin_index_col
            )
            if "path" in shard_meta.column_names:
                shard_meta = shard_meta.set_column(
                    shard_meta.schema.get_field_index("path"), "path", path_col
                )
            else:
                shard_meta = shard_meta.append_column("path", path_col)

            meta_tables.append(shard_meta)
            cur_bin_index += 1

        if meta_tables:
            combined = pa.concat_tables(meta_tables, promote_options="default")
            pq.write_table(combined, self.metadata_file)
            self.data = combined