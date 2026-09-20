"""Content-verified acquisition with a persistent response-byte budget."""

from __future__ import annotations

import fcntl
import hashlib
import io
import json
import operator
import os
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_json(value) + b"\n")
    os.replace(temporary, path)


class AcquisitionError(RuntimeError):
    pass


class NoBodyRedirect(HTTPRedirectHandler):
    """urllib normally drains redirect bodies without a size bound."""

    def http_error_302(self, request, response, code, message, headers):
        response.close()
        return super().http_error_302(request, io.BytesIO(), code, message, headers)

    http_error_301 = http_error_302
    http_error_303 = http_error_302
    http_error_307 = http_error_302
    http_error_308 = http_error_302


class BoundedHTTP:
    """Persistent response-body byte budget, including failed/retried requests.

    Reserve before opening each response: crashes retain a conservative charge.
    Redirect/error bodies are never read. Ignore-Range servers are closed before
    reading their body. Cached receipts are content-verified before every reuse.
    One cache lock prevents concurrent runs from spending the same budget.
    """

    def __init__(self, cache: Path, cap: int, opener=None):
        if not isinstance(cap, int) or isinstance(cap, bool) or cap <= 0:
            raise ValueError("network cap must be a positive integer")
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.lock = (self.cache / "acquisition.lock").open("a+b")
        fcntl.flock(self.lock, fcntl.LOCK_EX)
        self.ledger_path = self.cache / "network.json"
        try:
            self.ledger = (
                json.loads(self.ledger_path.read_text())
                if self.ledger_path.exists()
                else {"charged_bytes": 0, "attempts": []}
            )
            self.cap = cap
            self.opener = (
                opener if opener is not None else build_opener(NoBodyRedirect()).open
            )
            if self.ledger["charged_bytes"] > cap:
                raise AcquisitionError(
                    "cache already exceeds requested lifetime network cap"
                )
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if not self.lock.closed:
            fcntl.flock(self.lock, fcntl.LOCK_UN)
            self.lock.close()

    def _cache_paths(self, url: str, start: int | None, limit: int):
        key = sha256(canonical_json({"url": url, "start": start, "limit": limit}))
        return self.cache / (key + ".bin"), self.cache / (key + ".json")

    def is_cached(self, url: str, *, start: int | None, limit: int) -> bool:
        """A lookup hint only; fetch still validates every reused byte."""
        return all(path.exists() for path in self._cache_paths(url, start, limit))

    def fetch(
        self,
        url: str,
        *,
        limit: int,
        start: int | None = None,
        total: int | None = None,
    ) -> tuple[bytes, dict]:
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise ValueError("request limit must be a positive integer")
        if start is not None and (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(total, int)
            or isinstance(total, bool)
            or start < 0
            or start + limit > total
        ):
            raise ValueError("request Range must lie inside its declared file size")
        payload_path, receipt_path = self._cache_paths(url, start, limit)
        if receipt_path.exists() and payload_path.exists():
            try:
                receipt = json.loads(receipt_path.read_text())
                size = receipt["bytes"]
                if not isinstance(size, int) or not 0 <= size <= limit:
                    raise ValueError("invalid cached size")
                if payload_path.stat().st_size != size:
                    raise ValueError("invalid cached payload size")
                payload = payload_path.read_bytes()
                if sha256(payload) != receipt["sha256"]:
                    raise ValueError("invalid cached payload digest")
                if receipt["url"] != url or receipt["start"] != start:
                    raise ValueError("invalid cached request identity")
                if start is not None:
                    expected = f"bytes {start}-{start + limit - 1}/{total}"
                    full = start == 0 and limit == total and receipt["status"] == 200
                    if (
                        size != limit
                        or not full
                        and (
                            receipt["status"] != 206
                            or receipt["content_range"] != expected
                        )
                    ):
                        raise ValueError("invalid cached Range")
                elif receipt["status"] != 200 or size == limit:
                    raise ValueError("invalid cached metadata response")
            except (ValueError, KeyError, TypeError) as exc:
                raise AcquisitionError(
                    f"corrupt acquisition cache: {payload_path}"
                ) from exc
            return payload, {**receipt, "cache_hit": True}
        for attempt in range(3):
            if self.ledger["charged_bytes"] + limit > self.cap:
                raise AcquisitionError(
                    "hard lifetime network byte cap exhausted; reuse cache or reduce acquisition"
                )
            record = {
                "url": url,
                "start": start,
                "reserved_bytes": limit,
                "charged_bytes": limit,
                "status": "reserved",
            }
            self.ledger["charged_bytes"] += limit
            self.ledger["attempts"].append(record)
            atomic_json(self.ledger_path, self.ledger)
            received = 0
            try:
                headers = {
                    "Accept-Encoding": "identity",
                    "User-Agent": "parameter-golf-ultradata/1",
                }
                if start is not None:
                    headers["Range"] = f"bytes={start}-{start + limit - 1}"
                with self.opener(Request(url, headers=headers), timeout=60) as response:
                    status = response.status
                    if (
                        response.headers.get("Content-Encoding", "identity")
                        != "identity"
                    ):
                        raise AcquisitionError(
                            "compressed response cannot satisfy byte-range receipts"
                        )
                    if start is not None:
                        expected = f"bytes {start}-{start + limit - 1}/{total}"
                        full = start == 0 and limit == total and status == 200
                        if not full and (
                            status != 206
                            or response.headers.get("Content-Range") != expected
                        ):
                            raise AcquisitionError(
                                "server ignored or changed the requested byte Range"
                            )
                    elif status != 200:
                        raise AcquisitionError(f"unexpected HTTP status {status}")
                    declared = response.headers.get("Content-Length")
                    if declared is not None:
                        try:
                            declared = int(declared)
                        except ValueError as exc:
                            raise AcquisitionError(
                                "invalid response Content-Length"
                            ) from exc
                        if declared < 0 or declared > limit:
                            raise AcquisitionError(
                                "response Content-Length exceeds request cap"
                            )
                    chunks = []
                    while received < limit:
                        chunk = response.read(min(65536, limit - received))
                        if not chunk:
                            break
                        received += len(chunk)
                        chunks.append(chunk)
                    payload = b"".join(chunks)
                    if start is not None and received != limit:
                        raise AcquisitionError("truncated range response")
                    if start is None and received == limit:
                        raise AcquisitionError("metadata response reached hard cap")
                    receipt = {
                        "url": url,
                        "start": start,
                        "bytes": received,
                        "sha256": sha256(payload),
                        "status": status,
                        "content_range": response.headers.get("Content-Range"),
                        "etag": response.headers.get("ETag"),
                    }
                temporary = payload_path.with_suffix(".tmp")
                temporary.write_bytes(payload)
                os.replace(temporary, payload_path)
                atomic_json(receipt_path, receipt)
                record["status"] = "complete"
                return payload, {**receipt, "cache_hit": False}
            except HTTPError as exc:
                exc.close()
                record["status"] = f"http_{exc.code}"
                if exc.code not in {408, 429, 500, 502, 503, 504} or attempt == 2:
                    raise AcquisitionError(
                        f"HTTP acquisition failed: {exc.code}"
                    ) from exc
            except (URLError, TimeoutError, ConnectionError) as exc:
                record["status"] = type(exc).__name__
                if attempt == 2:
                    raise AcquisitionError("bounded HTTP retries exhausted") from exc
            finally:
                # A thrown read may have consumed bytes before returning. Keep
                # its full reservation, rather than underestimate failed I/O.
                if record["status"] == "complete":
                    self.ledger["charged_bytes"] -= limit - received
                    record["charged_bytes"] = received
                atomic_json(self.ledger_path, self.ledger)
            time.sleep(2**attempt)
        raise AcquisitionError("bounded HTTP retries exhausted")


# Allocation guards, not dataset sampling policies. Callers can inspect metadata
# before deciding which complete row groups fit their acquisition/memory budgets.
MAX_PARQUET_READ_BYTES = 64 * 1024**2
MAX_PARQUET_DECODE_BYTES = 512 * 1024**2


class _RangeIO(io.RawIOBase):
    """Bounded selected ranges, coalescing adjacent cold columns without gaps."""

    def __init__(self, http, url: str, size: int, ranges, receipts: list[dict]):
        super().__init__()
        self.http = http
        self.url = url
        self.size = size
        self.ranges = tuple(ranges)
        self.receipts = receipts
        self.position = 0
        self.fetch_ranges = []
        for low, high in sorted(self.ranges):
            cold = not http.is_cached(url, start=low, limit=high - low)
            if (
                self.fetch_ranges
                and cold
                and self.fetch_ranges[-1][2]
                and self.fetch_ranges[-1][1] == low
                and high - self.fetch_ranges[-1][0] <= MAX_PARQUET_READ_BYTES
            ):
                self.fetch_ranges[-1] = (self.fetch_ranges[-1][0], high, True)
            else:
                self.fetch_ranges.append((low, high, cold))
        self.buffer_start = 0
        self.buffer = b""

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        self._checkClosed()
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        self._checkClosed()
        offset = operator.index(offset)
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.size + offset
        else:
            raise ValueError("invalid seek origin")
        if not 0 <= position <= self.size:
            raise AcquisitionError("Parquet seek outside declared file bounds")
        self.position = position
        return position

    def read(self, size=-1):
        self._checkClosed()
        size = operator.index(size)
        if size < 0:
            raise AcquisitionError("unbounded Parquet read is forbidden")
        end = self.position + size
        if end > self.size:
            raise AcquisitionError("Parquet read outside declared file bounds")
        if size > MAX_PARQUET_READ_BYTES:
            raise AcquisitionError("Parquet read exceeds safe allocation bound")
        if not size:
            return b""
        if not any(low <= self.position and end <= high for low, high in self.ranges):
            raise AcquisitionError("Parquet read outside selected column ranges")
        if (
            not self.buffer_start
            <= self.position
            <= end
            <= self.buffer_start + len(self.buffer)
        ):
            if self.http.is_cached(self.url, start=self.position, limit=size):
                low, high = self.position, end
            else:
                low, high, _ = next(
                    bounds
                    for bounds in self.fetch_ranges
                    if bounds[0] <= self.position and end <= bounds[1]
                )
                if high - low > MAX_PARQUET_READ_BYTES:
                    low, high = self.position, end
            payload, receipt = self.http.fetch(
                self.url, start=low, limit=high - low, total=self.size
            )
            if len(payload) != high - low:
                raise AcquisitionError("truncated Parquet range response")
            self.receipts.append(receipt)
            self.buffer_start, self.buffer = low, payload
        offset = self.position - self.buffer_start
        self.position = end
        return self.buffer[offset : offset + size]

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


class BoundedParquetFile:
    """Read only the footer and explicitly selected row-group column chunks.

    PyArrow decodes the standard Parquet format. Supplying parsed metadata avoids
    its initial speculative tail read, which otherwise downloads unrelated data.
    Every subsequent read is checked against the selected chunks before HTTP.
    """

    def __init__(self, http: BoundedHTTP, url: str, size: int):
        import pyarrow as pa
        import pyarrow.parquet as pq

        if not isinstance(size, int) or isinstance(size, bool) or size < 12:
            raise AcquisitionError("invalid Parquet file size")
        self.http = http
        self.url = url
        self.size = size
        self.receipts: list[dict] = []
        with _RangeIO(
            http, url, size, [(0, 4), (size - 8, size)], self.receipts
        ) as stream:
            magic = stream.read(4)
            stream.seek(-8, io.SEEK_END)
            trailer = stream.read(8)
        if magic != b"PAR1" or trailer[4:] != b"PAR1":
            raise AcquisitionError("invalid or unsupported Parquet magic")
        footer_size = int.from_bytes(trailer[:4], "little")
        self.footer_start = size - 8 - footer_size
        if footer_size <= 0 or self.footer_start < 4:
            raise AcquisitionError("invalid Parquet footer bounds")
        if footer_size > MAX_PARQUET_READ_BYTES:
            raise AcquisitionError("Parquet footer exceeds safe allocation bound")
        with _RangeIO(
            http, url, size, [(self.footer_start, size - 8)], self.receipts
        ) as stream:
            stream.seek(self.footer_start)
            footer = stream.read(footer_size)
        try:
            with pq.ParquetFile(
                pa.BufferReader(magic + footer + trailer),
                pre_buffer=False,
                thrift_string_size_limit=MAX_PARQUET_READ_BYTES,
                thrift_container_size_limit=1_000_000,
            ) as reader:
                self.metadata = reader.metadata
        except (pa.ArrowException, OSError, ValueError) as exc:
            raise AcquisitionError("invalid Parquet footer metadata") from exc

    def read_row_group(self, index: int, columns: list[str]):
        import pyarrow as pa
        import pyarrow.parquet as pq

        if not isinstance(index, int) or not 0 <= index < self.metadata.num_row_groups:
            raise ValueError("Parquet row-group index out of bounds")
        if not columns or any(
            not isinstance(name, str) or not name for name in columns
        ):
            raise ValueError("explicit nonempty Parquet columns are required")
        group = self.metadata.row_group(index)
        chunks = [group.column(i) for i in range(group.num_columns)]
        selected = []
        for name in columns:
            matches = [
                chunk
                for chunk in chunks
                if chunk.path_in_schema == name
                or chunk.path_in_schema.startswith(name + ".")
            ]
            if not matches:
                raise ValueError(f"unknown Parquet column: {name}")
            selected.extend(matches)
        # Overlapping root/leaf selectors may name the same physical chunk.
        selected = {chunk.path_in_schema: chunk for chunk in selected}.values()
        ranges = []
        decoded_bytes = 0
        for chunk in selected:
            offsets = [chunk.data_page_offset]
            if chunk.has_dictionary_page:
                offsets.append(chunk.dictionary_page_offset)
            start = min(offsets)
            length = chunk.total_compressed_size
            decoded_bytes += chunk.total_uncompressed_size
            if (
                chunk.file_path
                or start < 4
                or length <= 0
                or start + length > self.footer_start
                or chunk.total_uncompressed_size < 0
            ):
                raise AcquisitionError("invalid or external Parquet column chunk")
            if length > MAX_PARQUET_READ_BYTES:
                raise AcquisitionError("Parquet column exceeds safe allocation bound")
            ranges.append((start, start + length))
        if decoded_bytes > MAX_PARQUET_DECODE_BYTES:
            raise AcquisitionError(
                "Parquet row group exceeds safe decoded allocation bound"
            )
        try:
            with _RangeIO(
                self.http, self.url, self.size, ranges, self.receipts
            ) as stream:
                reader = pq.ParquetFile(
                    stream,
                    metadata=self.metadata,
                    pre_buffer=False,
                    buffer_size=0,
                    thrift_string_size_limit=MAX_PARQUET_READ_BYTES,
                    thrift_container_size_limit=1_000_000,
                )
                return reader.read_row_group(index, columns=columns, use_threads=False)
        except (pa.ArrowException, OSError) as exc:
            raise AcquisitionError("Parquet selected-column decoding failed") from exc
