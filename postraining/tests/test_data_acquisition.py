from __future__ import annotations

import io
import json
import random

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from postraining.data_acquisition import (
    AcquisitionError,
    BoundedHTTP,
    BoundedParquetFile,
    MAX_PARQUET_READ_BYTES,
    _RangeIO,
    canonical_json,
    sha256,
)


URL = "https://example.com/pinned.parquet"


class Response(io.BytesIO):
    def __init__(self, payload, *, status=206, headers=None):
        super().__init__(payload)
        self.status = status
        self.headers = headers or {}
        self.bytes_read = 0

    def read(self, size=-1):
        result = super().read(size)
        self.bytes_read += len(result)
        return result


class RangeServer:
    def __init__(self, payload):
        self.payload = payload
        self.ranges = []

    def __call__(self, request, timeout):
        start, end = map(
            int, request.get_header("Range").removeprefix("bytes=").split("-")
        )
        assert 0 <= start <= end < len(self.payload)
        self.ranges.append((start, end + 1))
        return Response(
            self.payload[start : end + 1],
            headers={
                "Content-Length": str(end - start + 1),
                "Content-Range": f"bytes {start}-{end}/{len(self.payload)}",
            },
        )


def parquet_fixture():
    rng = random.Random(123)
    table = pa.table(
        {
            "name": ["first", "second", "third", "fourth"],
            "tests": [{"input": [str(i)], "output": [str(i + 1)]} for i in range(4)],
            "unused_solutions": [rng.randbytes(128 * 1024) for _ in range(4)],
        }
    )
    sink = io.BytesIO()
    pq.write_table(table, sink, row_group_size=2, compression=None)
    return table, sink.getvalue()


def test_projected_row_group_preserves_nested_data_without_downloading_other_columns(
    tmp_path,
):
    table, payload = parquet_fixture()
    server = RangeServer(payload)
    http = BoundedHTTP(tmp_path, cap=len(payload), opener=server)
    try:
        source = BoundedParquetFile(http, URL, len(payload))
        assert source.metadata.num_row_groups == 2
        projected = source.read_row_group(1, columns=["name", "tests"])
        assert projected.equals(table.slice(2, 2).select(["name", "tests"]))
        metadata = pq.read_metadata(io.BytesIO(payload))
        forbidden = []
        for group_index in range(metadata.num_row_groups):
            group = metadata.row_group(group_index)
            for column_index in range(group.num_columns):
                column = group.column(column_index)
                if group_index == 0 or column.path_in_schema == "unused_solutions":
                    start = column.data_page_offset
                    if column.has_dictionary_page:
                        start = min(start, column.dictionary_page_offset)
                    forbidden.append((start, start + column.total_compressed_size))
        assert all(
            stop <= low or start >= high
            for start, stop in server.ranges
            for low, high in forbidden
        )
        assert http.ledger["charged_bytes"] == sum(
            end - start for start, end in server.ranges
        )
        receipts = source.receipts
        assert all(
            receipt["sha256"]
            == sha256(payload[receipt["start"] : receipt["start"] + receipt["bytes"]])
            for receipt in receipts
        )
        network_before = list(server.ranges)
        again = BoundedParquetFile(http, URL, len(payload))
        assert again.read_row_group(1, columns=["name", "tests"]).equals(projected)
        assert server.ranges == network_before
        assert all(receipt["cache_hit"] for receipt in again.receipts)
    finally:
        http.close()


def test_range_stream_enforces_seeks_read_bounds_and_selected_intervals(tmp_path):
    server = RangeServer(b"0123456789")
    http = BoundedHTTP(tmp_path, cap=100, opener=server)
    try:
        with _RangeIO(http, URL, 10, [(2, 5), (8, 10)], []) as stream:
            assert stream.seek(2) == 2
            assert stream.read(3) == b"234"
            assert stream.tell() == 5
            with pytest.raises(AcquisitionError, match="selected"):
                stream.read(1)
            assert stream.tell() == 5
            assert stream.seek(-2, io.SEEK_END) == 8
            target = bytearray(2)
            assert stream.readinto(target) == 2
            assert target == b"89"
            assert stream.read(0) == b""
            with pytest.raises(AcquisitionError, match="bounds"):
                stream.read(1)
            with pytest.raises(AcquisitionError, match="unbounded"):
                stream.read()
            for offset, whence in [
                (-1, io.SEEK_SET),
                (1, io.SEEK_END),
                (-11, io.SEEK_CUR),
            ]:
                with pytest.raises(AcquisitionError, match="bounds"):
                    stream.seek(offset, whence)
            with pytest.raises(ValueError, match="origin"):
                stream.seek(0, 42)
        assert server.ranges == [(2, 5), (8, 10)]
        with pytest.raises(ValueError):
            stream.tell()
        with _RangeIO(
            http, URL, MAX_PARQUET_READ_BYTES + 1, [(0, MAX_PARQUET_READ_BYTES + 1)], []
        ) as stream:
            with pytest.raises(AcquisitionError, match="allocation"):
                stream.read(MAX_PARQUET_READ_BYTES + 1)
        assert server.ranges == [(2, 5), (8, 10)]
    finally:
        http.close()


@pytest.mark.parametrize(
    "payload",
    [
        b"BAD!" + b"\x01\x00\x00\x00PAR1",
        b"PAR1" + b"\x00\x00\x00\x00BAD!",
        b"PAR1" + b"\x00\x00\x00\x00PAR1",
        b"PAR1" + b"\xff\xff\xff\xffPAR1",
        b"PAR1invalid" + (7).to_bytes(4, "little") + b"PAR1",
    ],
)
def test_invalid_parquet_magic_and_footer_fail_closed(tmp_path, payload):
    server = RangeServer(payload)
    http = BoundedHTTP(tmp_path, cap=100, opener=server)
    try:
        with pytest.raises(AcquisitionError):
            BoundedParquetFile(http, URL, len(payload))
    finally:
        http.close()


def test_invalid_projection_does_not_download_row_data(tmp_path):
    _, payload = parquet_fixture()
    server = RangeServer(payload)
    http = BoundedHTTP(tmp_path, cap=len(payload), opener=server)
    try:
        source = BoundedParquetFile(http, URL, len(payload))
        before = list(server.ranges)
        for index, columns in [
            (-1, ["name"]),
            (2, ["name"]),
            (0, []),
            (0, ["missing"]),
        ]:
            with pytest.raises(ValueError):
                source.read_row_group(index, columns)
        assert server.ranges == before
    finally:
        http.close()


def test_cache_corruption_is_not_silently_refetched(tmp_path):
    server = RangeServer(b"abcdefgh")
    http = BoundedHTTP(tmp_path, cap=100, opener=server)
    try:
        _, receipt = http.fetch(URL, start=2, limit=3, total=8)
        key = sha256(canonical_json({"url": URL, "start": 2, "limit": 3}))
        (tmp_path / (key + ".bin")).write_bytes(b"bad")
        with pytest.raises(AcquisitionError, match="corrupt"):
            http.fetch(URL, start=2, limit=3, total=8)
        assert server.ranges == [(2, 5)]
        assert http.ledger["charged_bytes"] == receipt["bytes"]
    finally:
        http.close()


def test_cached_range_cannot_be_reused_with_a_different_total(tmp_path):
    server = RangeServer(b"abcdefgh")
    http = BoundedHTTP(tmp_path, cap=100, opener=server)
    try:
        http.fetch(URL, start=2, limit=3, total=8)
        with pytest.raises(AcquisitionError, match="corrupt"):
            http.fetch(URL, start=2, limit=3, total=9)
        assert server.ranges == [(2, 5)]
    finally:
        http.close()


def test_failed_read_keeps_full_lifetime_reservation(tmp_path):
    class BrokenResponse(Response):
        def read(self, size=-1):
            super().read(size)
            raise ConnectionError("connection lost after consuming response bytes")

    response = BrokenResponse(b"abcd", headers={"Content-Range": "bytes 0-3/8"})
    http = BoundedHTTP(tmp_path, cap=4, opener=lambda request, timeout: response)
    try:
        # The first attempt consumes the entire cap; retry cannot open a response.
        with pytest.raises(AcquisitionError, match="cap exhausted"):
            http.fetch(URL, start=0, limit=4, total=8)
        assert response.closed
        assert response.bytes_read == 4
    finally:
        http.close()
    ledger = json.loads((tmp_path / "network.json").read_text())
    assert ledger["charged_bytes"] == 4
    reopened = BoundedHTTP(
        tmp_path, cap=4, opener=lambda request, timeout: pytest.fail("network retry")
    )
    try:
        with pytest.raises(AcquisitionError, match="cap exhausted"):
            reopened.fetch(URL, start=0, limit=4, total=8)
    finally:
        reopened.close()


def test_larger_source_budget_has_no_fixed_one_gib_ceiling(tmp_path):
    response = Response(b"{}", status=200)
    http = BoundedHTTP(
        tmp_path, cap=2 * 1024**3, opener=lambda request, timeout: response
    )
    try:
        assert http.fetch(URL, limit=2 * 1024**3)[0] == b"{}"
        assert http.ledger["charged_bytes"] == 2
    finally:
        http.close()


@pytest.mark.parametrize(
    "headers,payload",
    [
        ({"Content-Range": "bytes 1-4/8"}, b"abcd"),
        ({"Content-Range": "bytes 0-3/8", "Content-Encoding": "gzip"}, b"abcd"),
        ({"Content-Range": "bytes 0-3/8"}, b"abc"),
    ],
)
def test_changed_encoded_and_truncated_ranges_never_enter_cache(
    tmp_path, headers, payload
):
    response = Response(payload, headers=headers)
    http = BoundedHTTP(tmp_path, cap=4, opener=lambda request, timeout: response)
    try:
        with pytest.raises(AcquisitionError):
            http.fetch(URL, start=0, limit=4, total=8)
        assert http.ledger["charged_bytes"] == 4
        assert response.closed
        assert list(tmp_path.glob("*.bin")) == []
    finally:
        http.close()


def test_adjacent_selected_columns_share_one_request_without_reading_a_gap(tmp_path):
    server = RangeServer(b"0123456789secret")
    http = BoundedHTTP(tmp_path, cap=100, opener=server)
    try:
        with _RangeIO(http, URL, 16, [(0, 2), (2, 4), (4, 6), (8, 10)], []) as stream:
            assert stream.read(2) == b"01"
            assert stream.read(2) == b"23"
            assert stream.read(2) == b"45"
            stream.seek(8)
            assert stream.read(2) == b"89"
        assert server.ranges == [(0, 6), (8, 10)]
        assert http.ledger["charged_bytes"] == 8
    finally:
        http.close()


def test_batching_reuses_older_individual_column_cache(tmp_path):
    server = RangeServer(b"012345")
    http = BoundedHTTP(tmp_path, cap=100, opener=server)
    try:
        http.fetch(URL, start=0, limit=2, total=6)
        with _RangeIO(http, URL, 6, [(0, 2), (2, 4), (4, 6)], []) as stream:
            assert [stream.read(2) for _ in range(3)] == [b"01", b"23", b"45"]
        assert server.ranges == [(0, 2), (2, 6)]
        assert http.ledger["charged_bytes"] == 6
    finally:
        http.close()
