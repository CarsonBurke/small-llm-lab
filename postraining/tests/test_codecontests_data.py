from __future__ import annotations

from copy import deepcopy
import io
import re

import pytest

from postraining import codecontests_data as adapter
from postraining.data_acquisition import AcquisitionError, BoundedHTTP, canonical_json
from postraining.verifiable_tasks import MAX_CASE_BYTES, MAX_TEST_BYTES, MAX_TEST_CASES


FIRST_TRAIN_FILE = "data/train-00000-of-00039-e991a271dbfa9925.parquet"


def problem():
    return {
        "name": "123_A. Sum",
        "description": "Read two integers from standard input. Print their sum to standard output.",
        "public_tests": {"input": ["1 2\n"], "output": ["3\n"]},
        "private_tests": {"input": ["40 2\n", "-4 8\n"], "output": ["42\n", "4\n"]},
        "generated_tests": {"input": ["1000 27\n"], "output": ["1027\n"]},
        "source": 2,
        "input_file": "",
        "output_file": "",
        "cf_contest_id": 123,
        "cf_index": "A",
        "cf_tags": ["math"],
        "cf_rating": 800,
        "difficulty": 7,
        "is_description_translated": False,
        "time_limit": {"seconds": 2, "nanos": 0},
        "memory_limit_bytes": 256000000,
        "solutions": {
            "language": [3],
            "solution": ["print(sum(map(int, input().split())))"],
        },
        "incorrect_solutions": {"language": [3], "solution": ["print(0)"]},
    }


def normalize(row, **location):
    return adapter.normalize_codecontests_row(
        row,
        **(
            {
                "file": FIRST_TRAIN_FILE,
                "row_group": 2,
                "row_index": 7,
                "file_row_index": 107,
            }
            | location
        ),
    )


def tree(size=180227735):
    entries = [
        {
            "type": "file",
            "path": f"data/train-{index:05d}-of-00039-{index:016x}.parquet",
            "size": size,
        }
        for index in range(39)
    ]
    entries[0]["path"] = FIRST_TRAIN_FILE
    return entries + [
        {
            "type": "file",
            "path": "data/valid-00000-of-00001-5e672c5751f060d3.parquet",
            "size": size,
        },
        {
            "type": "file",
            "path": "data/test-00000-of-00001-9c49eeff30aacaa8.parquet",
            "size": size,
        },
        {"type": "file", "path": "README.md", "size": 13010},
    ]


def test_full_suite_is_preserved_in_order_outside_query():
    original = problem()
    before = deepcopy(original)
    row = normalize(original)
    assert row["ground_truth"] == {
        "call_type": "std",
        "fn_name": None,
        "inputs": ["1 2\n", "40 2\n", "-4 8\n", "1000 27\n"],
        "outputs": ["3\n", "42\n", "4\n", "1027\n"],
    }
    assert "1000 27" not in row["query"]
    assert "40 2" not in row["query"]
    assert row["provenance"]["suite_counts"] == {
        "public_tests": 1,
        "private_tests": 2,
        "generated_tests": 1,
    }
    assert original == before
    changed = deepcopy(original)
    changed["private_tests"]["output"][0] = "999\n"
    assert (
        normalize(changed)["provenance"]["full_suite_sha256"]
        != row["provenance"]["full_suite_sha256"]
    )


def test_original_sources_and_exact_locations_remain_traceable():
    identities = []
    for source, name in enumerate(adapter.SOURCES):
        raw = problem()
        raw["source"] = source
        row = normalize(raw, row_index=source, file_row_index=100 + source)
        provenance = row["provenance"]
        assert provenance["original_source"] == name
        assert provenance["original_source_id"] == source
        assert provenance["file"] == FIRST_TRAIN_FILE
        assert provenance["row_group"] == 2
        assert provenance["row_index"] == source
        assert provenance["file_row_index"] == 100 + source
        assert provenance["split"] == "train"
        assert provenance["license"] == "CC-BY-4.0"
        assert row["source_revision"] == adapter.REVISION
        identities.append(row["uuid"])
    assert len(set(identities)) == len(adapter.SOURCES)


@pytest.mark.parametrize("suite", adapter.SUITES)
def test_missing_null_or_mismatched_suite_rejects_whole_problem(suite):
    raw = problem()
    del raw[suite]
    with pytest.raises(ValueError, match=f"missing_{suite}"):
        normalize(raw)
    raw[suite] = None
    with pytest.raises(ValueError, match=f"missing_{suite}"):
        normalize(raw)
    raw[suite] = {"input": ["1\n"], "output": []}
    with pytest.raises(ValueError, match=f"invalid_{suite}"):
        normalize(raw)
    raw[suite] = {"input": None, "output": None}
    with pytest.raises(ValueError, match=f"invalid_{suite}"):
        normalize(raw)


def test_public_only_is_rejected_but_explicit_empty_generated_suite_is_valid():
    raw = problem()
    raw["generated_tests"] = {"input": [], "output": []}
    assert normalize(raw)["ground_truth"]["inputs"] == ["1 2\n", "40 2\n", "-4 8\n"]
    raw["private_tests"] = {"input": [], "output": []}
    with pytest.raises(ValueError, match="public_only_tests"):
        normalize(raw)


@pytest.mark.parametrize("value", [None, 42, "", " \n"])
def test_invalid_hidden_output_is_not_dropped(value):
    raw = problem()
    raw["private_tests"]["output"][1] = value
    with pytest.raises(ValueError, match="invalid_private_tests"):
        normalize(raw)


def test_test_count_boundary_rejects_instead_of_truncating():
    raw = problem()
    raw["private_tests"] = {
        "input": ["1\n"] * (MAX_TEST_CASES - 2),
        "output": ["1\n"] * (MAX_TEST_CASES - 2),
    }
    assert len(normalize(raw)["ground_truth"]["inputs"]) == MAX_TEST_CASES
    raw["generated_tests"]["input"].append("2\n")
    raw["generated_tests"]["output"].append("2\n")
    with pytest.raises(ValueError, match="too_many_test_cases"):
        normalize(raw)
    assert len(raw["private_tests"]["input"]) == MAX_TEST_CASES - 2


def test_case_bytes_are_utf8_bounded_without_whole_record_cap():
    raw = problem()
    raw["private_tests"]["input"][0] = "é" * (MAX_CASE_BYTES // 2)
    row = normalize(raw)
    assert len(row["ground_truth"]["inputs"][1].encode()) == MAX_CASE_BYTES
    assert len(canonical_json(raw)) > MAX_CASE_BYTES
    raw["private_tests"]["input"][0] += "é"
    with pytest.raises(ValueError, match="test_case_bytes_exceeded"):
        normalize(raw)


def test_total_suite_bytes_rejects_without_discarding_cases():
    raw = problem()
    raw["public_tests"] = {"input": [], "output": []}
    raw["generated_tests"] = {"input": [], "output": []}
    count = MAX_TEST_BYTES // (2 * MAX_CASE_BYTES)
    raw["private_tests"] = {
        "input": ["x" * MAX_CASE_BYTES] * count,
        "output": ["y" * MAX_CASE_BYTES] * count,
    }
    assert normalize(raw)["provenance"]["test_bytes"] == MAX_TEST_BYTES
    raw["generated_tests"] = {"input": [""], "output": ["1"]}
    with pytest.raises(ValueError, match="test_suite_bytes_exceeded"):
        normalize(raw)


@pytest.mark.parametrize(
    "description,reason",
    [
        ("This is an interactive problem.", "interactive"),
        ("Output any valid permutation.", "multiple_valid_outputs"),
        (
            "If there are several solutions, print one of them.",
            "multiple_valid_outputs",
        ),
        (
            "Your absolute or relative error must not exceed 1e-6.",
            "custom_checker_or_tolerance",
        ),
        ("The special judge checks your answer.", "custom_checker_or_tolerance"),
        ("Input file: sums.in\nOutput file: sums.out", "non_stdio"),
    ],
)
def test_identifiable_unsupported_judges_are_excluded(description, reason):
    raw = problem()
    raw["description"] = description
    with pytest.raises(ValueError, match=reason):
        normalize(raw)


def test_file_io_and_missing_contract_are_rejected():
    raw = problem()
    raw["input_file"] = "sums.in"
    with pytest.raises(ValueError, match="non_stdio"):
        normalize(raw)
    raw["input_file"] = None
    with pytest.raises(ValueError, match="missing_io_contract"):
        normalize(raw)


def test_selection_is_reproducible_stratified_and_train_only():
    entries = tree()
    selected = adapter.select_train_shards(entries, seed=1337, max_shards=12)
    assert selected == adapter.select_train_shards(
        list(reversed(entries)), seed=1337, max_shards=12
    )
    indices = [int(entry["path"].split("-")[1]) for entry in selected]
    assert all(
        i * 39 // 12 <= index < (i + 1) * 39 // 12 for i, index in enumerate(indices)
    )
    assert (
        adapter.select_train_shards(entries, seed=1337, max_shards=39) == entries[:39]
    )
    with pytest.raises(AcquisitionError, match="all 39"):
        adapter.select_train_shards(entries[1:], seed=1337, max_shards=12)


class Response(io.BytesIO):
    def __init__(self, payload, *, status, headers):
        super().__init__(payload)
        self.status = status
        self.headers = headers


def test_acquisition_projects_columns_preserves_hidden_cases_and_reports_rejections(
    tmp_path,
):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    valid = problem()
    # Distinct, sizeable solution chunks make unwanted solution reads observable.
    valid["solutions"]["solution"] = ["correct reference " * 1000]
    valid["incorrect_solutions"]["solution"] = ["wrong reference " * 1000]
    invalid = deepcopy(valid)
    invalid["private_tests"]["output"] = []
    buffer = io.BytesIO()
    pq.write_table(
        pa.Table.from_pylist([valid, invalid]),
        buffer,
        compression="NONE",
        row_group_size=1,
    )
    payload = buffer.getvalue()
    metadata = pq.read_metadata(io.BytesIO(payload))
    forbidden = []
    for i in range(metadata.num_row_groups):
        group = metadata.row_group(i)
        for j in range(group.num_columns):
            column = group.column(j)
            if column.path_in_schema.split(".", 1)[0] in (
                "solutions",
                "incorrect_solutions",
            ):
                start = column.dictionary_page_offset
                if start is None or start < 0:
                    start = column.data_page_offset
                forbidden.append((start, start + column.total_compressed_size))
    entries = tree(size=len(payload))
    selected = adapter.select_train_shards(entries, seed=1337, max_shards=2)
    selected_urls = {
        f"https://huggingface.co/datasets/{adapter.DATASET}/resolve/{adapter.REVISION}/{item['path']}"
        for item in selected
    }

    def opener(request, timeout):
        if request.full_url == adapter.TREE_URL:
            body = canonical_json(entries)
            return Response(
                body, status=200, headers={"Content-Length": str(len(body))}
            )
        if request.full_url == adapter.CARD_URL:
            body = b"---\nlicense:\n- cc-by-4.0\n---\n"
            return Response(
                body, status=200, headers={"Content-Length": str(len(body))}
            )
        assert request.full_url in selected_urls
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", request.get_header("Range"))
        assert match is not None
        start, end = map(int, match.groups())
        assert all(end < low or start >= high for low, high in forbidden)
        body = payload[start : end + 1]
        return Response(
            body,
            status=206,
            headers={
                "Content-Length": str(len(body)),
                "Content-Range": f"bytes {start}-{end}/{len(payload)}",
            },
        )

    http = BoundedHTTP(tmp_path, cap=4 * 1024**2 + 8 * len(payload), opener=opener)
    try:
        rows, manifest = adapter.acquire_codecontests(http, max_shards=2)
    finally:
        http.close()
    assert len(rows) == 2
    assert {row["provenance"]["file"] for row in rows} == {
        entry["path"] for entry in selected
    }
    assert all(
        row["ground_truth"]["outputs"] == ["3\n", "42\n", "4\n", "1027\n"]
        for row in rows
    )
    assert manifest["selected_source_rows"] == 4
    assert manifest["filter_counts"] == {
        "examined_rows": 4,
        "accepted_rows": 2,
        "invalid_private_tests": 2,
    }
    for file in manifest["source_files"]:
        assert [group["accepted_rows"] for group in file["row_groups"]] == [1, 0]
        assert file["download_receipts"]
        assert file["row_groups"][1]["filter_counts"] == {"invalid_private_tests": 1}
