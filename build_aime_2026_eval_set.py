"""Download AIME 2026 and build DAPO-compatible evaluation parquets.

The pinned ``math-ai/aime26`` source contains AIME I as IDs 1--15 and
AIME II as IDs 16--30.  This builder preserves the source JSONL, validates
the complete ID/answer sequence, and writes separate form files plus a
combined file.  Every parquet uses the same prompt/reward schema consumed by
``eval_aime.py``, ``sample_latent.py``, and the latent-VAPO evaluator.

Run model-data preprocessing through the repository's ML queue:

    mlq submit --name download_aime_2026 --cwd "$PWD" \
        --max-parallel-runs 1 -- python3 build_aime_2026_eval_set.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from build_math_mix_dataset import DAPO_PREAMBLE, DAPO_REMINDER

SOURCE_REPOSITORY = "math-ai/aime26"
SOURCE_REVISION = "04e80cb40b681c93511a913a6ba30529e6323832"
SOURCE_URL = (
    "https://huggingface.co/datasets/math-ai/aime26/resolve/"
    f"{SOURCE_REVISION}/aime2026.jsonl"
)
SOURCE_SHA256 = "52822957957a3f577d1e9706c36a66a8108a3f99b6aff424cfb72dff0094a9ee"
UPSTREAM_EXPECTED_ANSWERS = (
    277,
    62,
    79,
    70,
    65,
    441,
    396,
    244,
    29,
    156,
    896,
    161,
    39,
    681,
    83,
    178,
    243,
    503,
    279,
    190,
    50,
    754,
    245,
    669,
    850,
    132,
    223,
    107,
    157,
    393,
)
AUTHORIZED_EXPECTED_ANSWERS = (
    *UPSTREAM_EXPECTED_ANSWERS[:24],
    340,
    *UPSTREAM_EXPECTED_ANSWERS[25:],
)
AUTHORIZED_REFERENCES = {
    "aime_i": "https://live.poshenloh.com/past-contests/aime/2026I",
    "aime_ii": "https://live.poshenloh.com/past-contests/aime/2026II",
    "aime_i_problem_10": (
        "https://live.poshenloh.com/past-contests/aime/2026I/problem/10"
    ),
    "aime_i_problem_13": (
        "https://live.poshenloh.com/past-contests/aime/2026I/problem/13"
    ),
    "aime_ii_problem_10": (
        "https://live.poshenloh.com/past-contests/aime/2026II/problem/10"
    ),
}

SCHEMA = pa.schema(
    [
        ("data_source", pa.string()),
        (
            "prompt",
            pa.list_(pa.struct([("content", pa.string()), ("role", pa.string())])),
        ),
        ("ability", pa.string()),
        (
            "reward_model",
            pa.struct([("ground_truth", pa.string()), ("style", pa.string())]),
        ),
        (
            "extra_info",
            pa.struct(
                [
                    ("index", pa.string()),
                    ("raw_problem", pa.string()),
                    ("split", pa.string()),
                    ("year", pa.int16()),
                    ("form", pa.string()),
                    ("problem_number", pa.int8()),
                    ("source_id", pa.int8()),
                ]
            ),
        ),
    ]
)


def download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "parameter-golf/1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def parse_source(payload: bytes) -> list[dict]:
    source_sha256 = hashlib.sha256(payload).hexdigest()
    if source_sha256 != SOURCE_SHA256:
        raise ValueError(
            f"expected pinned source SHA-256 {SOURCE_SHA256}, found {source_sha256}"
        )

    rows = []
    for line_number, raw_line in enumerate(payload.splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise ValueError(f"source line {line_number} is invalid JSON") from error
        rows.append(row)

    if len(rows) != 30:
        raise ValueError(f"expected 30 source problems, found {len(rows)}")
    ids = [int(row["id"]) for row in rows]
    if ids != list(range(1, 31)):
        raise ValueError(f"expected ordered source IDs 1--30, found {ids}")
    answers = tuple(int(row["answer"]) for row in rows)
    if answers != UPSTREAM_EXPECTED_ANSWERS:
        raise ValueError(
            "source answer sequence differs from the reviewed AIME 2026 key: "
            f"{answers}"
        )
    for row in rows:
        if not isinstance(row.get("problem"), str) or not row["problem"].strip():
            raise ValueError(f"source problem {row.get('id')} is empty")
        answer = int(row["answer"])
        if not 0 <= answer <= 999:
            raise ValueError(f"source problem {row['id']} has invalid answer {answer}")

    # The source uses one global ID sequence.  Pin the two form boundaries so
    # an upstream reorder cannot silently swap or offset the forms.
    if not rows[0]["problem"].startswith("Patrick started walking"):
        raise ValueError("source ID 1 is not AIME I problem 1")
    if not rows[15]["problem"].startswith("Find the sum of the $10$th terms"):
        raise ValueError("source ID 16 is not AIME II problem 1")
    return rows


def replace_once(problem: str, old: str, new: str, *, source_id: int) -> str:
    occurrences = problem.count(old)
    if occurrences != 1:
        raise ValueError(
            f"source problem {source_id} contains correction target {occurrences} times; "
            "expected exactly once"
        )
    return problem.replace(old, new, 1)


def apply_authorized_corrections(source_rows: list[dict]) -> list[dict]:
    """Correct known upstream transcriptions against MAA-authorized pages."""
    rows = [dict(row) for row in source_rows]
    by_id = {int(row["id"]): row for row in rows}

    by_id[10]["problem"] = replace_once(
        by_id[10]["problem"],
        "so that ${}\\overline{AC}$ is perpendicular $\\overline{BC},$",
        "so that $\\overline{A'C'}$ is perpendicular to $\\overline{BC},$",
        source_id=10,
    )
    by_id[13]["problem"] = replace_once(
        by_id[13]["problem"],
        "For each positive integer $r$ less than $502,$ define",
        "For each nonnegative integer $r$ less than $502,$ define",
        source_id=13,
    )
    by_id[13]["problem"] = replace_once(
        by_id[13]["problem"],
        "S_r=\\sum_{m\\ge 0}\\dbinom{10000}{502n+r},",
        "S_r=\\sum_{m\\ge 0}\\dbinom{10000}{502m+r},",
        source_id=13,
    )
    by_id[25]["problem"] = replace_once(
        by_id[25]["problem"],
        "Find the sum of all possible values of $BC.$",
        "Find the greatest possible value of $BC.$",
        source_id=25,
    )
    by_id[25]["answer"] = 340

    answers = tuple(int(row["answer"]) for row in rows)
    if answers != AUTHORIZED_EXPECTED_ANSWERS:
        raise AssertionError(f"corrected answer sequence is invalid: {answers}")
    return rows


def convert(source_rows: list[dict]) -> list[dict]:
    converted = []
    for source in source_rows:
        source_id = int(source["id"])
        form = "I" if source_id <= 15 else "II"
        problem_number = source_id if form == "I" else source_id - 15
        problem = source["problem"].strip()
        converted.append(
            {
                "data_source": "aime_2026",
                "prompt": [
                    {
                        "content": (
                            f"{DAPO_PREAMBLE}\n\n{problem}\n\n{DAPO_REMINDER}"
                        ),
                        "role": "user",
                    }
                ],
                "ability": "MATH",
                "reward_model": {
                    "ground_truth": str(int(source["answer"])),
                    "style": "rule-lighteval/MATH_v2",
                },
                "extra_info": {
                    "index": f"aime_2026_{form.lower()}/{problem_number}",
                    "raw_problem": problem,
                    "split": "test",
                    "year": 2026,
                    "form": form,
                    "problem_number": problem_number,
                    "source_id": source_id,
                },
            }
        )
    return converted


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def atomic_write_parquet(path: Path, rows: list[dict]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), temporary, compression="zstd")
    temporary.replace(path)


def validate_outputs(
    *,
    raw_path: Path,
    corrected_source_path: Path,
    form_i_path: Path,
    form_ii_path: Path,
    combined_path: Path,
    payload: bytes,
    source_rows: list[dict],
    rows: list[dict],
) -> None:
    if raw_path.read_bytes() != payload:
        raise AssertionError("persisted raw source differs from downloaded payload")
    corrected_rows = [
        json.loads(line)
        for line in corrected_source_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if corrected_rows != source_rows:
        raise AssertionError("persisted corrected JSONL differs from corrected source rows")

    form_i = rows[:15]
    form_ii = rows[15:]
    for path, expected in (
        (form_i_path, form_i),
        (form_ii_path, form_ii),
        (combined_path, rows),
    ):
        table = pq.read_table(path)
        if table.schema != SCHEMA:
            raise AssertionError(f"{path} has unexpected schema: {table.schema}")
        if table.to_pylist() != expected:
            raise AssertionError(f"{path} contents differ from converted rows")

    indices = [row["extra_info"]["index"] for row in rows]
    if len(indices) != len(set(indices)):
        raise AssertionError("converted rows contain duplicate indices")
    answers = tuple(int(row["reward_model"]["ground_truth"]) for row in rows)
    if answers != AUTHORIZED_EXPECTED_ANSWERS:
        raise AssertionError(f"persisted answer sequence is invalid: {answers}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="postraining/data")
    parser.add_argument("--source-url", default=SOURCE_URL)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = download(args.source_url)
    upstream_rows = parse_source(payload)
    source_rows = apply_authorized_corrections(upstream_rows)
    rows = convert(source_rows)
    form_i = [row for row in rows if row["extra_info"]["form"] == "I"]
    form_ii = [row for row in rows if row["extra_info"]["form"] == "II"]
    if len(form_i) != 15 or len(form_ii) != 15:
        raise AssertionError("AIME 2026 forms must contain 15 problems each")

    raw_path = output_dir / "aime-2026-source.jsonl"
    corrected_source_path = output_dir / "aime-2026-corrected.jsonl"
    combined_path = output_dir / "aime-2026.parquet"
    form_i_path = output_dir / "aime-2026-i.parquet"
    form_ii_path = output_dir / "aime-2026-ii.parquet"
    atomic_write_bytes(raw_path, payload)
    corrected_payload = (
        "\n".join(json.dumps(row, ensure_ascii=False) for row in source_rows) + "\n"
    ).encode("utf-8")
    atomic_write_bytes(corrected_source_path, corrected_payload)
    atomic_write_parquet(form_i_path, form_i)
    atomic_write_parquet(form_ii_path, form_ii)
    atomic_write_parquet(combined_path, rows)
    validate_outputs(
        raw_path=raw_path,
        corrected_source_path=corrected_source_path,
        form_i_path=form_i_path,
        form_ii_path=form_ii_path,
        combined_path=combined_path,
        payload=payload,
        source_rows=source_rows,
        rows=rows,
    )

    manifest = {
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": SOURCE_REVISION,
        "source_url": args.source_url,
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "upstream_declared_license": "Apache-2.0",
        "copyright_note": (
            "Contest questions are MAA-copyrighted; the authorized references state "
            "that they reproduce the problems with MAA permission. Resolve redistribution "
            "rights before publishing these local files."
        ),
        "raw_file": raw_path.name,
        "corrected_source_file": corrected_source_path.name,
        "corrected_source_sha256": hashlib.sha256(corrected_payload).hexdigest(),
        "combined_file": combined_path.name,
        "form_i_file": form_i_path.name,
        "form_ii_file": form_ii_path.name,
        "problems": len(rows),
        "form_i_problems": len(form_i),
        "form_ii_problems": len(form_ii),
        "source_id_mapping": "1-15 = AIME I; 16-30 = AIME II",
        "template": "verbatim DAPO-Math-17K prompt wrapper",
        "upstream_corrections": [
            {
                "source_id": 10,
                "form_problem": "AIME I #10",
                "change": "AC -> A'C' and restored missing 'to' in perpendicular clause",
                "answer_changed": False,
            },
            {
                "source_id": 13,
                "form_problem": "AIME I #13",
                "change": "positive -> nonnegative r; 502n+r -> 502m+r",
                "answer_changed": False,
            },
            {
                "source_id": 25,
                "form_problem": "AIME II #10",
                "change": "sum of all possible values -> greatest possible value",
                "answer_changed": "850 -> 340",
            },
        ],
        "authorized_references": AUTHORIZED_REFERENCES,
        "answer_key_validated_against_authorized_references": True,
        "evaluation_limitations": [
            "AIME I #15's grid is represented as source text, not its original rendering.",
            "AIME II #2 embeds Asymptote source rather than a rendered figure.",
        ],
    }
    manifest_path = output_dir / "aime-2026.manifest.json"
    atomic_write_bytes(
        manifest_path,
        (json.dumps(manifest, indent=2) + "\n").encode("utf-8"),
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
