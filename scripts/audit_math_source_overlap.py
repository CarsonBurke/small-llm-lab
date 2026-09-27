"""Audit DAPO/UltraData RL overlap; broader retrieval proposes pairs, never deletes rows.

Run through mlq. This is a cross-source RL audit, not SFT/pretraining exposure.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import difflib
import json
from pathlib import Path
import re

import numpy as np
from scipy import sparse

from postraining.core import load_unique_math_rows
from postraining.math_rl_pool import canonical_problem, file_sha256
from postraining.problem_overlap import (
    ProblemOverlapIndex, number_multiset, numbers_contained, overlap_tokens,
    skeleton_key, template_shingles,
)


def numeric_skeleton(text):
    """Candidate retrieval only: discards numerical values but keeps their positions."""
    return re.sub(r"\d+(?:\.\d+)?", "#", skeleton_key(text))


def char_features(text):
    clean = " ".join(overlap_tokens(text))
    return {clean[i:i + 3] for i in range(len(clean) - 2)}


def cosine_candidates(left, right, *, threshold=0.5, top_k=3):
    """Symmetric character-trigram TF-IDF cosine, queried in bounded blocks.

    IDF fitted on both pools. Features present in >10% of all rows are
    ignored; this is only a broad candidate retriever, never identity proof.
    """
    features = [char_features(text) for text in left + right]
    counts = Counter(feature for row in features for feature in row)
    vocab = {feature: i for i, (feature, count) in enumerate(sorted(counts.items()))
             if count <= len(features) * 0.1}
    # Reindex after frequency filtering to avoid unnecessarily wide matrices.
    vocab = {feature: i for i, feature in enumerate(vocab)}
    indices, values, indptr = [], [], [0]
    for row in features:
        for feature in sorted(row):
            if feature in vocab:
                indices.append(vocab[feature])
                values.append(np.log((1 + len(features)) / (1 + counts[feature])) + 1)
        indptr.append(len(indices))
    matrix = sparse.csr_matrix((np.asarray(values, dtype=np.float32), indices, indptr),
                               shape=(len(features), len(vocab)))
    norm = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    matrix = sparse.diags(1 / np.maximum(norm, 1e-12)).dot(matrix).tocsr()
    lhs, rhs = matrix[:len(left)], matrix[len(left):]
    found = {}
    # Top-k in both directions prevents one heavily templated side hiding a
    # candidate which is a close neighbor only from the other side.
    for reverse, queries, reference in [(False, lhs, rhs), (True, rhs, lhs)]:
        for start in range(0, queries.shape[0], 64):
            scores = (queries[start:start + 64] @ reference.T).toarray()
            for local, row in enumerate(scores):
                selected = np.flatnonzero(row >= threshold)
                selected = sorted(selected, key=lambda i: (-float(row[i]), int(i)))[:top_k]
                for other in selected:
                    pair = (int(other), start + local) if reverse else (start + local, int(other))
                    found[pair] = float(row[other])
    return found


def load_pool(path):
    rows = load_unique_math_rows(path)
    return rows, [canonical_problem(row) for row in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dapo", type=Path, default=Path("postraining/data/dapo-math-17k-v2-ud3dedup.parquet"))
    parser.add_argument("--ultradata", type=Path, default=Path("postraining/data/ultradata-math-rl-v5.parquet"))
    parser.add_argument("--reviewed-manifest", type=Path, default=Path("postraining/data/reviewed_candidate_20260926_v1/mixture.manifest.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    dapo_rows, dapo = load_pool(args.dapo)
    ud_rows, ud = load_pool(args.ultradata)
    pools = {"dapo": (args.dapo, dapo_rows), "ultradata_math": (args.ultradata, ud_rows)}
    provenance = {}
    for name, (path, rows) in pools.items():
        manifest = json.loads(path.with_suffix(".manifest.json").read_text())
        digest = file_sha256(path)
        if digest != manifest["output_sha256"] or len(rows) != manifest["problems"]:
            raise ValueError(f"pool no longer matches manifest: {path}")
        provenance[name] = {"path": str(path), "sha256": digest, "rows": len(rows), "builder_counts": manifest["counts"]}
    template = template_shingles([dapo, ud])
    official = {}
    forward = ProblemOverlapIndex(ud, template=template)
    for i, text in enumerate(dapo):
        for match in forward.matches(text):
            official[i, match.reference] = {"matcher": match.matcher, "containment": match.containment, "shared_shingles": match.shared_shingles}
    backward = ProblemOverlapIndex(dapo, template=template)
    reverse_pairs = {(match.reference, j) for j, text in enumerate(ud) for match in backward.matches(text)}
    if reverse_pairs != official.keys():
        raise ValueError("production matcher unexpectedly asymmetric")
    print(json.dumps({"phase": "production_matcher", "pairs": len(official)}), flush=True)
    candidates = cosine_candidates(dapo, ud)
    skeletons = defaultdict(list)
    for j, text in enumerate(ud):
        skeletons[numeric_skeleton(text)].append(j)
    numeric_pairs = {(i, j) for i, text in enumerate(dapo) for j in skeletons.get(numeric_skeleton(text), [])}
    pairs = set(candidates) | numeric_pairs | set(official)
    reviewed_manifest = json.loads(args.reviewed_manifest.read_text())
    reviewed_ids = {}
    for entry in reviewed_manifest["sources"]:
        if entry["name"] not in pools:
            continue
        path = Path(entry["path"])
        if file_sha256(path) != entry["sha256"]:
            raise ValueError(f"reviewed source changed: {path}")
        rows, _ = load_pool(path)
        reviewed_ids[entry["name"]] = {str(row["extra_info"]["index"]) for row in rows}
    records = []
    for i, j in sorted(pairs, key=lambda pair: (-candidates.get(pair, 0), pair)):
        left, right = dapo_rows[i], ud_rows[j]
        left_id, right_id = str(left["extra_info"]["index"]), str(right["extra_info"]["index"])
        records.append({
            "dapo_id": left_id, "ultradata_id": right_id,
            "dapo_problem": dapo[i], "ultradata_problem": ud[j],
            "dapo_gold": left["reward_model"]["ground_truth"], "ultradata_gold": right["reward_model"]["ground_truth"],
            "production_match": official.get((i, j)), "char_trigram_cosine": candidates.get((i, j)),
            "numeric_skeleton_equal": (i, j) in numeric_pairs,
            "digit_guard_passes": numbers_contained(number_multiset(dapo[i]), number_multiset(ud[j])),
            "symmetric_sequence_ratio": (difflib.SequenceMatcher(None, dapo[i], ud[j], autojunk=False).ratio() + difflib.SequenceMatcher(None, ud[j], dapo[i], autojunk=False).ratio()) / 2,
            "both_in_reviewed_candidate": left_id in reviewed_ids['dapo'] and right_id in reviewed_ids['ultradata_math'],
            "status": "candidate_requires_semantic_review",
        })
    result = {"schema": "math_source_overlap_audit/v1", "scope": "Cross-source DAPO versus UltraData RL pools only; not SFT/pretraining exposure or eval contamination.",
              "pools": provenance, "reviewed_manifest": str(args.reviewed_manifest), "reviewed_manifest_sha256": file_sha256(args.reviewed_manifest),
              "reviewed_rows": {k: len(v) for k, v in reviewed_ids.items()},
              "production_matcher": {"pairs": len(official), "by_matcher": dict(Counter(x['matcher'] for x in official.values())), "symmetric": True, "template_shingles": len(template)},
              "broader_retrieval": {"pairs": len(records), "cosine_threshold": 0.5, "top_k_each_direction": 3, "numeric_skeleton_pairs": len(numeric_pairs)},
              "limitations": "Lexical retrieval misses translations and sufficiently different paraphrases. Similarity and numeric skeletons are candidates, not duplicate labels. Different numeric parameters or target transformations may define distinct tasks.",
              "pairs": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "pairs"}), flush=True)


if __name__ == "__main__":
    main()
