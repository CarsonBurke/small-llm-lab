"""Fail-closed authorization for OPSD after both frozen teacher gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from postraining.opsd.teacher_logit_gate import TEACHER_LOGIT_GATE_SCHEMA
from postraining.opsd.teacher_uplift import TEACHER_UPLIFT_SCHEMA
from postraining.opsd.teacher_uplift import atomic_json


OPSD_AUTHORIZATION_SCHEMA = "opsd_dual_frozen_gate_authorization/v1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--generation-gate", required=True)
    parser.add_argument("--logit-gate", required=True)
    args = parser.parse_args()
    output = Path("postraining/runs") / args.name
    if output.exists():
        parser.error(f"refusing to overwrite authorization {output}")
    generation = json.loads(Path(args.generation_gate).read_text())
    logit = json.loads(Path(args.logit_gate).read_text())
    if generation.get("schema") != TEACHER_UPLIFT_SCHEMA:
        parser.error("generation gate has an incompatible schema")
    if logit.get("schema") != TEACHER_LOGIT_GATE_SCHEMA:
        parser.error("logit gate has an incompatible schema")
    if (logit.get("args") or {}).get("panel_role") != "authorization":
        parser.error("development logit panels cannot authorize OPSD training")
    if (logit.get("conditioning") or {}).get("method") != (
        "answer_only_frozen_self_rationalization_extension"
    ):
        parser.error("logit gate did not test the self-rationalized teacher")
    if generation["checkpoint_sha256"] != logit["checkpoint_sha256"]:
        parser.error("frozen gates used different checkpoints")
    if generation["gate_sha256"] != logit["gate_sha256"]:
        parser.error("frozen gates used different held-out data")
    if generation.get("data_manifest_sha256") != logit.get(
        "data_manifest_sha256"
    ):
        parser.error("frozen gates used different data manifests")
    decisions = {
        "generation": generation["decision"],
        "logit": logit["decision"],
    }
    decision = "pass" if set(decisions.values()) == {"pass"} else "fail"
    result = {
        "schema": OPSD_AUTHORIZATION_SCHEMA,
        "decision": decision,
        "gate_decisions": decisions,
        "checkpoint_sha256": generation["checkpoint_sha256"],
        "gate_sha256": generation["gate_sha256"],
        "data_manifest_sha256": generation["data_manifest_sha256"],
        "inputs": {
            "generation_gate": args.generation_gate,
            "logit_gate": args.logit_gate,
        },
    }
    output.mkdir(parents=True)
    atomic_json(result, output / "results.json")
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if decision == "pass" else 2)


if __name__ == "__main__":
    main()
