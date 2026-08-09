"""Fail-closed authorization for OPSD after both frozen teacher gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from postraining.opsd.data import file_sha256
from postraining.opsd.data import TEACHER_PROMPT_SCHEMA
from postraining.opsd.schemas import OPSD_PROMPT_SCHEMA
from postraining.opsd.teacher_logit_gate import TEACHER_LOGIT_GATE_SCHEMA
from postraining.opsd.teacher_uplift import TEACHER_UPLIFT_SCHEMA
from postraining.opsd.teacher_uplift import atomic_json


OPSD_AUTHORIZATION_SCHEMA = "opsd_dual_frozen_gate_authorization/v2"
AUTHORIZATION_GATE_CONTRACT = {
    "generation_schema": TEACHER_UPLIFT_SCHEMA,
    "logit_schema": TEACHER_LOGIT_GATE_SCHEMA,
    "logit_panel_role": "authorization",
    "conditioning_method": "answer_only_frozen_self_rationalization_extension",
    "teacher_prompt_schema": TEACHER_PROMPT_SCHEMA,
    "opsd_prompt_schema": OPSD_PROMPT_SCHEMA,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--generation-gate", required=True)
    parser.add_argument("--logit-gate", required=True)
    args = parser.parse_args()
    output = Path("postraining/runs") / args.name
    if output.exists():
        parser.error(f"refusing to overwrite authorization {output}")
    generation_path = Path(args.generation_gate)
    logit_path = Path(args.logit_gate)
    generation = json.loads(generation_path.read_text())
    logit = json.loads(logit_path.read_text())
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
    if generation.get("opsd_prompt_schema") != logit.get("opsd_prompt_schema"):
        parser.error("frozen gates used different OPSD prompt schemas")
    if generation.get("teacher_prompt_schema") != TEACHER_PROMPT_SCHEMA:
        parser.error("generation gate used a different teacher prompt schema")
    if generation.get("opsd_prompt_schema") != OPSD_PROMPT_SCHEMA:
        parser.error("frozen gates used a different OPSD prompt schema")
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
        "gate_contract": dict(AUTHORIZATION_GATE_CONTRACT),
        "input_artifacts": {
            "generation_gate": {
                "path": str(generation_path),
                "sha256": file_sha256(generation_path),
            },
            "logit_gate": {
                "path": str(logit_path),
                "sha256": file_sha256(logit_path),
            },
        },
    }
    output.mkdir(parents=True)
    atomic_json(result, output / "results.json")
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if decision == "pass" else 2)


if __name__ == "__main__":
    main()
