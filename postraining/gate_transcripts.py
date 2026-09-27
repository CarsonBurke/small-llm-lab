"""Complete frozen-policy gate evidence, including unsuccessful completions."""

import json
import threading
from pathlib import Path

from postraining.latent_rollout import emitted_token_rows


class GateTranscriptWriter:
    def __init__(self, path: Path, tokenizer, stop_ids):
        self.path = path
        # A resumed evaluation must use a new evidence destination; appending
        # repeat zero again would make prompt/sample identities ambiguous.
        with self.path.open("x"):
            pass
        self.tokenizer = tokenizer
        self.stop_ids = frozenset(stop_ids)
        self.lock = threading.Lock()
        self.repeat = 0

    def write(self, batch, verdicts, row):
        emitted = emitted_token_rows(batch)
        rewards = batch.reward_scalar.tolist()
        records = []
        for index, (tokens, verdict, reward) in enumerate(
            zip(emitted, verdicts, rewards, strict=True)
        ):
            records.append({
                "schema": "frozen_policy_gate_transcript/v1",
                "repeat": self.repeat,
                "source": row.get("_rl_source", "math"),
                "extra_info": row.get("extra_info"),
                "prompt": row["prompt"],
                "reward_model": row["reward_model"],
                "verification_info": row.get("verification_info"),
                "sample": index,
                "reward": float(reward),
                "structural_format_ok": bool(verdict.format_ok),
                "parsed_answer": verdict.parsed_answer,
                "terminated": any(token in self.stop_ids for token in tokens),
                "token_ids": tokens,
                "text": self.tokenizer.decode(tokens),
            })
        payload = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
        with self.lock, self.path.open("a") as stream:
            stream.write(payload)
