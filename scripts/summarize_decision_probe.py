#!/usr/bin/env python3
"""Per-step summary of a decision-state probe file (scripts/probe_mac_decision_states.py)."""
import json
import sys
from collections import defaultdict
from statistics import mean

for path in sys.argv[1:]:
    print(path)
    rows = defaultdict(lambda: defaultdict(list))
    for r in json.load(open(path))["results"]:
        s = rows[r["step"]]
        if r["kind"] == "setup_wording":
            s["pi_up"].append(r["up_probability"]); s["A_up"].append(r["up_advantage"])
            s["pi_whole"].append(r["whole_probability"]); s["A_whole"].append(r["whole_advantage"])
        else:
            s["pi_correct"].append(r["correct_probability"]); s["A_correct"].append(r["correct_advantage"])
            s["critic_pref"].append(float(r["critic_prefers_correct"]))
            s["nontop_pos"].append(mean(a > 0 for a in r["top_advantages"][1:]))
    keys = ["pi_up", "A_up", "pi_whole", "pi_correct", "A_correct", "critic_pref", "nontop_pos"]
    print("step " + " ".join(k.rjust(10) for k in keys))
    for step in sorted(rows):
        print(str(step).ljust(5) + " ".join(f"{mean(rows[step][k]):10.4f}" for k in keys))
