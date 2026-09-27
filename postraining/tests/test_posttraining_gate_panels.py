import pytest

from scripts.prepare_posttraining_gates import select_panel


def row(index, module):
    return {"prompt": [{"role": "user", "content": str(index)}],
            "extra_info": {"module": module}}


def test_science_panel_balances_datasets_and_ignores_input_order():
    rows = [row(f"{source}-{i}", f"{source}_choice_{3 + i % 2}")
            for source in ("sciq", "arc_easy", "arc_challenge", "openbookqa")
            for i in range(20)]
    panel = select_panel(rows, "science_mc", 32)
    assert panel == select_panel(list(reversed(rows)), "science_mc", 32)
    for start in range(0, 32, 4):
        assert len({r["extra_info"]["module"].rsplit("_choice_", 1)[0]
                    for r in panel[start:start + 4]}) == 4
    assert len({r["prompt"][0]["content"] for r in panel}) == 32


def test_small_stratum_exhausts_without_repeating_prompts():
    rows = [row(0, "numeric")] + [row(i, "choice_4") for i in range(1, 10)]
    panel = select_panel(rows, "ultradata_knowledge", 8)
    assert len(panel) == 8
    assert sum(r["extra_info"]["module"] == "numeric" for r in panel) == 1
    with pytest.raises(ValueError):
        select_panel(rows, "ultradata_knowledge", 11)
