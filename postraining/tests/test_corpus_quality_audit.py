import ast
from types import SimpleNamespace

from scripts.audit_rl_corpus_quality import literal_truth


def test_vacuous_assertions_are_proved_without_executing_calls():
    assert literal_truth(ast.parse("f(x) == bad if False else True", mode="eval").body) is True
    assert literal_truth(ast.parse("f(x) or True", mode="eval").body) is True
    assert literal_truth(ast.parse("f(x) == 42", mode="eval").body) is None
    assert literal_truth(ast.parse("f(x)", mode="eval").body) is None
    assert literal_truth(ast.parse("False", mode="eval").body) is False


def test_choice_budget_includes_bos(monkeypatch):
    from postraining import choice_rl_pool

    monkeypatch.setattr(choice_rl_pool, "strip_math_prompt_framing", lambda text: (text, None))
    monkeypatch.setattr(choice_rl_pool, "contaminated", lambda *args: False)
    monkeypatch.setattr(choice_rl_pool, "contains_benchmark_question", lambda *args: False)
    tokenizer = SimpleNamespace(encode=lambda text: list(range(len(text))), bos_id=lambda: 50256)
    screens = SimpleNamespace(tokenizer=tokenizer, exact=set(), ngrams=set(), benchmarks=None,
                              owners=None, max_prompt_tokens=256)
    assert choice_rl_pool.screen_problem("x" * 255, screens) == ("x" * 255, 256)
    assert choice_rl_pool.screen_problem("x" * 256, screens) == "over_prompt_budget"
