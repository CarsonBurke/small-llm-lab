from scripts.build_frozen_critic_pool import eligible_module, synthetic_rows


def test_arithmetic_pool_is_unique_exact_and_deterministic():
    rows = synthetic_rows(4096, 1337)
    assert rows == synthetic_rows(4096, 1337)
    questions = [row['prompt'][0]['content'] for row in rows]
    assert len(set(questions)) == len(rows) == 4096
    assert {row['extra_info']['difficulty_digits'] for row in rows} == {1, 2, 3, 4}
    for row, question in zip(rows, questions, strict=True):
        expression = question.removeprefix('What is ').removesuffix('?')
        parts = expression.split()
        operands = [int(value) for value in parts[::2]]
        assert all(value >= 2 for value in operands)
        if '/' in parts:
            assert operands[0] % operands[1] == 0
            answer = operands[0] // operands[1]
        elif '*' in parts:
            assert operands == sorted(operands)
            answer = operands[0] * operands[1]
        elif '-' in parts:
            assert operands[0] != operands[1]
            answer = operands[0] - operands[1]
        else:
            assert operands == sorted(operands)
            answer = sum(operands)
        assert str(answer) == row['reward_model']['ground_truth']
        assert row['extra_info']['prompt_contract'] == 'bare'


def test_known_guess_modules_are_excluded():
    assert eligible_module('arithmetic__mixed')
    assert eligible_module('algebra__linear_2d')
    assert eligible_module('numbers__round_number')
    for module in ('comparison__pair', 'numbers__gcd', 'numbers__place_value',
                   'numbers__lcm', 'algebra__sequence_next_term'):
        assert not eligible_module(module)
