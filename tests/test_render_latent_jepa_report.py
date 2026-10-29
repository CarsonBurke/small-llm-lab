"""Offline reporting checks: arithmetic, verifier structure and provenance."""
import json
from pathlib import Path

import pytest

from scripts.render_latent_jepa_report import (
    compare_segments, diff_html, numbers_for, parse_answer, read_jsonl,
    reference, render_report,
)


VALUES = [56, 59, 10, 84, 14, 55, 35, 44, 53, 33, 48, 74]
TASK = {'schema': 'latent_jepa_sum_task_v1', 'terms': 12, 'write_every': 4,
        'heldout_modulus': 64, 'vocab': 256}


def test_correct_trajectory_and_segments():
    prompt, trace, segments = reference(VALUES, 4)
    assert prompt.startswith('<bos> 56 + 59')
    assert [s['total'] for s in segments] == [223, 410, 565]
    assert segments[0]['expression'] == '56 + 59 + 10 + 84 + 14'
    assert segments[1]['expression'] == '223 + 55 + 35 + 44 + 53'
    assert parse_answer(trace) == 565
    assert all(not r['notes'] for r in compare_segments(segments, trace))


def test_distinguishes_local_arithmetic_copy_and_carry_errors():
    _, trace, segments = reference(VALUES, 4)
    corrupt = trace.replace('= 223', '= 222').replace('223 + 55', '225 + 55')
    corrupt = corrupt.replace('410 + 33', '410 + 34')
    rows = compare_segments(segments, corrupt)
    assert 'Incorrect local addition' in rows[0]['notes']
    assert 'Subtotal error -1' in rows[0]['notes']
    assert 'Opening value differs from reference' in rows[1]['notes']
    assert 'Does not carry preceding emitted subtotal' in rows[1]['notes']
    assert 'Copied terms differ from prompt' in rows[2]['notes']
    assert rows[2]['generated']['local_sum'] == 566


def test_structural_answer_verifier():
    assert parse_answer('<think> junk </think> <answer> 565 </answer> <eos>') == 565
    assert parse_answer('<answer> 565 </answer> </answer> <eos>') is None
    assert parse_answer('<answer> 565 </answer>') is None
    assert parse_answer('<answer> 56 g5 </answer> <eos>') is None
    assert parse_answer('<answer> 565 </answer> <answer> 565 </answer> <eos>') is None
    assert parse_answer('<answer> 565 </answer> <eos> <answer> 999') == 565


def test_never_infers_inputs_from_generation_and_checks_target():
    row = {'step': 10, 'panel': 'free_mean', 'target': 565, 'text': '56 + 59'}
    assert numbers_for(row, 0, None, TASK)[0] is None
    panel = {'numbers': [VALUES], 'seed': 1, 'device': 'cuda'}
    assert numbers_for(row, 0, panel, TASK)[0] == VALUES
    with pytest.raises(ValueError, match='target mismatch'):
        numbers_for(dict(row, target=566), 0, panel, TASK)
    assert numbers_for(dict(row, panel='unknown'), 0, panel, TASK)[0] is None


def test_noisy_rows_use_first_problem():
    panel = {'numbers': [VALUES], 'seed': 1}
    row = {'step': 10, 'panel': 'free_s0.4', 'target': 565}
    assert numbers_for(row, 3, panel, TASK)[0] == VALUES


def test_diff_escapes_html():
    a, b, first = diff_html('<think> 123 </think>', '<script>alert(1)</script>')
    assert '<script>' not in b
    assert '&lt;script&gt;' in b
    assert 'First differing task token' in first


def make_run(tmp_path):
    run = tmp_path / 'run<script>'
    run.mkdir()
    manifest = {'contract': {'task': TASK, 'arm': 'latent', 'stage': 'pretrain',
                            'config': {'anchor': 'prototype', 'inverse_weight': .5,
                                       'sigreg_weight': .0025, 'span_weight': .5,
                                       'span_sigreg_weight': 2., 'target_detach': False,
                                       'self_feed_passes': 0, 'latent_noise': 0.}}}
    (run / 'manifest.json').write_text(json.dumps(manifest))
    _, trace, _ = reference(VALUES, 4)
    rows = [{'step': 10, 'panel': 'free_mean', 'target': 565, 'reward': 1., 'text': trace},
            {'step': 20, 'panel': 'free_mean', 'target': 565, 'reward': 0.,
             'text': '<script>alert(1)</script>'}]
    (run / 'samples.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (run / 'metrics.jsonl').write_text(json.dumps({'step': 10, 'kind': 'bench',
                                                  'free_mean/accuracy': .25}) + '\n')
    refs = run / 'references.json'
    refs.write_text(json.dumps({'schema': 'latent_jepa_sum_panel_v1', 'task': TASK,
                                'numbers': [VALUES], 'method': 'Test fixture'}))
    return run, refs


def test_report_provenance_selection_and_escaping(tmp_path):
    run, refs = make_run(tmp_path)
    html = render_report(run, refs)
    assert 'Correct trajectory' in html
    assert '223 + 55 + 35 + 44 + 53 = 410' in html
    assert '25.00%' in html
    assert 'Test fixture' in html
    assert 'data-step="20" >' in html
    assert '<script>alert(1)</script>' not in html
    assert '&lt;script&gt;alert(1)&lt;/script&gt;' in html
    assert 'per-transcript rewards' in html
    assert 'smaller loss even when the decoded final answer is wrong' in html
    assert 'Original inputs were not saved' in render_report(run)
    with pytest.raises(ValueError, match='No saved transcripts at step'):
        render_report(run, refs, step=15)


def test_jsonl_active_writer(tmp_path):
    path = tmp_path / 'metrics.jsonl'
    path.write_text('{"step": 1}\n{"step": ')
    assert read_jsonl(path) == [{'step': 1}]
    path.write_text('{"step": 1}\n{"step": \n')
    with pytest.raises(json.JSONDecodeError):
        read_jsonl(path)


def test_rejects_html_in_numeric_transcript_fields(tmp_path):
    run, refs = make_run(tmp_path)
    rows = read_jsonl(run / 'samples.jsonl')
    rows[0]['target'] = '<script>alert(1)</script>'
    (run / 'samples.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    with pytest.raises(ValueError, match='must be integers'):
        render_report(run, refs)
