#!/usr/bin/env python3
"""Render saved latent-JEPA sum transcripts as an offline HTML report.

Example:
    python3 scripts/render_latent_jepa_report.py --run-dir postraining/runs/NAME \
        --references postraining/runs/NAME/reference_panel.json

This script reads logs only. It neither loads a model nor uses a GPU. Reference
inputs must be saved explicitly or supplied in a panel export; generated text is
never treated as evidence of the original problem.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from difflib import SequenceMatcher
from html import escape
import json
from pathlib import Path
import re


TOKEN = re.compile(r'<[^>]*>|g\d+|\d|[^\s]')


def read_jsonl(path: Path) -> list[dict]:
    """Ignore only an unfinished final record while a run is writing its log."""
    if not path.exists():
        return []
    text = path.read_text()
    lines = text.splitlines()
    rows = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines) - 1 and not text.endswith('\n'):
                continue
            raise
    return rows


def reference(values: list[int], stride: int) -> tuple[str, str, list[dict]]:
    """Mirror SumTask.completion: first write follows stride additions."""
    total = values[0]
    expression = [str(total)]
    segments = []
    start = 1
    for i, value in enumerate(values[1:], 1):
        expression.append(str(value))
        total += value
        if i % stride == 0 or i == len(values) - 1:
            segments.append({'expression': ' + '.join(expression), 'total': total,
                             'new_terms': values[start:i + 1], 'start': int(expression[0])})
            expression = [str(total)]
            start = i + 1
    prompt = '<bos> ' + ' + '.join(map(str, values)) + ' ='
    trace = '<think> ' + ' '.join(f"{s['expression']} = {s['total']} ;" for s in segments)
    trace += f' </think> <answer> {total} </answer> <eos>'
    return prompt, trace, segments


def parse_answer(text: str) -> int | None:
    """Match the saved task's structural verifier on its rendered tokens."""
    if '<eos>' not in text:
        return None
    before = text.split('<eos>', 1)[0].strip()
    if before.count('<answer>') != 1 or not before.endswith('</answer>'):
        return None
    span = before.split('<answer>', 1)[1][:-len('</answer>')].strip()
    if not re.fullmatch(r'\d+', span, re.ASCII):
        return None
    return int(span)


def generated_segments(text: str) -> list[dict]:
    think = text.split('</think>', 1)[0]
    if think.startswith('<think>'):
        think = think[len('<think>'):]
    rows = []
    for chunk in think.split(';'):
        chunk = chunk.strip()
        if not chunk:
            continue
        match = re.fullmatch(r'(\d+(?:\s*\+\s*\d+)*)\s*=\s*(\d+)', chunk, re.ASCII)
        if match:
            values = [int(v.strip()) for v in match[1].split('+')]
            rows.append({'raw': chunk, 'values': values, 'total': int(match[2]),
                         'local_sum': sum(values)})
        else:
            rows.append({'raw': chunk, 'values': None, 'total': None, 'local_sum': None})
    return rows


def compare_segments(expected: list[dict], text: str) -> list[dict]:
    actual = generated_segments(text)
    rows = []
    for i in range(max(len(expected), len(actual))):
        ref = expected[i] if i < len(expected) else None
        gen = actual[i] if i < len(actual) else None
        notes = []
        if ref is None:
            notes.append('Extra segment')
        elif gen is None:
            notes.append('Missing segment')
        elif gen['values'] is None:
            notes.append('Malformed expression')
        else:
            if gen['total'] != gen['local_sum']:
                notes.append('Incorrect local addition')
            if gen['values'][0] != ref['start']:
                notes.append('Opening value differs from reference')
            if gen['values'][1:] != ref['new_terms']:
                notes.append('Copied terms differ from prompt')
            if gen['total'] != ref['total']:
                notes.append(f"Subtotal error {gen['total'] - ref['total']:+d}")
            if i and i <= len(actual) - 1 and actual[i - 1]['total'] is not None:
                if gen['values'][0] != actual[i - 1]['total']:
                    notes.append('Does not carry preceding emitted subtotal')
        rows.append({'reference': ref, 'generated': gen, 'notes': notes})
    return rows


def diff_html(expected: str, actual: str) -> tuple[str, str, str]:
    a, b = TOKEN.findall(expected), TOKEN.findall(actual)
    left, right = [], []
    first = 'Exact token match'
    found = False
    for op, i, j, k, l in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op != 'equal' and not found:
            first = f"First differing task token: {min(i + 1, len(a) + 1)} (reference alignment)"
            found = True
        left.extend(f'<span class="{ "same" if op == "equal" else "expected" }">{escape(t)}</span>' for t in a[i:j])
        right.extend(f'<span class="{ "same" if op == "equal" else "wrong" }">{escape(t)}</span>' for t in b[k:l])
    # Tokens are space separated deliberately: each digit is one model token.
    return ' '.join(left), ' '.join(right), first


def numbers_for(row: dict, index: int, panel: dict | None, task: dict) -> tuple[list[int] | None, str]:
    values = row.get('numbers')
    provenance = 'Inputs stored with transcript'
    if values is None and panel is not None:
        if row['panel'] not in ('free_mean', 'greedy') and not row['panel'].startswith('free_s'):
            return None, 'Unknown panel layout; no reference assumed'
        which = row.get('problem_index', index if row['panel'] in ('free_mean', 'greedy') else 0)
        try:
            values = panel['numbers'][which]
        except (IndexError, KeyError, TypeError):
            raise ValueError(f'Reference panel has no problem {which}') from None
        provenance = f"Fixed panel, problem {which + 1}; seed {panel.get('seed', '?')}, {panel.get('device', '?')} RNG"
        if panel.get('method'):
            provenance += f" · {panel['method']}"
    if values is None:
        return None, 'Original inputs were not saved. Supply --references to show a verified trajectory.'
    if not isinstance(values, list) or len(values) != task['terms'] or any(type(v) is not int or not 10 <= v <= 99 for v in values):
        raise ValueError('Reference inputs must be a list of the configured number of two-digit integers')
    if sum(values) != row['target']:
        raise ValueError(f"Reference target mismatch at step {row['step']} panel {row['panel']}: {sum(values)} != {row['target']}")
    return values, provenance


def format_number(value: object) -> str:
    if isinstance(value, (float, int)):
        return f'{value:.6g}'
    return str(value)


def transcript_card(row: dict, index: int, panel: dict | None, task: dict) -> str:
    values, provenance = numbers_for(row, index, panel, task)
    actual = row['text']
    parsed = parse_answer(actual)
    status = 'Correct answer' if parsed == row['target'] else 'Malformed answer' if parsed is None else 'Wrong answer'
    tone = 'good' if status == 'Correct answer' else 'bad'
    answer = 'unparseable' if parsed is None else str(parsed)
    heading = f"{row['panel']} · sample {index + 1}"
    body = f'<article class="sample" data-panel="{escape(row["panel"], quote=True)}"><div class="sample-head"><h3>{escape(heading)}</h3><span class="badge {tone}">{status}</span></div>'
    body += f'<div class="answerline">Target <b>{row["target"]}</b><span>Parsed answer <b>{answer}</b></span><span>Saved reward <b>{row["reward"]:g}</b></span></div>'
    body += f'<p class="muted">{escape(provenance)}</p>'
    if values is not None:
        prompt, correct, segments = reference(values, task['write_every'])
        expected_html, generated_html, first = diff_html(correct, actual)
        body += f'<div class="prompt"><small>Original problem</small><code>{escape(prompt)}</code></div>'
        body += '<div class="trajectory"><div><h4>Correct trajectory</h4>'
        body += f'<pre>{escape(correct)}</pre></div><div><h4>Generated trajectory</h4><pre>{escape(actual)}</pre></div></div>'
        body += '<div class="table-wrap"><table><thead><tr><th>Segment</th><th>Correct expression</th><th>Generated expression</th><th>Sum of emitted operands</th><th>Diagnosis</th></tr></thead><tbody>'
        for i, comparison in enumerate(compare_segments(segments, actual), 1):
            ref, gen, notes = comparison['reference'], comparison['generated'], comparison['notes']
            reftext = f"{ref['expression']} = {ref['total']}" if ref else '—'
            gentext = gen['raw'] if gen else '—'
            local = str(gen['local_sum']) if gen and gen['local_sum'] is not None else '—'
            body += f'<tr><td>{i}</td><td><code>{escape(reftext)}</code></td><td><code>{escape(gentext)}</code></td><td>{local}</td><td class="{"bad" if notes else "good"}">{escape("; ".join(notes) or "Exact segment")}</td></tr>'
        body += '</tbody></table></div>'
        body += f'<details><summary>Task-token alignment · {escape(first)}</summary><p class="muted">Each digit is a separate task token. Green marks differing reference tokens; red marks generated replacements or insertions.</p><div class="trajectory"><div><h4>Reference</h4><pre>{expected_html}</pre></div><div><h4>Generated</h4><pre>{generated_html}</pre></div></div></details>'
    else:
        body += f'<h4>Generated trajectory</h4><pre>{escape(actual)}</pre><p class="notice">Correct final answer: {row["target"]}. Intermediate reference sums cannot be recovered safely from this generated text.</p>'
    if parsed is None:
        body += '<p class="notice">The saved verifier requires one answer span containing only digits, closed immediately before the first EOS. This completion fails that structure.</p>'
    return body + '</article>'


def metric_cards(metrics: list[dict], step: int) -> str:
    bench = next((r for r in reversed(metrics) if r.get('step') == step and r.get('kind') == 'bench'), {})
    train = next((r for r in reversed(metrics) if r.get('step', step + 1) <= step and r.get('kind') == 'train'), {})
    keys = [('free_mean/accuracy', 'Full-panel exact answer'), ('free_mean/valid', 'Full-panel valid answers'),
            ('free_mean/partial_0', 'First written subtotal'), ('free_mean/partial_1', 'Second written subtotal'),
            ('free_mean/partial_2', 'Third written subtotal')]
    body = '<div class="metrics">'
    for key, label in keys:
        if key in bench:
            body += f'<div class="metric"><small>{label}</small><strong>{bench[key]:.2%}</strong><span>bench step {step:,}</span></div>'
    for key, label in [('pred_tf', 'Teacher-forced latent MSE'), ('span_pred', 'Span prediction MSE')]:
        if key in train:
            body += f'<div class="metric"><small>{label}</small><strong>{train[key]:.5f}</strong><span>train step {train["step"]:,} · batch average</span></div>'
    return body + '</div><p class="muted">Bench percentages describe the evaluation panel, not just the displayed examples. Training losses are batch averages and are not per-transcript rewards or latent distances.</p>'


def objective_html(contract: dict) -> str:
    cfg = contract.get('config', {})
    if contract.get('arm') != 'latent' or contract.get('stage') != 'pretrain':
        return '<p>The manifest does not identify a latent pretraining run. Inspect its saved trainer before interpreting the optimization signal.</p>'
    formula = f"latent MSE + {cfg.get('inverse_weight', '?')} × anchor CE + {cfg.get('sigreg_weight', '?')} × latent SIGReg"
    if cfg.get('span_weight', 0):
        formula += f" + {cfg['span_weight']} × span MSE + {cfg.get('span_sigreg_weight', '?')} × span SIGReg"
    anchor = 'learned token prototypes' if cfg.get('anchor') == 'prototype' else 'the inverse-token head'
    attached = 'The target encoder also receives gradients (targets are attached).' if cfg.get('target_detach') is False else 'Target latents are detached.'
    exposure = f"Self-feed passes: {cfg.get('self_feed_passes', '?')}; training latent noise: {cfg.get('latent_noise', '?')}."
    return f'''<p><b>Yes: closeness supplies a dense learning signal.</b> The predictor minimizes squared distance between its next latent and the encoder's next-token latent on training text. A smaller error gives a smaller loss even when the decoded final answer is wrong. This is supervised pretraining, rather than a positive RL reward assigned to a rollout.</p>
    <div class="formula">{escape(formula)}</div>
    <p>The anchor uses {anchor}; SIGReg shapes the latent distribution. Span prediction, when enabled, regresses an encoding of intervening text from the span endpoints. {attached} {escape(exposure)}</p>
    <p>The logged transcript <code>reward</code> is a separate binary exact-answer evaluation score. It does not grade latent proximity. Readout and next-token monitor heads train on detached model outputs; their CE does not update the latent model. The saved transcripts contain no per-rollout latent distances, so this report cannot assign latent credit to individual arithmetic steps.</p>
    <p class="muted">A low teacher-forced latent MSE does not by itself establish numerical closeness: the geometry is learned, and free-running inputs differ from reference histories. Source: run manifest and saved source/model.py, pretrain.py, task.py.</p>'''


CSS = '''
:root{color-scheme:dark;--ink:#e3edf3;--muted:#a4b5c2;--line:#304451;--green:#72dfbd;--red:#ff9ca8}*{box-sizing:border-box}body{margin:0;background:#101920;color:var(--ink);font:15px/1.6 system-ui,sans-serif}main{max-width:1440px;margin:auto;padding:40px 30px 80px}header{padding:28px 0}h1{font-size:clamp(28px,4vw,46px);line-height:1.15;margin:8px 0 18px;letter-spacing:-1.5px}h2{font-size:23px;margin:0 0 15px}h3{font-size:18px;margin:0}h4{margin:0 0 10px;font-size:14px}p{margin:12px 0}.eyebrow{color:var(--green);letter-spacing:2px;font-size:12px;font-weight:800;text-transform:uppercase}.muted,small{color:var(--muted)}header .muted{overflow-wrap:anywhere}.toolbar{position:sticky;top:0;z-index:2;display:flex;gap:20px;align-items:center;flex-wrap:wrap;padding:14px 20px;background:#182630f2;border:1px solid var(--line);border-radius:14px;backdrop-filter:blur(12px)}select{margin-left:10px;padding:8px;border:1px solid var(--line);border-radius:6px;font:inherit;background:#213440;color:var(--ink)}.section,.sample{background:#192731;border:1px solid var(--line);padding:26px;border-radius:17px;margin-top:22px}.section{background:#15252d}.sample-head{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap}.badge{padding:3px 11px;border-radius:20px;font-size:12px;font-weight:700;background:#462a34}.badge.good{background:#173e35}.bad{color:var(--red)}.good{color:var(--green)}.answerline{display:flex;gap:22px;flex-wrap:wrap;margin-top:14px}.answerline b{font-size:20px;margin-left:5px}.prompt{padding:14px 18px;border-left:3px solid var(--green);background:#122e30;margin:18px 0}.prompt code{display:block}.trajectory{display:grid;grid-template-columns:1fr 1fr;gap:18px}.trajectory>div{min-width:0}pre,code{font-family:ui-monospace,SFMono-Regular,Consolas,monospace}pre{font-size:13px;line-height:1.9;white-space:pre-wrap;overflow-wrap:anywhere;padding:16px;background:#111e27;border:1px solid #304451;border-radius:10px;margin:0}.table-wrap{overflow-x:auto;margin:18px 0}table{width:100%;border-collapse:collapse;font-size:13px}th{text-align:left;font-size:11px;text-transform:uppercase;color:var(--muted);letter-spacing:.5px}td,th{padding:12px 10px;border-bottom:1px solid var(--line);vertical-align:top}td code{white-space:normal}td:first-child{width:65px}td:last-child{min-width:190px}details{padding-top:10px}summary{cursor:pointer;color:var(--green);font-weight:600}.expected{background:#164b3d;color:#a4f4d6;padding:2px}.wrong{background:#5c2837;color:#ffb4bf;padding:2px}.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(165px,1fr));gap:12px;margin-top:20px}.metric{border:1px solid var(--line);background:#192731;border-radius:12px;padding:15px}.metric small,.metric span{display:block;font-size:12px}.metric strong{display:block;font-size:26px;margin:4px 0}.metric span{color:var(--muted)}.formula{font:14px/1.8 ui-monospace,monospace;background:#183730;padding:14px;border-radius:9px;overflow-wrap:anywhere}.notice{padding:10px 14px;background:#3a3020;border-radius:8px}.checkpoint[hidden],.sample[hidden]{display:none}footer{margin-top:25px;color:var(--muted);font-size:12px}@media(max-width:800px){main{padding:20px 12px}.section,.sample{padding:18px}.trajectory{grid-template-columns:1fr}h1{letter-spacing:-.6px}.toolbar{gap:8px;position:relative}table{min-width:750px}}@media print{.toolbar{position:static}details{display:none}.sample{break-inside:avoid}.section,.sample{box-shadow:none}}
'''

JS = '''
const stepSelect=document.getElementById('step');
const panelSelect=document.getElementById('panel');
function update(){document.querySelectorAll('.checkpoint').forEach(el=>{el.hidden=el.dataset.step!==stepSelect.value;});document.querySelectorAll('.sample').forEach(el=>{el.hidden=panelSelect.value!=='all'&&el.dataset.panel!==panelSelect.value;});}
stepSelect.addEventListener('change',update);panelSelect.addEventListener('change',update);update();
'''


def render_report(run_dir: Path, references: Path | None = None, step: int | None = None) -> str:
    manifest = json.loads((run_dir / 'manifest.json').read_text())
    contract = manifest.get('contract', {})
    task = contract.get('task', {})
    if task.get('schema') != 'latent_jepa_sum_task_v1':
        raise ValueError('Report supports latent_jepa_sum_task_v1 only')
    rows = read_jsonl(run_dir / 'samples.jsonl')
    if not rows:
        raise ValueError('No saved transcripts in samples.jsonl')
    for row in rows:
        if type(row.get('step')) is not int or type(row.get('target')) is not int:
            raise ValueError('Transcript step and target must be integers')
        if not isinstance(row.get('panel'), str) or not isinstance(row.get('text'), str):
            raise ValueError('Transcript panel and text must be strings')
        if type(row.get('reward')) not in (int, float):
            raise ValueError('Transcript reward must be numeric')
    metrics = read_jsonl(run_dir / 'metrics.jsonl')
    panel = json.loads(references.read_text()) if references else None
    if panel:
        if panel.get('schema') != 'latent_jepa_sum_panel_v1':
            raise ValueError('Unsupported reference panel schema')
        for key in ('terms', 'write_every', 'heldout_modulus', 'vocab'):
            if key in panel.get('task', {}) and panel['task'][key] != task[key]:
                raise ValueError(f'Reference task mismatch: {key}')
    steps = sorted({r['step'] for r in rows})
    selected = steps[-1] if step is None else step
    if selected not in steps:
        raise ValueError(f'No saved transcripts at step {selected}; available: {steps}')
    panels = sorted({r['panel'] for r in rows})
    title = f'Latent JEPA · {run_dir.name}'
    body = f'<header><div class="eyebrow">Saved rollout inspection</div><h1>Where the trajectory diverges</h1><p class="muted">{escape(run_dir.name)}</p><p>Correct arithmetic beside the model’s decoded latent rollout. Inspect local addition, copied operands, carried subtotals, and final answer separately.</p></header>'
    body += '<div class="toolbar"><label>Checkpoint<select id="step">'
    body += ''.join(f'<option value="{s}" {"selected" if s == selected else ""}>Step {s:,}</option>' for s in steps)
    body += '</select></label><label>Panel<select id="panel"><option value="all">All panels</option>'
    body += ''.join(f'<option value="{escape(p, quote=True)}">{escape(p)}</option>' for p in panels)
    body += '</select></label><span class="muted">Offline report · select any saved checkpoint</span></div>'
    body += '<section class="section"><h2>What learning signal does the latent get?</h2>' + objective_html(contract) + '</section>'
    for s in steps:
        selected_rows = [r for r in rows if r['step'] == s]
        correct = sum(parse_answer(r['text']) == r['target'] for r in selected_rows)
        body += f'<section class="checkpoint" data-step="{s}" {"hidden" if s != selected else ""}><div class="section"><h2>Step {s:,}</h2><p>Displayed saved examples: <b>{correct} / {len(selected_rows)}</b> exact answers. These are fixed illustrative samples, not a random estimate of accuracy.</p>'
        body += metric_cards(metrics, s) + '</div>'
        counter: dict[str, int] = defaultdict(int)
        for row in selected_rows:
            index = counter[row['panel']]
            counter[row['panel']] += 1
            body += transcript_card(row, index, panel, task)
        body += '</section>'
    body += f'<footer>Inputs: {escape(str(run_dir / "samples.jsonl"))}, metrics.jsonl, manifest.json'
    if references:
        body += f'; references: {escape(str(references))}'
    body += '. No model execution. The report is a snapshot of the logs when rendered.</footer>'
    return f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{escape(title)}</title><style>{CSS}</style></head><body><main>{body}</main><script>{JS}</script></body></html>'


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', required=True, type=Path)
    ap.add_argument('--references', type=Path, help='Fixed input panel exported from this run’s task/benchmark RNG')
    ap.add_argument('--step', type=int, help='Initially selected checkpoint (default: latest saved samples)')
    ap.add_argument('--output', type=Path, help='Default: RUN_DIR/transcript_report.html')
    args = ap.parse_args()
    output = args.output or args.run_dir / 'transcript_report.html'
    refs = args.references
    if refs is None:
        for name in ('references.json', 'reference_panel.json'):
            if (args.run_dir / name).exists():
                refs = args.run_dir / name
                break
    try:
        html = render_report(args.run_dir, refs, args.step)
    except (ValueError, KeyError) as exc:
        ap.error(str(exc))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html)
    print(output.resolve())


if __name__ == '__main__':
    main()
