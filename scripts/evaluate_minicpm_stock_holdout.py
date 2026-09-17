#!/usr/bin/env python3
"""Queue-only stock Transformers evaluation on a saved controller prompt holdout."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from postraining.hf_runtime import prepare_text_only_transformers_runtime
from postraining.core import answer_style, load_unique_math_rows, verify_answer


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def save(path, result):
    temporary=path.with_suffix('.partial')
    temporary.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest=json.loads(args.manifest.read_text())
    data=Path(manifest['args']['data'])
    if hashlib.sha256(data.read_bytes()).hexdigest()!=manifest['data_sha256']:
        raise ValueError('Dataset differs from controller holdout')
    hashes=manifest['evaluation_prompt_sha256']
    wanted=set(hashes); by_hash={}
    for row in load_unique_math_rows(str(data)):
        key=digest(row['prompt'])
        if key in wanted:
            if key in by_hash and by_hash[key]['reward_model']!=row['reward_model']:
                raise ValueError('Conflicting heldout verifier labels')
            by_hash.setdefault(key,row)
    rows=[by_hash[key] for key in hashes]
    if len(rows)!=8 or len(wanted)!=8:
        raise ValueError('Expected exact eight-prompt holdout')
    rows=rows[:4]
    hashes=hashes[:4]
    samples=1
    group_prompts=4
    budget=128
    result={'schema':'minicpm-stock-holdout/v1','status':'initializing','manifest':str(args.manifest),
            'evaluation_prompt_sha256':hashes,'source':Path(__file__).read_text(),
            'scope':'Bounded inference diagnostic: four prompts, one continuation each, at most128 new tokens. Stock compiled BF16 SDPA and official sampling. Not a reasoning-quality or completion-rate evaluation.',
            'model_id':'openbmb/MiniCPM5-1B','revision':'87179e5c1f455ef22e6223592d2d61351b525bfc',
            'samples_per_prompt':samples,'max_new_tokens':budget,'arms':{}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    save(args.output,result)
    started=time.monotonic()
    try:
        import faulthandler
        faulthandler.dump_traceback_later(120, repeat=True)
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required; submit through mlq')
        prepare_text_only_transformers_runtime()
        from transformers import AutoModelForCausalLM, AutoTokenizer, CompileConfig
        print('Loading cached tokenizer and BF16 model',flush=True)
        torch.manual_seed(manifest['args']['seed'])
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32=False
        tokenizer=AutoTokenizer.from_pretrained(result['model_id'],revision=result['revision'],local_files_only=True)
        tokenizer.padding_side='left'
        model=AutoModelForCausalLM.from_pretrained(result['model_id'],revision=result['revision'],dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).to('cuda').eval()
        print('Model loaded; resolving official sampling',flush=True)
        effective,_=model._prepare_generation_config(None)
        result['effective_sampling']={k:getattr(effective,k) for k in ('temperature','top_p','top_k','repetition_penalty','eos_token_id','do_sample')}
        if (effective.temperature,effective.top_p,effective.top_k)!=(.9,.95,50):
            raise ValueError('Official default sampling changed')
        eos=set(effective.eos_token_id)
        close=tokenizer.convert_tokens_to_ids('</think>')
        encoded=[tokenizer.apply_chat_template(row['prompt'],tokenize=True,add_generation_prompt=True,enable_thinking=True,return_dict=True)['input_ids'] for row in rows]
        if max(map(len,encoded))>1024:
            raise ValueError('Unexpected prompt truncation required')
        result['encoded_prompt_sha256']=[digest(ids) for ids in encoded]
        compile_config=CompileConfig(fullgraph=True,mode=None,options={'triton.cudagraphs':True,'emulate_precision_casts':True})


        with torch.inference_mode():
            for name,reserve in [('stock_short_diagnostic',0)]:
                arm={'responses':[],'groups':[],'answer_reserve_tokens':reserve}
                result['arms'][name]=arm
                for offset in range(0,len(rows),group_prompts):
                    selected=encoded[offset:offset+group_prompts]
                    width=max(map(len,selected))
                    ids=torch.full((len(selected),width),effective.pad_token_id,dtype=torch.long,device='cuda')
                    mask=torch.zeros_like(ids)
                    for i,seq in enumerate(selected):
                        ids[i,-len(seq):]=torch.tensor(seq,device='cuda');mask[i,-len(seq):]=1
                    torch.manual_seed(manifest['args']['seed']+100000+offset)
                    torch.cuda.synchronize();group_start=time.monotonic()
                    print('Compiling and generating four bounded continuations',flush=True)
                    generated=model.generate(input_ids=ids,attention_mask=mask,num_return_sequences=samples,
                        max_new_tokens=budget,use_cache=True,cache_implementation='static',compile_config=compile_config,
                        disable_compile=False)
                    torch.cuda.synchronize()
                    if not hasattr(model,'_compiled_call'):
                        raise RuntimeError('Stock decoding did not compile; no silent eager fallback')
                    elapsed=time.monotonic()-group_start
                    continuations=generated[:,width:].cpu()
                    for flat,continuation in enumerate(continuations):
                        local,sample=divmod(flat,samples);pidx=offset+local
                        token_list=continuation.tolist()
                        ending=next((i+1 for i,t in enumerate(token_list) if t in eos),len(token_list))
                        token_list=token_list[:ending]
                        closed=next((i for i,t in enumerate(token_list) if t==close),None)
                        text=tokenizer.decode(token_list,skip_special_tokens=True)
                        correct,_=verify_answer(text,rows[pidx]['reward_model']['ground_truth'],answer_style(rows[pidx]))
                        arm['responses'].append({'prompt_index':pidx,'prompt_sha256':hashes[pidx],'sample_index':sample,
                            'correct':bool(correct),'text':text,'stream_positions':len(token_list),
                            'thoughts':closed if closed is not None else len(token_list),
                            'forced_close':bool(reserve and closed==budget-reserve-1),
                            'answer_termination':'eos' if token_list[-1] in eos else 'token_budget',
                            'final_token_id':token_list[-1]})
                    arm['groups'].append({'offset':offset,'seconds':elapsed,'trajectories':len(continuations)})
                    arm['generation_completed']=True
                    result['status']=name
                    save(args.output,result)
                    print(json.dumps({'arm':name,'offset':offset,'seconds':elapsed}),flush=True)
                    del generated,continuations
        result['status']='completed'
    except BaseException as exc:
        result['status']='failed';result['error']=repr(exc)
        raise
    finally:
        result['wall_seconds']=time.monotonic()-started
        faulthandler.cancel_dump_traceback_later()
        save(args.output,result)


if __name__=='__main__':
    main()
