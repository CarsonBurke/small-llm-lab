import sys as _sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn.functional as F

from ablations import sota_train_gpt as base

base.torch.compile=lambda model,**kwargs:model


class MetricGPT(base.GPT):
	def _forward_hidden(self,input_ids):
		x=self.tok_emb(input_ids);x=F.rms_norm(x,(x.size(-1),))
		if self.embed_proj is not None:x=self.embed_proj(x)
		x0=x;skips=[];enc_iter=self.encoder_indices if self.looping_active else range(self.num_encoder_layers);dec_iter=self.decoder_indices if self.looping_active else range(self.num_encoder_layers,self.num_encoder_layers+self.num_decoder_layers)
		for i in enc_iter:x=self.blocks[i](x,x0);skips.append(x)
		for(skip_idx,i)in enumerate(dec_iter):
			if skip_idx<self.num_skip_weights and skips:
				scaled_skip=self.skip_weights[skip_idx].to(dtype=x.dtype)[None,None,:]*skips.pop()
				if self.skip_gates is not None:g=torch.sigmoid(self.skip_gates[skip_idx].to(dtype=x.dtype))[None,None,:];x=torch.lerp(scaled_skip,x,g)
				else:x=x+scaled_skip
			x=self.blocks[i](x,x0)
		x=self.final_norm(x)
		if self.head_proj is not None:x=self.head_proj(x)
		return x
	def _metric_logits(self,x):
		x=x.to(dtype=torch.bfloat16)
		prototypes=self.tok_emb.weight if self.tie_embeddings else self.lm_head.weight
		prototypes=prototypes.to(dtype=x.dtype)
		logits=F.linear(x,prototypes)
		logits=logits+logits
		logits-=prototypes.square().sum(dim=-1,dtype=x.dtype)[None,None,:]
		logits-=x.square().sum(dim=-1,keepdim=True,dtype=x.dtype)
		return logits
	def forward_logits(self,input_ids):return self._metric_logits(self._forward_hidden(input_ids))
	def forward(self,input_ids,target_ids):
		x=self._forward_hidden(input_ids).reshape(-1,self.tok_emb.weight.size(1));targets=target_ids.reshape(-1);chunk_tokens=4096;loss_sum=x.new_zeros((),dtype=torch.float32)
		for start in range(0,x.size(0),chunk_tokens):
			x_chunk=x[start:start+chunk_tokens];y_chunk=targets[start:start+chunk_tokens];logits=self._metric_logits(x_chunk[:,None,:]).squeeze(1)
			loss_sum+=F.cross_entropy(logits.float(),y_chunk,reduction='sum')
		return loss_sum/targets.numel()


def train_and_eval(h,device):
	base.random.seed(h.seed);base.np.random.seed(h.seed);base.torch.manual_seed(h.seed);base.torch.cuda.manual_seed_all(h.seed);val_data=base.ValidationData(h,device);base.log(f"train_shards: {len(list(base.Path(h.datasets_dir).resolve().glob('fineweb_train_*.bin')))}");base.log(f"val_tokens: {val_data.val_tokens.numel()-1}");base_model,compiled_model=base.train_model(h,device,val_data);base.torch._dynamo.reset();base.timed_eval('pre-quantization post-ema',base.eval_val,h,device,val_data,compiled_model)


base.GPT=MetricGPT
base.train_and_eval=train_and_eval


if __name__=='__main__':base.main()
