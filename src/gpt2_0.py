import torch
from torch import nn
import math
import torch.nn.functional as F
from dataclasses import dataclass
import inspect
import torch.utils
from dataload import Dataloader

class  CausalSelfAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn=nn.Linear(config.n_embd,3*config.n_embd)
        self.c_proj = nn.Linear(config.n_embd,config. n_embd)
        self.c_proj.set_scale=1.0
        self.register_buffer(
            "bias", 
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size), persistent=False)
        # self.dropout=nn.Dropout(config.dropout)
        self.n_head=config.n_head
        self.n_embd=config.n_embd

    def forward(self,x):
        B,T,C=x.shape
        qkv=self.c_attn(x)
        q,k,v=qkv.split(self.n_embd,dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # wei = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))#(B,T,T)
        # wei=wei.masked_fill(self.bias[:,:,:T,:T]==0,float('-inf'))
        # wei=F.softmax(wei,-1)
        # # wei=self.dropout(wei)
        # y=wei @ v
        'a really faster way to implement self attention is flash attention'
        y=F.scaled_dot_product_attention(q,k,v,is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C) 
        self.out=self.c_proj(y)
        return self.out

class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.set_scale=1.0
        # self.dropout = nn.Dropout(config.dropout)
    
    def forward(self,x):
        x=self.c_fc(x)
        x=self.gelu(x)
        x=self.c_proj(x)
        # self.out=self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.ln_1=nn.LayerNorm(config.n_embd)
        self.attn=CausalSelfAttention(config)
        self.ln_2=nn.LayerNorm(config.n_embd)
        self.mlp=MLP(config)
        
        
    
    def forward(self,x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    
@dataclass
class GPTConfig:
    block_size :int=1024
    vocab_size :int=50257
    n_layer :int=12
    n_head :int=12
    n_embd :int=768
    bias :bool=True

class GPT(nn.Module):
    def __init__(self,config) -> None:
        super().__init__()
        self.config=config

        self.transformer=nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size,config.n_embd),
            wpe=nn.Embedding(config.block_size,config.n_embd),
            # drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd)
        ))
        self.lm_head=nn.Linear(config.n_embd,config.vocab_size,bias=False)

        #weights tying concept
        self.transformer.wte.weight=self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std=0.02
            if hasattr(module, 'set_scale'):
                std=(2*self.config.n_layer )**-0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx,target=None):
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) 
        if target is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1), ignore_index=-1)
        else:
            loss=None

        return logits,loss 

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
    def config_optimizer(self,weight_decay,lr,betas,device):
        params={pn:p for pn,p in self.named_parameters()}
        params={pn:p for pn,p in params.items() if p.requires_grad}
        decayed_points=[p for pn,p in params.items() if p.dim()>=2]
        non_decayed_points=[p for pn,p in params.items() if p.dim()<2]
        optimizer_groups=[
            {'params':decayed_points,'weight_decay':weight_decay},
            {'params':non_decayed_points,'weight_decay':0}]
        num_decay_params=sum(p.numel() for p in decayed_points)
        num_nodecay_params=sum(p.numel() for p in non_decayed_points)
        print(f'there are {len(decayed_points)} decayed_params with total num of parameters = {num_decay_params}')
        print(f'there are {len(non_decayed_points)} non_decayed_params with total num of parameters = {num_nodecay_params}')
        # Create AdamW optimizer and use the fused version if it is available
        #if cuda available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()


        optimizer=torch.optim.AdamW(optimizer_groups,lr=lr,betas=betas,eps=1e-8,fused=use_fused)
        return optimizer

#----------------------------------------------------------------------------------------------------------------
import time

device='cpu'

# if torch.cuda.is_available():
#     device='cuda'
# elif hasattr(torch.backends,'mps') and torch.backends.mps.is_available():
#     device='mps'
print(f'using {device}')

model=GPT(GPTConfig(vocab_size=50304))

num_return_sequences = 5
max_length = 30

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
#b=32,t=1024
train_loader=Dataloader(B=4,T=32)

# this operation is req in case of GPU but i am using CPU so it is not necessary but if using mps
#then set prec to medium
# torch.set_float32_matmul_precision('high')

model=GPT(GPTConfig())
model.eval()
# model.to(device)
model=torch.compile(model)

max_lr=6e-4
min_lr=max_lr*0.1
warmup_steps=10
max_steps=50
'''
This lr scheduler is implemntation of cosine scheduling
and learning rate is given as 
η(t) = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(t * π / T))

here T is max_steps-warmup_steps
t is it-warmup_steps
'''
def get_lr(it):
    if it<warmup_steps :
        return max_lr*(it+1)/warmup_steps
    if it>max_steps :
        return min_lr
    
    decay_ratio=(it-warmup_steps)/(max_steps-warmup_steps)
    assert 0<decay_ratio<1
    coeff=0.5*(1.0+math.cos(math.pi*decay_ratio))
    return min_lr+coeff*(max_lr-min_lr)

# optimizer=torch.optim.AdamW(params=model.parameters(),lr=3e-4,betas=(0.9,0.995),eps=1e-8)
optimizer=model.config_optimizer(weight_decay=0.01,lr=max_lr,betas=(0.9,0.995),device=device)
t0=time.time()
from tqdm import tqdm
for step in tqdm(range(max_steps)):
    start_time=time.time()
    x,y=train_loader.next_batch()
    # if gpu is used
    # with torch.autocast(device_type=device, dtype=torch.float16):
    #     logits,loss=model(x,y)
    #     import code;code.interact(local=locals())
    logits,loss=model(x,y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm=torch.nn.utils.clip_grad_norm(model.parameters(),1.0)
    #lr scheduler
    lr=get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr']=lr

    optimizer.step()
    end_time = time.time()
    time_taken = end_time - start_time
    tokens_per_sec=(train_loader.B*train_loader.T)/(time_taken)
    tqdm.write(f' loss {loss:.2f} time taken {time_taken:.2f} norm {norm:.2f} sec tokens per sec {tokens_per_sec}')


print('total time taken ',time.time() - t0)
import sys
sys.exit(0)

torch.manual_seed(42)
# torch.cuda.manual_seed(42)

while x.size(1) < 200:
    with torch.no_grad():
        logits,loss=model(x,y)
        logits=logits[:, -1, :]
        probs=F.softmax(logits,dim=-1)
        topk_probs,topk_indices=torch.topk(probs,10,dim=-1)
        next_token=torch.multinomial(topk_probs, num_samples=1)
        xcol=torch.gather(topk_indices,-1,next_token)
        x=torch.cat((x, xcol), dim=1)
print(x.shape)
print(enc.decode(x[0,:32].tolist()))
for i in range(B):
    tokens=x[i,:]
    text=enc.decode(tokens.tolist())
    print('->',text)
    print('################################')
print('total time:',time.time()-start_time)

'''
mps is faster
time mps=9.10 cpu=14.93
mps with high precision=9.423
mps with medium precision=8.187

'''