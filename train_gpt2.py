import torch 
from dataclasses import dataclass
import torch.nn as nn
from torch.nn import functional as F
import math


class CausalSelfAttention(nn.Module):
    def __init__(self, config):  
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT=1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
                 .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self,x):
        B,T,C=x.size()
        qkv=self.c_attn(x)
        q,k,v=qkv.split(self.n_embd,dim=2)
        k=k.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        q=q.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        v=v.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.c_fc=nn.Linear(config.n_embd,4*config.n_embd)
        self.gelu=nn.GELU(approximate='tanh')
        self.c_proj=nn.Linear(4*config.n_embd,config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT=1
        
       
    
    def forward(self,x):
        x=self.c_fc(x)
        x=self.gelu(x)
        x=self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()   
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

    

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "wpe": nn.Embedding(config.block_size, config.n_embd),
            "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            "ln_f": nn.LayerNorm(config.n_embd)
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        #weight sharing
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
    
    def _init_weights(self,module):
        for module in self.modules():
            std=0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std*=(2*self.config.n_layer)**-0.5
                if isinstance(module, nn.Linear):
                    torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.ones_(module.weight)
    
    def forward(self,idx,targets=None):
        B,T = idx.size()
        assert T <= self.config.block_size, "Cannot forward, model block size is exhausted."
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits,loss
    
    #loading pre-trained weights from huggingface copy-pasted code
    @classmethod
    def from_pretrained(cls, model_type):   
        """Loads pretrained GPT-2 model weights from HuggingFace"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

     
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]

        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024

        config = GPTConfig(**config_args)
        model = cls(config)   

    
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd = model.state_dict()

        transposed = [
            'attn.c_attn.weight', 'attn.c_proj.weight',
            'mlp.c_fc.weight', 'mlp.c_proj.weight'
        ]

        sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias')]
        sd_keys_hf = [
            k for k in sd_hf.keys()
            if not k.endswith('.attn.masked_bias') and not k.endswith('.attn.bias')
        ]

        assert len(sd_keys) == len(sd_keys_hf), \
            f"Key mismatch {len(sd_keys)} vs {len(sd_keys_hf)}"

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model



#########################
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps' #for mac
print(f"using device: {device}")


num_return_sequences=1
max_lenght=100

import tiktoken

class DataLoaderLite:
    def __init__(self,B,T):
        self.B=B
        self.T=T
        self.enc=tiktoken.get_encoding("gpt2")
        with open("output.txt", 'r', encoding='utf-8') as f:
            text = f.read()
        tokens=self.enc.encode(text)
        self.tokens=torch.tensor(tokens)
        print(f'loaded {self.tokens} tokens')
        print(f'1 epoch = {len(self.tokens)//(B*T)} batches')

        self.current_position=0
    def next_batch(self):
        B,T=self.B,self.T
        buf=self.tokens[self.current_position:self.current_position+B*T+1]
        x=buf[:-1].view(B,T).to(device)
        y=buf[1:].view(B,T).to(device)
        self.current_position+=B*T
        if self.current_position+B*T+1>=len(self.tokens):
            self.current_position=0
        return x,y

torch.manual_seed(1337)
if device=='cuda':
    torch.cuda.manual_seed(1337)
train_loader = DataLoaderLite(B=4,T=32)
torch.set_float32_matmul_precision('high') #tf32 for cuda  8x speedup

#model=GPT.from_pretrained('gpt2')

model=GPT(GPTConfig())
model.eval()
model.to(device)
#model=torch.compile(model)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

import time 
for i in range(2640):
    t0=time.time()
    x,y=train_loader.next_batch()
    optimizer.zero_grad()
    with torch.autocast(device_type=device, dtype=torch.bfloat16 ):
        logits,loss=model(x,y)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize() if device=='cuda' else None
    t1=time.time()
    dt=(t1-t0)*1000

    print(f"step {i+1}, loss: {loss.item():.4f}, dt: {dt:.2f}ms")
    
    print(loss.item())
#  import sys; sys.exit(0)

print("didnt crash yay!")


torch.manual_seed(42)
torch.cuda.manual_seed(42)
x, y = train_loader.next_batch()
prompt_len = x.size(1) 

torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_lenght:
    with torch.no_grad():
        logits, _ = model(x)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)

        ix = torch.multinomial(topk_probs, 1)
        xcol = torch.gather(topk_indices, 1, ix)
        x = torch.cat((x, xcol), dim=1)

enc = tiktoken.get_encoding("gpt2")

# prompt_len = x.size(1) - (max_lenght - x.size(1))
for i in range(num_return_sequences):
    tokens = x[i, :max_lenght].tolist()
    prompt_tokens = tokens[:prompt_len]
    predicted_tokens = tokens[prompt_len:]
    prompt_text = enc.decode(prompt_tokens)
    predicted_text = enc.decode(predicted_tokens)
    print(f"=== SAMPLE {i+1} ===")
    print("[PROMPT]:")
    print(prompt_text)
    print("[PREDICTED]:")
    print(predicted_text)