import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import warnings


class MLP(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.c_fc    = nn.Linear(d_model,d_model*4, bias=True)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(d_model*4,d_model, bias=True)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self,d_model, n_head):
        super().__init__()
        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head = n_head
        self.qkv = nn.Linear(d_model, 3 * d_model,bias=True)
        self.attn_out = nn.Linear(d_model,d_model,bias=True)

    def forward(self,x):
        batch, length, dim = x.size()

        q,k,v = self.qkv(x).split(self.d_model,dim=2)
        k = k.view(batch,length,self.n_head,dim // self.n_head).transpose(1,2) # (batch,n_head,length,head_dim)
        q = q.view(batch,length,self.n_head,dim // self.n_head).transpose(1,2)
        v = v.view(batch,length,self.n_head,dim // self.n_head).transpose(1,2)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
    
        scale_factor = 1 / math.sqrt(q.size(-1)) 
        attn_bias = torch.zeros(length, length, dtype=q.dtype,device=x.device)
        temp_mask = torch.ones(length, length, dtype=torch.bool,device=x.device).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(q.dtype)

        attn_weight = q @ k.transpose(-2, -1) * scale_factor

        attn_weight += attn_bias
        attn_scores = torch.softmax(attn_weight, dim=-1)

        out = attn_scores @ v

        out = out.transpose(1,2).contiguous().view(batch,length,dim)
        out = self.attn_out(out)

        return out.squeeze()

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, d_model, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

             
class Block(nn.Module):
    def __init__(self,d_model,n_head):
        super().__init__()
        self.ln_1 = LayerNorm(d_model,bias=True)
        self.attn = CausalSelfAttention(d_model, n_head)
        self.attn_out_resid_dummy = nn.Identity()

        self.ln_2 = LayerNorm(d_model, bias=True)
        self.mlp = MLP(d_model)

    def forward(self,x):
        x = self.ln_1(x)
        o = self.attn(x)
            
        x = x + o
        x = self.attn_out_resid_dummy(x) #dummy so we can hook
        self.attn_out = x + o
        x = self.ln_2(x) 

        x = x + self.mlp(x)
        return x

class GPT(nn.Module):
    def __init__(self,input_dim,d_model,n_head,context_length,seed=10):
        super().__init__()        
        #set seed
        torch.manual_seed(seed)
        self.context_length = context_length
        self.transformer = nn.ModuleDict(dict(
		    wte = nn.Linear(input_dim,d_model),
		    wpe = nn.Embedding(context_length,d_model),
		    h = Block(d_model,n_head),
		    out = nn.Linear(d_model,input_dim)
		    ))
        
    def forward(self,x):
        device = x.device
        #rather than asserting,just raise a warning
        warnings.warn(f"This model is not designed to handle sequences longer than the context length, current length {x.size(1)}, block size is only {self.context_length}")
        #cut the sequence to the context length
        x = x[:,-self.context_length:]
		# forward the model itself

        pos = torch.arange(0, x.size(1), dtype=torch.long, device=device) # shape (t)
        embed = self.transformer.wte(x) # token embeddings of shape (b, t, n_embd)

        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = embed + pos_emb
			
        x = self.transformer.h(x)

        return self.transformer.out(x),self.transformer.h.attn_out