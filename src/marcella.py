import torch
from torch.nn import MultiheadAttention, LayerNorm, GELU, Module, Linear, Dropout
from torch.optim import AdamW
from torch.nn.functional import scaled_dot_product_attention

N_HEADS = 12
D_MODEL = 384
VOCAB = 50
ATTN_DROPOUT = 0.1

class MultiHeadFlashAttention(Module):
    def __init__(self, dropout, attn_dropout=ATTN_DROPOUT):
        super().__init__()
        
        if D_MODEL%N_HEADS !=0:
            raise ValueError(f'D_MODEL: {D_MODEL} must be divisible by N_HEADS: {N_HEADS}')
        
        self.k = Linear(D_MODEL, D_MODEL, bias=False)
        self.v = Linear(D_MODEL, D_MODEL, bias=False)
        self.q = Linear(D_MODEL, D_MODEL, bias=False)
        self.attn_dropout = attn_dropout
        self.linear = Linear(D_MODEL, D_MODEL, bias=False)
        self.dropout = Dropout(dropout)

    def forward(self, x):
        B, S, _ = x.shape
        k = self.k(x).view(B, S, N_HEADS, D_MODEL//N_HEADS)
        v = self.v(x).view(B, S, N_HEADS, D_MODEL//N_HEADS)
        q = self.q(x).view(B, S, N_HEADS, D_MODEL//N_HEADS)
        
        out = scaled_dot_product_attention(query=q,
                                           key=k,
                                           value=v,
                                           dropout_p=self.attn_dropout if self.training else 0.0,
                                           is_causal=True)
        
        out = self.linear(out.view(B, S, D_MODEL))
        return self.dropout(out)