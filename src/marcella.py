import torch
from torch.nn import LayerNorm, GELU, Module, Linear, Dropout
from torch.optim import AdamW
from src.attention import Attention

N_HEADS = 12
D_MODEL = 384
VOCAB = 32000
ATTN_DROPOUT = 0.1

class TransformerBlock(Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.norm1 = LayerNorm(D_MODEL)
        self.attn = Attention(dropout)
        self.norm2 = LayerNorm(D_MODEL)
        self.ffn = FeedForwardNetwork(dropout)
        
    def forward(self, x, kv_cache=None):
        attn_out = self.attn(self.norm1(x), kv_cache)
        x = x + attn_out
        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out
        return x
    
class FeedForwardNetwork(Module):
    def __init__(self, dropout = 0.1):
        super().__init__()
        hidden_dim = 4 * D_MODEL
        self.layer1 = Linear(D_MODEL, hidden_dim, bias=False)
        self.act= GELU()
        self.layer2 = Linear(hidden_dim, D_MODEL, bias=False)
        self.dropout = Dropout(dropout)

    def forward(self, x):
        x = self.layer1(x)
        x = self.act(x)
        x = self.layer2(x)
        return self.dropout(x)