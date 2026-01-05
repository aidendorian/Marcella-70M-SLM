import torch
from torch.nn import LayerNorm, GELU, Module, Linear, Dropout, Embedding
from torch.optim import AdamW
from src.attention import Attention

class TransformerBlock(Module):
    def __init__(self,
                 embed_dim:int,
                 num_heads:int,
                 ffn_dropout:float,
                 attn_dropout:float):
        super().__init__()
        
        self.norm1 = LayerNorm(embed_dim)
        self.attn = Attention(embed_dim=embed_dim,
                              num_heads=num_heads,
                              flash_attn_dropout=attn_dropout)
        self.norm2 = LayerNorm(embed_dim)
        self.ffn = FeedForwardNetwork(embed_dim=embed_dim,
                                      dropout=ffn_dropout)
        
    def forward(self, x, kv_cache=None):
        attn_out = self.attn(self.norm1(x), kv_cache)
        x = x + attn_out
        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out
        return x
    
class FeedForwardNetwork(Module):
    def __init__(self,
                 embed_dim,
                 dropout):
        super().__init__()

        self.layer1 = Linear(embed_dim, 4 * embed_dim, bias=False)
        self.act= GELU()
        self.layer2 = Linear(4 * embed_dim, embed_dim, bias=False)
        self.dropout = Dropout(dropout)

    def forward(self, x):
        x = self.layer1(x)
        x = self.act(x)
        x = self.layer2(x)
        return self.dropout(x)
    
class Marcella(Module):
    def __init__(self,
                 vocab_size:int=32000,
                 embed_dim:int=768,
                 num_transformer_layers:int=32,
                 num_heads:int=12,
                 attn_dropout:float=0.0,
                 ):
        super().__init__()
        
        self.token_embed = Embedding(vocab_size, embed_dim)
        self.lm_head = Embedding(embed_dim, vocab_size)