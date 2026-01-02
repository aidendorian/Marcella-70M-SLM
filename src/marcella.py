import torch
from torch.nn import MultiheadAttention, LayerNorm, GELU, Module, Linear, Dropout
from torch.optim import AdamW
from torch.nn.functional import scaled_dot_product_attention

N_HEADS = 12
D_MODEL = 384
VOCAB = 32000
ATTN_DROPOUT = 0.1

