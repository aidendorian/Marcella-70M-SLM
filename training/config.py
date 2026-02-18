import torch

class Config:
    def __init__(self):
        self.vocab_size = 32000
        self.embed_dim = 576
        self.num_transformer_layers = 12
        self.num_heads = 12
        self.attn_dropout = 0.05
        self.ffn_dropout = 0.1
        self.dataset_name = "draco976/wikipedia-bookcorpus"
        self.dataset_split = 'train'
        self.tkn_model = 'models/Marcella_vocab_32K.model'
        self.block_size = 512
        self.batch_size = 4
        self.num_workers = 4
        self.pin_memory = True
        self.prefetch_factor = 2
        self.persistent_workers = True
        self.max_samples = None
        self.validation_prompt = 'It takes a special amount of patience to handle it, special care and especially some very accurate love. Yet you toss it out with such insignificance, it may not be important to you - but it is my heart.'
        self.device = torch.device('cuda')
        self.max_seq_len = 512
        self.accumulation_steps = 6