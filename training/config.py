import torch

class Config:
    def __init__(self):
        self.vocab_size = 32000
        self.embed_dim = 576
        self.num_transformer_layers = 12
        self.num_heads = 12
        self.attn_dropout = 0.05
        self.ffn_dropout = 0.1
        self.data_dir = 'training/data'
        self.tkn_model = 'models/Marcella_vocab_32K_v2.model'
        self.block_size = 1024
        self.batch_size = 4
        self.num_workers = 0
        self.pin_memory = False
        self.prefetch_factor = None
        self.persistent_workers = False
        self.max_samples = None
        self.validation_prompt = "The Earth circles the star called"
        self.device = torch.device('cuda')
        self.max_seq_len = 1024
        self.accumulation_steps = 8
        self.TOTAL_STEPS = 61_035
        self.WARMUP_STEPS = 610
        self.T_MAX = self.TOTAL_STEPS - self.WARMUP_STEPS
        self.LR_MAX = 2e-4
        self.LR_MIN = 2e-5
        
        self.epochs_ft = 4
        self.dataset_ft = "yahma/alpaca-cleaned"
        self.batch_size_ft = 4
        self.TOTAL_STEPS_ft = 3_250
        self.WARMUP_STEPS_ft = 75
        self.T_MAX_ft = self.TOTAL_STEPS_ft - self.WARMUP_STEPS_ft
        self.LR_MAX_ft = 3e-5
        self.LR_MIN_ft = 3e-6