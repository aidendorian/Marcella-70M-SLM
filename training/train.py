import torch
from bitsandbytes.optim.adamw import AdamW8bit
from tqdm import tqdm
from training.config import Config
from torch.nn import CrossEntropyLoss
from torch.amp.autocast_mode import autocast
import os
from training.dataloader import get_data
from src.marcella import Marcella
from training.checkpoint import load_checkpoint, save_checkpoint

config = Config()
device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')
print(f'Device: {device}')

os.mkdir('training/checkpoints', )

model = Marcella(vocab_size=config.vocab_size, 
                 embed_dim=config.embed_dim,
                 num_transformer_layers=config.num_transformer_layers,
                 num_heads=config.num_heads,
                 attn_dropout=config.attn_dropout,
                 ffn_dropout=config.ffn_dropout
                 ).to(device)

assert model is not None

optimizer = AdamW8bit(model.parameters(), lr=1e-4)

data = get_data(dataset_name=config.dataset_name,
                dataset_split=config.dataset_split,
                tkn_model=config.tkn_model,
                block_size=config.block_size,
                batch_size=config.batch_size,
                num_workers=config.batch_size,
                pin_memory=config.pin_memory,
                prefetch_factor=config.prefetch_factor,
                persistent_workers=config.persistent_workers,
                max_samples=config.max_samples)

loss_fn = CrossEntropyLoss()

RESUME_FROM_CHECKPOINT = False
checkpoint_path = ''

start_iter = 0

if RESUME_FROM_CHECKPOINT:
    _, model, optimizer, start_iter = load_checkpoint(model, optimizer, checkpoint_path)
    print(f'Resuming Training from Iter: {start_iter} from {checkpoint_path}')

for i, (x, y) in enumerate(tqdm(data, desc='Training')):
    model.train()
    if start_iter > i:
        continue
    
    print(f'Starting Training from iter {i}')
    
    x, y = x.to(device), y.to(device)
    logits = model(x)
    B, T, C = logits.shape
    logits = logits.view(B*T, C)
    y = y.view(B*T)
    loss = loss_fn(logits, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if i % 100000:
        save_checkpoint(model, optimizer, 'training/checkpoints', f'{i}_marcella.pth', loss, i)