import torch
import math
from src.marcella import Marcella
from training.config import Config
from training.checkpoint import save_checkpoint
from bitsandbytes.optim.adamw import AdamW8bit
from torch.optim import lr_scheduler
from training.finetuning_dataloader import get_data
from torch.nn import CrossEntropyLoss
from torch.amp.autocast_mode import autocast

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(False)

config = Config()

model = Marcella(
    vocab_size=config.vocab_size,
    embed_dim=config.embed_dim,
    num_transformer_layers=config.num_transformer_layers,
    num_heads=config.num_heads,
    attn_dropout=config.attn_dropout,
    ffn_dropout=config.ffn_dropout,
).to(config.device)

pretrain_ckpt = torch.load("models/marcella_pretrained.pt", map_location=config.device, weights_only=True)
state = pretrain_ckpt.get("model_state_dict", pretrain_ckpt)
model.load_state_dict(state)
print(f"[finetune] loaded pretrained weights")

no_decay = {"bias", "norm"}
param_groups = [
    {
        "params": [p for n, p in model.named_parameters()
                   if not any(nd in n for nd in no_decay)],
        "weight_decay": 0.1,
    },
    {
        "params": [p for n, p in model.named_parameters()
                   if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]

optimizer = AdamW8bit(param_groups, lr=config.LR_MAX_ft, betas=(0.9, 0.95), eps=1e-8)

def lr_lambda(step: int) -> float:
    if step < config.WARMUP_STEPS_ft:
        return step / max(1, config.WARMUP_STEPS_ft)
    t      = step - config.WARMUP_STEPS_ft
    cosine = 0.5 * (1.0 + math.cos(math.pi * t / config.T_MAX_ft))
    lr     = config.LR_MIN_ft + (config.LR_MAX_ft - config.LR_MIN_ft) * cosine
    return lr / config.LR_MAX_ft

scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

torch._dynamo.config.suppress_errors = True
compiled_model = torch.compile(model, mode="default")

data    = get_data()
loss_fn = CrossEntropyLoss(ignore_index=-1)

@torch.no_grad()
def compute_val_loss(val_x, val_y, val_mask):
    compiled_model.eval()  # type: ignore
    with autocast(device_type="cuda", dtype=torch.bfloat16):
        logits     = compiled_model(val_x)
        B, T, C    = logits.shape
        masked_y   = val_y.clone()
        masked_y[~val_mask] = -1
        loss = loss_fn(logits.view(B * T, C), masked_y.view(B * T))
    compiled_model.train() # type: ignore
    return loss.item()

compiled_model.train() # type: ignore
optimizer.zero_grad()

step = 0
accum_loss = 0.0
val_x_cache = None
val_y_cache = None
val_mask_cache = None

print(f"[finetune] starting — {config.TOTAL_STEPS_ft} steps, "
      f"LR {config.LR_MAX_ft:.1e}→{config.LR_MIN_ft:.1e}")

for x, y, mask in data:
    x    = x.to(config.device)
    y    = y.to(config.device)
    mask = mask.to(config.device)

    if val_x_cache is None:
        val_x_cache = x.clone()
        val_y_cache = y.clone()
        val_mask_cache = mask.clone()

    with autocast(device_type="cuda", dtype=torch.bfloat16):
        logits  = compiled_model(x)
        B, T, C = logits.shape

        masked_y = y.clone()
        masked_y[~mask] = -1

        loss = loss_fn(logits.view(B * T, C), masked_y.view(B * T))
        loss = loss / config.accumulation_steps

    loss.backward()
    accum_loss += loss.item()

    if (step + 1) % config.accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        opt_step = (step + 1) // config.accumulation_steps
        lr_now = optimizer.param_groups[0]['lr']

        if opt_step % 50 == 0:
            val_loss = compute_val_loss(val_x_cache, val_y_cache, val_mask_cache)
            print(f"step {opt_step:>5} | train loss {accum_loss:.4f} "
                  f"| val loss {val_loss:.4f} | lr {lr_now:.2e}")

        if opt_step % 500 == 0:
            save_checkpoint(
                model, optimizer, scheduler,
                save_dir="training/checkpoints",
                filename=f"finetune_{opt_step}.pth",
                loss=accum_loss,
                iteration=opt_step,
            )
        accum_loss = 0.0

    step += 1

save_checkpoint(
    model, optimizer, scheduler,
    save_dir="training/checkpoints",
    filename="finetune_final.pth",
    iteration=step,
)

torch.save(model.state_dict(), "models/marcella_finetuned.pt")