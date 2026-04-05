import asyncio
import json
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.marcella import Marcella
from src.tokenizer import Tokenizer
from training.config import Config

cfg = Config()
app = FastAPI(title="Marcella API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

tokenizer = Tokenizer(tokenizer_model=cfg.tkn_model)

MODELS = {
    "pretrained": "models/marcella_pretrained.pt",
    "finetuned":  "models/marcella_finetuned.pt",
}

state = {
    "model":        None,
    "active_model": "pretrained",
}

INSTRUCTION_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"


def _apply_chat_template(prompt: str) -> str:
    """Wrap a raw prompt in the instruction/response template used during finetuning."""
    return INSTRUCTION_TEMPLATE.format(instruction=prompt)


def _build_model() -> Marcella:
    m = Marcella(
        vocab_size=cfg.vocab_size,
        embed_dim=cfg.embed_dim,
        num_transformer_layers=cfg.num_transformer_layers,
        num_heads=cfg.num_heads,
    )
    m.to(cfg.device)
    m.eval()
    return m


def _load_weights(model: Marcella, model_key: str) -> Marcella:
    path = MODELS[model_key]
    ckpt = torch.load(path, map_location=cfg.device, weights_only=True)
    weights = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(weights)
    return model


state["model"] = _load_weights(_build_model(), "pretrained")
print(f"[marcella] loaded pretrained — {sum(p.numel() for p in state['model'].parameters()):,} params")


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int    = Field(default=200, ge=1,   le=1024)
    temperature: float = Field(default=0.8, ge=0.1, le=1.0)
    top_k: int         = Field(default=50,  ge=1,   le=500)

class LoadModelRequest(BaseModel):
    model: str


@app.post("/load_model")
def load_model(req: LoadModelRequest):
    if req.model not in MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model '{req.model}'. Choose: {list(MODELS.keys())}")
    if req.model == state["active_model"]:
        return {"status": "already_loaded", "model": req.model}

    try:
        _load_weights(state["model"], req.model)
        state["active_model"] = req.model
        print(f"[marcella] switched to {req.model}")
        return {"status": "ok", "model": req.model}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Weights file not found: {MODELS[req.model]}")


@app.get("/active_model")
def active_model():
    return {"model": state["active_model"]}


_DONE = object()


def token_id_to_text(token_id: int) -> str:
    piece = tokenizer.tokenizer.IdToPiece(token_id)
    return piece.replace("▁", " ")


@torch.inference_mode()
def _generate_tokens(req: GenerateRequest):
    model = state["model"]
    prompt = (
        _apply_chat_template(req.prompt)
        if state["active_model"] == "finetuned"
        else req.prompt
    )

    ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([ids], dtype=torch.long, device=cfg.device)

    kv_cache = model.init_kv_cache(
        batch_size=1,
        max_seq_len=cfg.max_seq_len,
        device=cfg.device,
        dtype=torch.float32,
    )

    logits = model(input_ids, kv_cache)

    for _ in range(req.max_tokens):
        next_logits = logits[:, -1, :] / req.temperature

        if req.top_k > 0:
            vals, _ = torch.topk(next_logits, req.top_k)
            next_logits = next_logits.masked_fill(
                next_logits < vals[:, -1].unsqueeze(-1), float("-inf")
            )

        probs   = F.softmax(next_logits.float(), dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)

        if next_id.item() == tokenizer.eos_id:
            break

        yield token_id_to_text(next_id.item())  # type: ignore
        logits = model(next_id, kv_cache)

    yield _DONE


def _next_token(gen):
    return next(gen)


@app.post("/generate")
async def generate(req: GenerateRequest):
    async def event_stream():
        loop = asyncio.get_event_loop()
        gen  = _generate_tokens(req)
        while True:
            result = await loop.run_in_executor(None, _next_token, gen)
            if result is _DONE:
                yield "data: [DONE]\n\n"
                break
            payload = json.dumps({"token": result})
            yield f"data: {payload}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/health")
def health():
    return {"status": "ok", "device": str(cfg.device), "model": state["active_model"]}