"""
Compute per-head pre-softmax attention logit std across OLMo-2 training checkpoints.

Architecture notes (OLMo-2 vs Pythia):
  - Layers: model.model.layers[i].self_attn   (vs model.gpt_neox.layers)
  - Projections: separate q_proj, k_proj, v_proj  (vs fused query_key_value)
  - GQA: num_key_value_heads < num_attention_heads; KV heads expanded before scores
  - Optional q_norm/k_norm (OLMo-2 only): applied after projection, before rotary
  - Optional clip_qkv: applied after projection

Hook strategy: register on q_proj (called with hidden_states as a positional arg).
  args[0] = hidden_states; reuse q_proj output, re-run k_proj for K.
  (self_attn is called with all-keyword args, so its hook args tuple is empty.)

Data: HuggingFaceFW/fineweb streaming, prefetched once before the revision loop.
  Only documents with >= context_length tokens are kept (no padding needed).

Output per revision (JSON lines appended to out_file):
  attn_logit_std: list (n_layers, num_heads)  pre-softmax score std
  attn_entropy:   list (n_layers, num_heads)  softmax entropy (nats)

Usage:
    python olmo/eval_attn_logit_std.py \\
        --model allenai/OLMo-2-1124-13B \\
        --revisions stage1-step1100-tokens10B stage1-step596057-tokens5001B \\
        --n_eval_tokens 2000000 \\
        --out_file results/olmo2_13b_attn_logit_std.jsonl
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

CACHE_DIR = "./cache"


# ── RoPE helpers (Llama/OLMo style, defined inline for portability) ─────────────

def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """
    cos/sin must be broadcastable with (B, H, L, head_dim).
    Caller is responsible for shape: pass already-4D tensors.
    """
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """(B, kv_heads, L, D) → (B, kv_heads*n_rep, L, D)."""
    if n_rep == 1:
        return x
    B, H, L, D = x.shape
    return x[:, :, None, :, :].expand(B, H, n_rep, L, D).reshape(B, H * n_rep, L, D)


# ── Model ────────────────────────────────────────────────────────────────────────

def load_model(model_name: str, revision: str) -> AutoModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map="auto",
        # offload_buffers=True,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ── Data prefetch ────────────────────────────────────────────────────────────────

def prefetch_batches(tokenizer, n_tokens: int, context_length: int, batch_size: int):
    """
    Stream fineweb; keep only docs with >= context_length tokens (no padding).
    Returns a list of (batch_size, context_length) int64 tensors.
    """
    n_chunks = n_tokens // context_length
    chunks = []
    data_iter = iter(load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True))

    print(f"Collecting {n_chunks} full-length chunks ({context_length} tok each) …", flush=True)
    docs_seen = 0
    while len(chunks) < n_chunks:
        text = next(data_iter)["text"]
        docs_seen += 1
        ids = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]
        if len(ids) >= context_length:
            chunks.append(ids[:context_length])

    # Group into batches (drop the last incomplete batch)
    batches = []
    for i in range(0, len(chunks) - batch_size + 1, batch_size):
        batches.append(torch.stack(chunks[i : i + batch_size]))  # (B, L)

    actual_tokens = len(batches) * batch_size * context_length
    print(f"  {len(batches)} batches × {batch_size} × {context_length} = "
          f"{actual_tokens / 1e6:.1f}M tokens  (scanned {docs_seen} docs)", flush=True)
    return batches


# ── Core computation ─────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_attn_logit_std(model, batches: list) -> tuple[list, list]:
    """
    Returns (attn_logit_std, attn_entropy), each shape (n_layers, num_heads).
    std:     per-head std of pre-softmax scores, ddof=1, skip q=0, averaged over batches.
    entropy: per-head entropy (nats) of softmax attention, averaged over all q and batches.
    """
    cfg          = model.config
    num_heads    = cfg.num_attention_heads
    num_kv_heads = cfg.num_key_value_heads
    n_kv_groups  = num_heads // num_kv_heads
    head_dim     = cfg.hidden_size // num_heads
    n_layers     = cfg.num_hidden_layers
    clip_qkv     = getattr(cfg, "clip_qkv", None)

    std_sums     = np.zeros((n_layers, num_heads), dtype=np.float64)
    entropy_sums = np.zeros((n_layers, num_heads), dtype=np.float64)
    n_batches    = 0

    # OLMo-2 (newer transformers) puts rotary_emb at model level, not per-layer.
    # OLMo-1 keeps it on each attention module. Detect once and close over below.
    _model_rotary_emb = getattr(model.model, "rotary_emb", None)

    for batch_idx, input_ids in enumerate(batches):
        B, L = input_ids.shape

        hooks = []
        for layer_idx, layer in enumerate(model.model.layers):
            attn = layer.self_attn

            def make_hook(idx: int, attn_mod):
                def hook(_module, args, q_out):
                    # args[0]: hidden_states passed positionally to q_proj
                    # q_out:   output of q_proj (not yet clipped/normed)
                    hs = args[0].detach()           # (B, L, hidden_size), bfloat16
                    B_, L_, _ = hs.shape
                    device = hs.device

                    # Reuse q_proj output; re-run k_proj (no side-effects for nn.Linear)
                    q_flat = q_out.detach()           # (B, L, num_heads*head_dim), bfloat16
                    k_flat = attn_mod.k_proj(hs)      # (B, L, num_kv_heads*head_dim), bfloat16

                    # clip_qkv (OLMo-1 only; OLMo-2 has no clip_qkv in config)
                    if clip_qkv is not None:
                        q_flat = q_flat.clamp(-clip_qkv, clip_qkv)
                        k_flat = k_flat.clamp(-clip_qkv, clip_qkv)

                    # q/k norm applied to FLAT output before reshape (OLMo-2 forward:
                    # query_states = self.q_norm(self.q_proj(hidden_states))).
                    # q_norm weight shape = (num_heads*head_dim,), not (head_dim,).
                    if hasattr(attn_mod, "q_norm"):
                        q_flat = attn_mod.q_norm(q_flat)
                    if hasattr(attn_mod, "k_norm"):
                        k_flat = attn_mod.k_norm(k_flat)

                    # Reshape and cast to float32 for numerical stability
                    q = q_flat.float().view(B_, L_, num_heads, head_dim).transpose(1, 2)
                    k = k_flat.float().view(B_, L_, num_kv_heads, head_dim).transpose(1, 2)
                    del q_flat, k_flat

                    # Rotary embeddings: OLMo-2 has rotary_emb at model level;
                    # OLMo-1 has it per-layer on attn_mod.
                    pos_ids = torch.arange(L_, device=device).unsqueeze(0).expand(B_, -1)
                    rotary = _model_rotary_emb if _model_rotary_emb is not None \
                             else attn_mod.rotary_emb
                    cos, sin = rotary(k, pos_ids)
                    # Normalize cos/sin to 4D for head-dim broadcasting.
                    # OLMo-1/2 return (B, L, D); some newer models return (1, 1, L, D).
                    if cos.dim() == 3:
                        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
                    q, k = apply_rotary_pos_emb(q.to(cos.dtype), k.to(cos.dtype), cos, sin)
                    q, k = q.float(), k.float()

                    # Expand KV for GQA: (B, num_kv_heads, L, D) → (B, num_heads, L, D)
                    k = repeat_kv(k, n_kv_groups)

                    # Pre-softmax scores (B, num_heads, L, L)
                    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)
                    del q, k

                    causal  = torch.tril(torch.ones(L_, L_, device=device, dtype=torch.bool))
                    vcounts = torch.arange(1, L_ + 1, device=device, dtype=torch.float32)

                    # Std (ddof=1), skip q=0 (only 1 valid key → variance undefined)
                    mean_q  = scores.masked_fill(~causal, 0.0).sum(-1) / vcounts  # (B, H, L)
                    diff_sq = (scores - mean_q.unsqueeze(-1)).pow_(2).masked_fill_(~causal, 0.0)
                    var_q   = diff_sq.sum(-1)[:, :, 1:] / (vcounts[1:] - 1)       # (B, H, L-1)
                    std_sums[idx] += var_q.clamp(min=0.0).sqrt_().mean(-1).mean(0).cpu().numpy()
                    del mean_q, diff_sq, var_q

                    # Entropy of softmax attention distribution (nats)
                    probs = torch.softmax(scores.masked_fill_(~causal, float("-inf")), dim=-1)
                    del scores
                    entropy_sums[idx] += (
                        -(probs * probs.clamp(min=1e-10).log_()).sum(-1).mean(-1).mean(0).cpu().numpy()
                    )

                return hook

            hooks.append(attn.q_proj.register_forward_hook(make_hook(layer_idx, attn)))

        first_device = next(model.parameters()).device
        model(input_ids=input_ids.to(first_device), use_cache=False)
        for h in hooks:
            h.remove()

        n_batches += 1
        if (batch_idx + 1) % 10 == 0:
            print(f"  [{batch_idx + 1}/{len(batches)}] batches done", flush=True)

    return (std_sums / n_batches).tolist(), (entropy_sums / n_batches).tolist()


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Per-head attention logit std across OLMo-2 training checkpoints."
    )
    parser.add_argument("--model", default="allenai/OLMo-2-1124-13B")
    parser.add_argument("--revisions", nargs="+", required=True,
                        help="HF revision strings, e.g. stage1-step1100-tokens10B ...")
    parser.add_argument("--n_eval_tokens", type=int, default=2_000_000)
    parser.add_argument("--context_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--out_file", default="results/olmo2_13b_attn_logit_std.jsonl")
    parser.add_argument("--append", action="store_true",
                        help="Append to out_file instead of overwriting")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=CACHE_DIR, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    batches = prefetch_batches(tokenizer, args.n_eval_tokens, args.context_length, args.batch_size)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_file)), exist_ok=True)

    with open(args.out_file, "a" if args.append else "w") as f:
        for rev in args.revisions:
            print(f"\n==> Loading {args.model} @ {rev}")
            t0 = time.time()
            model = load_model(args.model, rev)

            attn_std, attn_entropy = compute_attn_logit_std(model, batches)
            elapsed = time.time() - t0

            row = {
                "model": args.model,
                "revision": rev,
                "attn_logit_std": attn_std,
                "attn_entropy": attn_entropy,
                "elapsed_s": round(elapsed, 1),
            }
            n_layers = len(attn_std)
            n_heads  = len(attn_std[0]) if n_layers > 0 else 0
            print(f"  shape: ({n_layers}, {n_heads})")
            print(f"  layer-0 head stds:    {[round(x, 3) for x in attn_std[0]]}")
            print(f"  layer-0 head entropy: {[round(x, 3) for x in attn_entropy[0]]}")
            f.write(json.dumps(row) + "\n")
            f.flush()

            del model
            torch.cuda.empty_cache()

    print(f"\nDone. Results → {args.out_file}")


if __name__ == "__main__":
    main()
