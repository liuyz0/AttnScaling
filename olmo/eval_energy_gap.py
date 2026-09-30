"""
Compute energy gap histograms for attention heads and LM head across OLMo-2 checkpoints.

Energy gap for logit vector y (valid positions only):
    gap = (max(y) - second_max(y)) / std(y)

For attention heads: y = pre-softmax scores over valid (causal) key positions (q >= 1).
For LM head:        y = full vocabulary logits at each token position.

Hook strategy: same as eval_attn_logit_std.py — hook q_proj (positional hidden_states arg),
reuse its output for Q, re-run k_proj for K, apply norms/rotary/GQA expansion, compute scores.

Output per revision:
  <out_dir>/<revision>.npz          → attn_hist, attn_overflow, lm_hist, lm_overflow, bin_edges
  <out_dir>/<revision>_meta.json    → metadata

Usage:
    python olmo/eval_energy_gap.py \\
        --model allenai/OLMo-2-1124-13B \\
        --revisions stage1-step1100-tokens10B stage1-step596057-tokens5001B \\
        --n_eval_tokens 2000000 \\
        --out_dir results/energy_gap/olmo2_13b
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


# ── RoPE helpers ─────────────────────────────────────────────────────────────────

def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """cos/sin must be broadcastable with (B, H, L, head_dim); caller unsqueezes if needed."""
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
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
        offload_buffers=True,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ── Data prefetch ────────────────────────────────────────────────────────────────

def prefetch_batches(tokenizer, n_tokens: int, context_length: int, batch_size: int):
    """Stream fineweb; keep only docs >= context_length tokens (no padding)."""
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

    batches = []
    for i in range(0, len(chunks) - batch_size + 1, batch_size):
        batches.append(torch.stack(chunks[i : i + batch_size]))

    actual_tokens = len(batches) * batch_size * context_length
    print(f"  {len(batches)} batches × {batch_size} × {context_length} = "
          f"{actual_tokens / 1e6:.1f}M tokens  (scanned {docs_seen} docs)", flush=True)
    return batches


# ── Core computation ─────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_energy_gap_hists(
    model, batches: list, bin_edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.int64]:
    """
    Returns (attn_hist, attn_overflow, lm_hist, lm_overflow).
      attn_hist:     (n_layers, num_heads, n_bins) int64
      attn_overflow: (n_layers, num_heads)          int64
      lm_hist:       (n_bins,)                      int64
      lm_overflow:   scalar                         int64
    """
    cfg          = model.config
    num_heads    = cfg.num_attention_heads
    num_kv_heads = cfg.num_key_value_heads
    n_kv_groups  = num_heads // num_kv_heads
    head_dim     = cfg.hidden_size // num_heads
    n_layers     = cfg.num_hidden_layers
    clip_qkv     = getattr(cfg, "clip_qkv", None)
    n_bins       = len(bin_edges) - 1
    gap_max      = float(bin_edges[-1])

    attn_hist     = np.zeros((n_layers, num_heads, n_bins), dtype=np.int64)
    attn_overflow = np.zeros((n_layers, num_heads), dtype=np.int64)
    lm_hist       = np.zeros(n_bins, dtype=np.int64)
    lm_overflow   = np.int64(0)

    # OLMo-2 (newer transformers) puts rotary_emb at model level, not per-layer.
    _model_rotary_emb = getattr(model.model, "rotary_emb", None)

    for batch_idx, input_ids in enumerate(batches):
        B, L = input_ids.shape

        hooks = []
        for layer_idx, layer in enumerate(model.model.layers):
            attn = layer.self_attn

            def make_hook(idx: int, attn_mod):
                def hook(_module, args, q_out):
                    hs = args[0].detach()
                    B_, L_, _ = hs.shape
                    device = hs.device

                    q_flat = q_out.detach()           # (B, L, num_heads*head_dim), bfloat16
                    k_flat = attn_mod.k_proj(hs)      # (B, L, num_kv_heads*head_dim), bfloat16

                    # clip_qkv (OLMo-1 only; OLMo-2 has no clip_qkv)
                    if clip_qkv is not None:
                        q_flat = q_flat.clamp(-clip_qkv, clip_qkv)
                        k_flat = k_flat.clamp(-clip_qkv, clip_qkv)

                    # q/k norm applied to FLAT output before reshape (OLMo-2 forward:
                    # query_states = self.q_norm(self.q_proj(hidden_states))).
                    if hasattr(attn_mod, "q_norm"):
                        q_flat = attn_mod.q_norm(q_flat)
                    if hasattr(attn_mod, "k_norm"):
                        k_flat = attn_mod.k_norm(k_flat)

                    q = q_flat.float().view(B_, L_, num_heads, head_dim).transpose(1, 2)
                    k = k_flat.float().view(B_, L_, num_kv_heads, head_dim).transpose(1, 2)
                    del q_flat, k_flat

                    pos_ids = torch.arange(L_, device=device).unsqueeze(0).expand(B_, -1)
                    rotary = _model_rotary_emb if _model_rotary_emb is not None \
                             else attn_mod.rotary_emb
                    cos, sin = rotary(k, pos_ids)
                    if cos.dim() == 3:
                        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
                    q, k = apply_rotary_pos_emb(q.to(cos.dtype), k.to(cos.dtype), cos, sin)
                    q, k = q.float(), k.float()
                    k = repeat_kv(k, n_kv_groups)

                    # Pre-softmax scores (B, num_heads, L, L)
                    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)
                    del q, k

                    causal  = torch.tril(torch.ones(L_, L_, device=device, dtype=torch.bool))
                    vcounts = torch.arange(1, L_ + 1, device=device, dtype=torch.float32)

                    # Std over valid (causal) keys per query position, skip q=0
                    scores_v = scores.masked_fill(~causal, 0.0)
                    mean_q   = scores_v.sum(-1) / vcounts                                # (B, H, L)
                    diff_sq  = ((scores - mean_q.unsqueeze(-1)) ** 2).masked_fill(~causal, 0.0)
                    std_q    = (diff_sq.sum(-1)[:, :, 1:] / (vcounts[1:] - 1)).clamp(min=0.0).sqrt()
                    # std_q: (B, H, L-1), for q=1..L-1
                    del mean_q, diff_sq, scores_v

                    # Top-2 scores over valid keys, skip q=0
                    scores_ni = scores.masked_fill(~causal, float("-inf"))
                    del scores
                    top2 = scores_ni[:, :, 1:, :].topk(2, dim=-1).values  # (B, H, L-1, 2)
                    del scores_ni

                    gap = (top2[..., 0] - top2[..., 1]) / std_q.clamp(min=1e-8)  # (B, H, L-1)
                    del top2, std_q

                    # Histogram per head using torch.histc (faster than np.histogram)
                    for h in range(num_heads):
                        g = gap[:, h, :].reshape(-1).cpu().float()
                        hist = torch.histc(g, bins=n_bins, min=0.0, max=gap_max).long().numpy()
                        attn_hist[idx, h] += hist
                        attn_overflow[idx, h] += int((g > gap_max).sum())
                    del gap

                return hook

            hooks.append(attn.q_proj.register_forward_hook(make_hook(layer_idx, attn)))

        first_device = next(model.parameters()).device
        outputs = model(input_ids=input_ids.to(first_device), use_cache=False)
        for h in hooks:
            h.remove()

        # LM head energy gap: logits (B, L, vocab_size)
        logits = outputs.logits.float()
        std_lm  = logits.std(dim=-1)                          # (B, L)
        top2_lm = logits.topk(2, dim=-1).values               # (B, L, 2)
        del logits
        gap_lm = (top2_lm[..., 0] - top2_lm[..., 1]) / std_lm.clamp(min=1e-8)
        del std_lm, top2_lm
        g_lm = gap_lm.reshape(-1).cpu().float()
        del gap_lm
        lm_hist += torch.histc(g_lm, bins=n_bins, min=0.0, max=gap_max).long().numpy()
        lm_overflow += np.int64(int((g_lm > gap_max).sum()))

        if (batch_idx + 1) % 10 == 0:
            print(f"  [{batch_idx + 1}/{len(batches)}] batches done", flush=True)

    return attn_hist, attn_overflow, lm_hist, lm_overflow


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Energy gap histograms for attention heads and LM head across OLMo-2 checkpoints."
    )
    parser.add_argument("--model", default="allenai/OLMo-2-1124-13B")
    parser.add_argument("--revisions", nargs="+", required=True)
    parser.add_argument("--n_eval_tokens", type=int, default=2_000_000)
    parser.add_argument("--context_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gap_max", type=float, default=5.0,
                        help="Upper histogram edge; values above are counted as overflow")
    parser.add_argument("--n_bins", type=int, default=1000)
    parser.add_argument("--out_dir", default="results/energy_gap/olmo2_13b")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=CACHE_DIR, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    batches  = prefetch_batches(tokenizer, args.n_eval_tokens, args.context_length, args.batch_size)
    bin_edges = np.linspace(0.0, args.gap_max, args.n_bins + 1)

    os.makedirs(args.out_dir, exist_ok=True)

    for rev in args.revisions:
        print(f"\n==> Loading {args.model} @ {rev}")
        t0 = time.time()
        model = load_model(args.model, rev)

        attn_hist, attn_overflow, lm_hist, lm_overflow = compute_energy_gap_hists(
            model, batches, bin_edges
        )
        elapsed = time.time() - t0

        # Sanitize revision string for filenames (replace slashes)
        rev_key = rev.replace("/", "_")
        out_npz = os.path.join(args.out_dir, f"{rev_key}.npz")
        np.savez_compressed(
            out_npz,
            attn_hist=attn_hist,
            attn_overflow=attn_overflow,
            lm_hist=lm_hist,
            lm_overflow=np.int64(lm_overflow),
            bin_edges=bin_edges,
        )

        attn_total = int(attn_hist.sum()) + int(attn_overflow.sum())
        lm_total   = int(lm_hist.sum()) + int(lm_overflow)
        meta = {
            "model": args.model,
            "revision": rev,
            "n_eval_tokens": args.n_eval_tokens,
            "gap_max": args.gap_max,
            "n_bins": args.n_bins,
            "elapsed_s": round(elapsed, 1),
            "attn_hist_shape": list(attn_hist.shape),
            "attn_overflow_frac": round(float(attn_overflow.sum()) / attn_total, 5) if attn_total > 0 else 0.0,
            "lm_overflow_frac":   round(float(lm_overflow) / lm_total, 5)           if lm_total > 0 else 0.0,
        }
        with open(os.path.join(args.out_dir, f"{rev_key}_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        n_layers, n_heads, _ = attn_hist.shape
        print(f"  attn_hist shape: ({n_layers}, {n_heads}, {args.n_bins})")
        print(f"  attn overflow:   {int(attn_overflow.sum())} ({100 * meta['attn_overflow_frac']:.2f}%)")
        print(f"  lm  overflow:    {int(lm_overflow)} ({100 * meta['lm_overflow_frac']:.2f}%)")
        print(f"  saved to {out_npz}")

        del model
        torch.cuda.empty_cache()

    print(f"\nDone. Results → {args.out_dir}/")


if __name__ == "__main__":
    main()
