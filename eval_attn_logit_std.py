"""
Compute per-head pre-softmax attention logit std across training checkpoints.

For each attention layer and head:
  - Hook query_key_value projection; extract Q, K, V and apply the model's own
    rotary embeddings (RotaryEmbedding + apply_rotary_pos_emb) to match Pythia's
    internal computation faithfully.
  - Compute attention scores: (Q_rot @ K_rot^T) / sqrt(head_size)
  - For each query position q, compute std over valid key positions 0..q (causal mask).
  - Average over query positions (skip q=0 which has only 1 valid key) and batch.

Output per revision:
  attn_logit_std: list of shape (n_layers, num_heads), values are float
  attn_entropy:   list of shape (n_layers, num_heads), values are float (nats)

Usage:
    python eval_attn_logit_std.py \\
        --model EleutherAI/pythia-160m \\
        --revisions step128 step256 step512 \\
        --data_path data/pythia_deduped_pile_idxmaps/ \\
        --clean_eval \\
        --n_eval_tokens 200000 \\
        --out_file results/eval_attn_logit_std.jsonl
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoModelForCausalLM
from transformers.models.gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb

from data import PileMemmap
from eval import get_clean_eval_indices


def load_model(path: str, device: torch.device,
               revision: str | None = None) -> AutoModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        revision=revision,
        cache_dir="./cache",
        low_cpu_mem_usage=True,
        device_map=device,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def compute_attn_logit_std(model, loader: DataLoader, device: torch.device) -> list:
    """
    Returns (attn_logit_std, attn_entropy), each list[list[float]] shape (n_layers, num_heads).
    std: per-head std of pre-softmax scores, averaged over q>=1 positions and batches.
    entropy: per-head entropy (nats) of softmax attention distribution, averaged over
             all query positions and batches.
    """
    cfg = model.config
    num_heads = cfg.num_attention_heads
    head_size = cfg.hidden_size // num_heads
    n_layers = cfg.num_hidden_layers

    std_sums = np.zeros((n_layers, num_heads), dtype=np.float64)
    entropy_sums = np.zeros((n_layers, num_heads), dtype=np.float64)
    n_batches = 0

    for batch_idx, (input_ids, _) in enumerate(loader):
        input_ids = input_ids.to(device)
        B, L = input_ids.shape

        position_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        # causal_mask[q, k] = True iff k <= q  (valid attention positions)
        causal_mask = torch.tril(torch.ones(L, L, device=device, dtype=torch.bool))
        # valid_counts[q] = q + 1
        valid_counts = torch.arange(1, L + 1, device=device, dtype=torch.float32)

        # Process each layer inside its own hook so we never accumulate all layers'
        # QKV tensors simultaneously — only one (B, H, L, L) score matrix exists at
        # a time, which is the dominant memory cost for large models.
        hooks = []
        for i, layer in enumerate(model.gpt_neox.layers):
            def make_hook(idx: int, attn_mod):
                def hook(_module, _input, output):
                    qkv = output.detach().float()  # (B, L, H*3*D)
                    qkv = qkv.view(B, L, num_heads, 3 * head_size)
                    q = qkv[..., :head_size].permute(0, 2, 1, 3).contiguous()
                    k = qkv[..., head_size:2 * head_size].permute(0, 2, 1, 3).contiguous()
                    v = qkv[..., 2 * head_size:].permute(0, 2, 1, 3)
                    del qkv

                    rot_dim = attn_mod.rotary_ndims
                    cos, sin = attn_mod.rotary_emb(v, seq_len=L)
                    del v
                    q_r, k_r = apply_rotary_pos_emb(
                        q[..., :rot_dim], k[..., :rot_dim], cos, sin, position_ids
                    )
                    del cos, sin
                    q_rot = torch.cat((q_r, q[..., rot_dim:]), dim=-1)
                    k_rot = torch.cat((k_r, k[..., rot_dim:]), dim=-1)
                    del q, k, q_r, k_r

                    # Pre-softmax attention scores: (B, H, L, L)
                    scores = torch.matmul(q_rot, k_rot.transpose(-1, -2)) * (head_size ** -0.5)
                    del q_rot, k_rot

                    # Std (ddof=1), skip q=0
                    mean_q = scores.masked_fill(~causal_mask, 0.0).sum(dim=-1) / valid_counts
                    diff_sq = (scores - mean_q.unsqueeze(-1)).pow_(2).masked_fill_(~causal_mask, 0.0)
                    del mean_q
                    var_q = diff_sq.sum(dim=-1)[:, :, 1:] / (valid_counts[1:] - 1)
                    del diff_sq
                    std_sums[idx] += var_q.clamp_(min=0.0).sqrt_().mean(dim=-1).mean(dim=0).cpu().numpy()
                    del var_q

                    # Entropy of softmax attention distribution (nats)
                    probs = torch.softmax(scores.masked_fill_(~causal_mask, float('-inf')), dim=-1)
                    del scores
                    entropy_q = -(probs * probs.clamp(min=1e-10).log()).sum(dim=-1)
                    del probs
                    entropy_sums[idx] += entropy_q.mean(dim=-1).mean(dim=0).cpu().numpy()

                return hook
            hooks.append(layer.attention.query_key_value.register_forward_hook(
                make_hook(i, layer.attention)
            ))

        model(input_ids=input_ids, use_cache=False)
        for h in hooks:
            h.remove()

        n_batches += 1

        if (batch_idx + 1) % 10 == 0:
            print(f"  [{batch_idx + 1}/{len(loader)}] batches done", flush=True)

    return (std_sums / n_batches).tolist(), (entropy_sums / n_batches).tolist()


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-head attention logit std across training checkpoints."
    )
    parser.add_argument("--model", required=True,
                        help="HuggingFace model ID or local save_pretrained path")
    parser.add_argument("--revisions", nargs="+", type=str, default=["none"],
                        help="HF revisions (e.g. step128 step256). Use 'none' for no revision.")
    parser.add_argument("--data_path", type=str, required=True)

    data_mode = parser.add_mutually_exclusive_group()
    data_mode.add_argument("--clean_eval", action="store_true")
    data_mode.add_argument("--eval_shard_offset", type=int, default=None)

    parser.add_argument("--train_steps", type=int, default=79000)
    parser.add_argument("--train_world_size", type=int, default=8)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--sampler_seed", type=int, default=0)

    parser.add_argument("--n_eval_tokens", type=int, default=200_000)
    parser.add_argument("--context_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--out_file", type=str, default="results/eval_attn_logit_std.jsonl")
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()

    if not args.clean_eval and args.eval_shard_offset is None:
        args.clean_eval = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_eval_chunks = args.n_eval_tokens // args.context_length

    if args.clean_eval:
        full_dataset = PileMemmap(args.data_path, args.context_length)
        clean_indices = get_clean_eval_indices(
            total_chunks=len(full_dataset),
            world_size=args.train_world_size,
            batch_size=args.train_batch_size,
            steps=args.train_steps,
            sampler_seed=args.sampler_seed,
        )
        dataset = Subset(full_dataset, clean_indices[:n_eval_chunks].tolist())
        print(f"Using {len(dataset)} clean chunks ({len(dataset) * args.context_length / 1e6:.1f}M tokens)")
    else:
        from eval import EvalPileMemmap
        dataset = EvalPileMemmap(
            args.data_path,
            context_length=args.context_length,
            shard_offset=args.eval_shard_offset,
            max_tokens=args.n_eval_tokens,
        )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    print(f"Batches: {len(loader)}\n")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_file)), exist_ok=True)

    with open(args.out_file, "a" if args.append else "w") as f:
        for rev in args.revisions:
            revision = None if rev == "none" else rev
            print(f"Loading: {args.model} (revision={revision})")
            t0 = time.time()
            model = load_model(args.model, device, revision=revision)

            attn_std, attn_entropy = compute_attn_logit_std(model, loader, device)
            elapsed = time.time() - t0

            row = {
                "model": args.model,
                "revision": revision,
                "attn_logit_std": attn_std,
                "attn_entropy": attn_entropy,
                "elapsed_s": round(elapsed, 1),
            }
            n_layers = len(attn_std)
            n_heads = len(attn_std[0]) if n_layers > 0 else 0
            print(f"  attn_logit_std shape: ({n_layers}, {n_heads})")
            print(f"  layer 0 head stds:    {[round(x, 3) for x in attn_std[0]]}")
            print(f"  layer 0 head entropy: {[round(x, 3) for x in attn_entropy[0]]}")
            f.write(json.dumps(row) + "\n")
            f.flush()

            del model
            torch.cuda.empty_cache()

    print(f"\nResults written to {args.out_file}")


if __name__ == "__main__":
    main()
