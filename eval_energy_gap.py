"""
Compute energy gap histograms for attention heads and LM head across checkpoints.

Energy gap for a logit vector y (masked positions excluded):
    gap = (max(y) - second_max(y)) / std(y)

For each attention head, y = pre-softmax scores over valid (causal) key positions.
For the LM head, y = logits over the full vocabulary (no masking).

Distributions are stored as fixed-bin histograms to avoid saving all raw values.

Output per revision: <out_dir>/<revision>.npz
  attn_hist:      (n_layers, n_heads, n_bins)  int64
  attn_overflow:  (n_layers, n_heads)          int64  -- counts > gap_max
  lm_hist:        (n_bins,)                    int64
  lm_overflow:    scalar                       int64
  bin_edges:      (n_bins+1,)                  float64

Usage:
    python eval_energy_gap.py \\
        --model EleutherAI/pythia-410m \\
        --revisions none \\
        --data_path data/pythia_deduped_pile_idxmaps/ \\
        --clean_eval \\
        --n_eval_tokens 2000000 \\
        --out_dir results/energy_gap
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
def compute_energy_gap_hists(
    model, loader: DataLoader, device: torch.device, bin_edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.int64]:
    cfg = model.config
    num_heads = cfg.num_attention_heads
    head_size = cfg.hidden_size // num_heads
    n_layers = cfg.num_hidden_layers
    n_bins = len(bin_edges) - 1
    gap_max = float(bin_edges[-1])

    attn_hist = np.zeros((n_layers, num_heads, n_bins), dtype=np.int64)
    attn_overflow = np.zeros((n_layers, num_heads), dtype=np.int64)
    lm_hist = np.zeros(n_bins, dtype=np.int64)
    lm_overflow = np.int64(0)

    for batch_idx, (input_ids, _) in enumerate(loader):
        input_ids = input_ids.to(device)
        B, L = input_ids.shape

        qkv_outputs: dict[int, torch.Tensor] = {}
        hooks = []
        for i, layer in enumerate(model.gpt_neox.layers):
            def make_hook(idx: int):
                def hook(_module, _input, output):
                    qkv_outputs[idx] = output.detach()
                return hook
            hooks.append(layer.attention.query_key_value.register_forward_hook(make_hook(i)))

        outputs = model(input_ids=input_ids, use_cache=False)
        for h in hooks:
            h.remove()

        # LM head energy gaps: (B, L, vocab_size) → (B*L,)
        logits = outputs.logits.float()  # (B, L, vocab_size)
        std_lm = logits.std(dim=-1)      # (B, L)
        top2_lm = logits.topk(2, dim=-1).values  # (B, L, 2)
        del logits
        gap_lm = (top2_lm[..., 0] - top2_lm[..., 1]) / std_lm.clamp(min=1e-8)
        del std_lm, top2_lm
        gap_lm_np = gap_lm.reshape(-1).cpu().numpy()
        del gap_lm
        counts, _ = np.histogram(gap_lm_np, bins=bin_edges)
        lm_hist += counts
        lm_overflow += np.int64((gap_lm_np > gap_max).sum())

        # Shared tensors for attention
        position_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        causal_mask = torch.tril(torch.ones(L, L, device=device, dtype=torch.bool))
        valid_counts = torch.arange(1, L + 1, device=device, dtype=torch.float32)
        n_valid = valid_counts[1:]  # [2, 3, ..., L] for q=1..L-1

        for i in range(n_layers):
            qkv = qkv_outputs[i].float()  # (B, L, num_heads * 3 * head_size)
            qkv = qkv.view(B, L, num_heads, 3 * head_size)
            q = qkv[..., :head_size].permute(0, 2, 1, 3)
            k = qkv[..., head_size: 2 * head_size].permute(0, 2, 1, 3)
            v = qkv[..., 2 * head_size:].permute(0, 2, 1, 3)

            attn_mod = model.gpt_neox.layers[i].attention
            rot_dim = attn_mod.rotary_ndims
            q_r, q_p = q[..., :rot_dim], q[..., rot_dim:]
            k_r, k_p = k[..., :rot_dim], k[..., rot_dim:]
            cos, sin = attn_mod.rotary_emb(v, seq_len=L)
            q_r, k_r = apply_rotary_pos_emb(q_r, k_r, cos, sin, position_ids)
            q_rot = torch.cat((q_r, q_p), dim=-1)
            k_rot = torch.cat((k_r, k_p), dim=-1)
            del q, k, v, q_r, q_p, k_r, k_p

            # Pre-softmax scores: (B, H, L, L)
            scores = torch.matmul(q_rot, k_rot.transpose(-1, -2)) * (head_size ** -0.5)
            del q_rot, k_rot

            # Std over valid keys per query position (same approach as eval_attn_logit_std.py)
            scores_valid = scores.masked_fill(~causal_mask, 0.0)
            mean_q = scores_valid.sum(dim=-1) / valid_counts  # (B, H, L)
            diff_sq = ((scores - mean_q.unsqueeze(-1)) ** 2).masked_fill(~causal_mask, 0.0)
            del scores_valid, mean_q
            sum_diff_sq = diff_sq.sum(dim=-1)  # (B, H, L)
            del diff_sq
            # unbiased std for q=1..L-1 (ddof=1, n_valid >= 2)
            std_q = (sum_diff_sq[:, :, 1:] / (n_valid - 1)).clamp(min=0.0).sqrt()  # (B, H, L-1)
            del sum_diff_sq

            # Top-2 scores over valid keys per query position
            scores_neginf = scores.masked_fill(~causal_mask, float("-inf"))
            del scores
            top2_attn = scores_neginf.topk(2, dim=-1).values  # (B, H, L, 2)
            del scores_neginf
            top2_attn = top2_attn[:, :, 1:, :]  # skip q=0 → (B, H, L-1, 2)

            gap_attn = (top2_attn[..., 0] - top2_attn[..., 1]) / std_q.clamp(min=1e-8)
            del top2_attn, std_q
            # (B, H, L-1) → accumulate per head

            for h in range(num_heads):
                gap_np = gap_attn[:, h, :].reshape(-1).cpu().numpy()
                counts, _ = np.histogram(gap_np, bins=bin_edges)
                attn_hist[i, h] += counts
                attn_overflow[i, h] += np.int64((gap_np > gap_max).sum())

            del gap_attn

        del qkv_outputs

        if (batch_idx + 1) % 10 == 0:
            print(f"  [{batch_idx + 1}/{len(loader)}] batches done", flush=True)

    return attn_hist, attn_overflow, lm_hist, lm_overflow


def main():
    parser = argparse.ArgumentParser(
        description="Compute energy gap histograms for attention heads and LM head."
    )
    parser.add_argument("--model", required=True,
                        help="HuggingFace model ID or local save_pretrained path")
    parser.add_argument("--revisions", nargs="+", type=str, default=["none"],
                        help="HF revisions (e.g. step128 step256). Use 'none' for no revision.")
    parser.add_argument("--data_path", type=str, required=True)

    data_mode = parser.add_mutually_exclusive_group()
    data_mode.add_argument("--clean_eval", action="store_true")
    data_mode.add_argument("--eval_shard_offset", type=int, default=None)

    parser.add_argument("--train_steps", type=int, default=143000)
    parser.add_argument("--train_world_size", type=int, default=8)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--sampler_seed", type=int, default=0)

    parser.add_argument("--n_eval_tokens", type=int, default=2_000_000)
    parser.add_argument("--context_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--gap_max", type=float, default=20.0,
                        help="Upper edge of histogram; values above are counted as overflow")
    parser.add_argument("--n_bins", type=int, default=200)
    parser.add_argument("--out_dir", type=str, default="results/energy_gap")
    args = parser.parse_args()

    if not args.clean_eval and args.eval_shard_offset is None:
        args.clean_eval = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_eval_chunks = args.n_eval_tokens // args.context_length
    bin_edges = np.linspace(0.0, args.gap_max, args.n_bins + 1)

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

    os.makedirs(args.out_dir, exist_ok=True)

    for rev in args.revisions:
        revision = None if rev == "none" else rev
        rev_key = revision if revision is not None else "final"
        print(f"Loading: {args.model} (revision={revision})")
        t0 = time.time()
        model = load_model(args.model, device, revision=revision)

        attn_hist, attn_overflow, lm_hist, lm_overflow = compute_energy_gap_hists(
            model, loader, device, bin_edges
        )
        elapsed = time.time() - t0

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
        lm_total = int(lm_hist.sum()) + int(lm_overflow)
        meta = {
            "model": args.model,
            "revision": revision,
            "n_eval_tokens": args.n_eval_tokens,
            "gap_max": args.gap_max,
            "n_bins": args.n_bins,
            "elapsed_s": round(elapsed, 1),
            "attn_hist_shape": list(attn_hist.shape),
            "attn_overflow_frac": round(float(attn_overflow.sum()) / attn_total, 5) if attn_total > 0 else 0.0,
            "lm_overflow_frac": round(float(lm_overflow) / lm_total, 5) if lm_total > 0 else 0.0,
        }
        with open(os.path.join(args.out_dir, f"{rev_key}_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        n_layers, n_heads, _ = attn_hist.shape
        print(f"  attn_hist shape: ({n_layers}, {n_heads}, {args.n_bins})")
        print(f"  attn overflow: {int(attn_overflow.sum())} ({100 * meta['attn_overflow_frac']:.2f}%)")
        print(f"  lm  overflow: {int(lm_overflow)} ({100 * meta['lm_overflow_frac']:.2f}%)")
        print(f"  saved to {out_npz}")

        del model
        torch.cuda.empty_cache()

    print(f"\nResults written to {args.out_dir}/")


if __name__ == "__main__":
    main()
