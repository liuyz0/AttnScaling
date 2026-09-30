"""
Compute logit std and entropy for a single model checkpoint over an eval dataset.

For each temperature T:
  logit_std(T) = mean_tokens[ std(logits/T, dim=vocab) ]
               = mean_tokens[ std(logits, dim=vocab) ] / T
  entropy(T)   = mean_tokens[ H(softmax(logits/T)) ]

Since logit_std scales as 1/T, one forward pass suffices for all temperatures.
Entropy is nonlinear in T but shares the same logits, so it is also one pass.

Usage:
    # Student checkpoint (T=1)
    python eval_stats.py \\
        --model checkpoints/kl_001/step_0079000 \\
        --temperatures 1.0 \\
        --data_path data/pythia_deduped_pile_idxmaps/ \\
        --clean_eval \\
        --out_file results/eval_stats.jsonl

    # Teacher (4 temperatures, one forward pass)
    python eval_stats.py \\
        --model EleutherAI/pythia-160m --revision step79000 \\
        --temperatures 0.01 0.1 1.0 10.0 \\
        --data_path data/pythia_deduped_pile_idxmaps/ \\
        --clean_eval \\
        --out_file results/eval_stats.jsonl --append
"""

import argparse
import json
import os
import time

import torch
from torch.utils.data import DataLoader, Subset

from data import PileMemmap
from eval import EvalPileMemmap, get_clean_eval_indices, load_model


@torch.no_grad()
def compute_stats(
    model,
    loader: DataLoader,
    temperatures: list[float],
    device: torch.device,
    dtype: torch.dtype,
) -> list[dict]:
    """Returns one dict per temperature with logit_std and entropy."""
    raw_std_total = 0.0
    entropy_totals = {t: 0.0 for t in temperatures}
    n_batches = 0
    t0 = time.time()

    for batch_idx, (input_ids, _) in enumerate(loader):
        input_ids = input_ids.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=dtype):
            logits = model(input_ids=input_ids, use_cache=False).logits  # (B, L, V)

        logits = logits.float()
        raw_std_total += logits.std(dim=-1).mean().item()

        for temp in temperatures:
            entropy = torch.distributions.Categorical(
                logits=logits / temp
            ).entropy().mean().item()
            entropy_totals[temp] += entropy

        n_batches += 1
        if (batch_idx + 1) % 50 == 0:
            print(f"  [{batch_idx + 1}/{len(loader)}] {time.time() - t0:.0f}s", flush=True)

    avg_raw_std = raw_std_total / n_batches
    return [
        {
            "temperature": temp,
            "logit_std": round(avg_raw_std / temp, 6),
            "entropy": round(entropy_totals[temp] / n_batches, 6),
        }
        for temp in temperatures
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Compute logit std and entropy for a single model checkpoint."
    )
    parser.add_argument("--model", required=True,
                        help="HuggingFace model ID or local save_pretrained path")
    parser.add_argument("--revision", type=str, default=None,
                        help="HF revision (e.g. step79000)")
    parser.add_argument("--temperatures", nargs="+", type=float, default=[1.0])
    parser.add_argument("--data_path", type=str, required=True)

    data_mode = parser.add_mutually_exclusive_group()
    data_mode.add_argument("--clean_eval", action="store_true")
    data_mode.add_argument("--eval_shard_offset", type=int, default=None)

    parser.add_argument("--train_steps", type=int, default=79000)
    parser.add_argument("--train_world_size", type=int, default=8)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--sampler_seed", type=int, default=0)

    parser.add_argument("--n_eval_tokens", type=int, default=5_000_000)
    parser.add_argument("--context_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"],
                        default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--out_file", type=str, default="eval_stats.jsonl")
    parser.add_argument("--append", action="store_true",
                        help="Append to out_file instead of overwriting")
    args = parser.parse_args()

    if not args.clean_eval and args.eval_shard_offset is None:
        args.clean_eval = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
             "float16": torch.float16}[args.dtype]
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
        eval_mode_tag = "clean_eval"
        print(f"Using {len(dataset)} clean chunks ({len(dataset) * args.context_length / 1e6:.1f}M tokens)")
    else:
        dataset = EvalPileMemmap(
            args.data_path,
            context_length=args.context_length,
            shard_offset=args.eval_shard_offset,
            max_tokens=args.n_eval_tokens,
        )
        eval_mode_tag = f"shard_offset_{args.eval_shard_offset}"

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    print(f"Batches: {len(loader)}\n")

    print(f"Loading: {args.model} (revision={args.revision})")
    model = load_model(args.model, dtype, device, revision=args.revision)

    t0 = time.time()
    results = compute_stats(model, loader, args.temperatures, device, dtype)
    elapsed = time.time() - t0

    os.makedirs(os.path.dirname(os.path.abspath(args.out_file)), exist_ok=True)
    with open(args.out_file, "a" if args.append else "w") as f:
        for r in results:
            row = {
                "model": args.model,
                "revision": args.revision,
                **r,
                "n_eval_tokens": len(dataset) * args.context_length,
                "eval_mode": eval_mode_tag,
                "elapsed_s": round(elapsed, 1),
            }
            print(f"  T={r['temperature']}: logit_std={r['logit_std']:.4f}  entropy={r['entropy']:.4f}")
            f.write(json.dumps(row) + "\n")

    print(f"\nResults written to {args.out_file}")


if __name__ == "__main__":
    main()
