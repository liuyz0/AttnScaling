"""
Evaluate one or more model checkpoints (HuggingFace IDs or local save_pretrained paths).

Eval data modes (mutually exclusive):
  --clean_eval           Exact: replay DistributedSampler(seed=0) to identify the ~86%
                         of chunks that training never touched, then eval on those.
                         Requires --train_steps, --train_world_size, --train_batch_size.
  --eval_shard_offset N  Fast fallback: use shards [N:] from the directory. ~13% of
                         those chunks were in the training pool (uniform contamination).

Usage:
    # CE only, clean eval
    python eval.py \\
        --models checkpoints/kl_001/step_0079000 EleutherAI/pythia-160m \\
        --loss_types ce \\
        --data_path ./data/pythia_deduped_pile_idxmaps/ \\
        --clean_eval \\
        --out_file results/eval.jsonl

    # CE + KL + MSE at multiple temperatures
    python eval.py \\
        --models checkpoints/kl_001/step_0079000 checkpoints/mse_1/step_0079000 \\
        --loss_types ce kl mse \\
        --teacher_model EleutherAI/pythia-160m --teacher_revision step79000 \\
        --teacher_temperatures 0.01 0.1 1.0 10.0 \\
        --data_path ./data/pythia_deduped_pile_idxmaps/ \\
        --clean_eval \\
        --out_file results/eval.jsonl
"""

import argparse
import glob
import json
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoModelForCausalLM

from data import PileMemmap
from loss import ce_loss, kl_loss, mse_loss


# ---------------------------------------------------------------------------
# Exact clean-eval index selection
# ---------------------------------------------------------------------------

def get_clean_eval_indices(
    total_chunks: int,
    world_size: int,
    batch_size: int,
    steps: int,
    sampler_seed: int = 0,
) -> torch.Tensor:
    """
    Replay DistributedSampler's permutation (default seed=0) and return the chunk
    indices that training never reached.

    DistributedSampler interleaves the global permutation across ranks:
      rank r sees: perm[r], perm[r+world_size], perm[r+2*world_size], ...
    Each rank consumed `steps * batch_size` chunks. Together that's
      n_consumed = world_size * batch_size * steps  positions in the permutation.
    Everything at permutation position >= n_consumed is untouched.
    """
    n_consumed = world_size * batch_size * steps
    assert n_consumed < total_chunks, (
        f"Training consumed {n_consumed} >= total chunks {total_chunks}; "
        "need to handle multi-epoch replay."
    )
    g = torch.Generator()
    g.manual_seed(sampler_seed)
    perm = torch.randperm(total_chunks, generator=g)
    clean = perm[n_consumed:]
    print(f"Total chunks: {total_chunks:,}  |  consumed: {n_consumed:,}  |  clean: {len(clean):,}")
    return clean


# ---------------------------------------------------------------------------
# Shard-offset eval dataset (simple fallback)
# ---------------------------------------------------------------------------

class EvalPileMemmap(Dataset):
    """
    Uses only shards starting at shard_offset, capped at max_tokens.
    ~13% contamination (uniform across shards); fine for relative comparisons.
    """

    def __init__(self, data_dir: str, context_length: int, shard_offset: int = 0,
                 max_tokens: int | None = None):
        bin_files = sorted(glob.glob(os.path.join(data_dir, "*.bin")))[shard_offset:]
        if not bin_files:
            raise FileNotFoundError(
                f"No .bin files in {data_dir!r} after shard_offset={shard_offset}"
            )

        self.context_length = context_length
        chunk = context_length + 1
        self.shards = [np.memmap(f, dtype=np.uint16, mode="r") for f in bin_files]
        self.shard_n_chunks = [len(s) // chunk for s in self.shards]
        self.cumulative = np.cumsum([0] + self.shard_n_chunks)
        total = int(self.cumulative[-1])

        self._len = min(total, max_tokens // context_length) if max_tokens else total
        print(f"Eval shards [{shard_offset}:]: {len(bin_files)} files, "
              f"{self._len} chunks ({self._len * context_length / 1e6:.1f}M tokens)")

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        shard_idx = int(np.searchsorted(self.cumulative[1:], idx, side="right"))
        local_idx = idx - int(self.cumulative[shard_idx])
        chunk = self.context_length + 1
        start = local_idx * chunk
        tokens = torch.from_numpy(
            self.shards[shard_idx][start : start + chunk].astype(np.int64)
        )
        return tokens[:-1], tokens[1:]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(path: str, dtype: torch.dtype, device: torch.device,
               revision: str | None = None) -> AutoModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=dtype,
        attn_implementation="flash_attention_2",
        revision=revision,
        cache_dir="./cache",
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_model(
    student,
    teacher,
    loader: DataLoader,
    loss_types: list[str],
    teacher_temperatures: list[float],
    device: torch.device,
    dtype: torch.dtype,
) -> list[dict]:
    """Returns one result dict per (loss_type, teacher_temperature) combination."""
    keys: list[tuple[str, float | None]] = []
    for lt in loss_types:
        if lt == "ce":
            keys.append(("ce", None))
        else:
            for t in teacher_temperatures:
                keys.append((lt, t))

    totals = {k: 0.0 for k in keys}
    n_batches = 0
    t0 = time.time()

    for batch_idx, (input_ids, labels) in enumerate(loader):
        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=dtype):
            student_logits = student(input_ids=input_ids, use_cache=False).logits
            teacher_logits = None
            if teacher is not None:
                teacher_logits = teacher(input_ids=input_ids, use_cache=False).logits

        for (lt, temp) in keys:
            if lt == "ce":
                totals[(lt, temp)] += ce_loss(student_logits, labels).item()
            elif lt == "kl":
                totals[(lt, temp)] += kl_loss(student_logits, teacher_logits, temp).item()
            elif lt == "mse":
                totals[(lt, temp)] += mse_loss(student_logits, teacher_logits, temp).item()

        n_batches += 1
        if (batch_idx + 1) % 50 == 0:
            print(f"  [{batch_idx + 1}/{len(loader)}] {time.time() - t0:.0f}s", flush=True)

    results = []
    for (lt, temp), total in totals.items():
        avg_loss = total / n_batches
        row: dict = {
            "loss_type": lt,
            "teacher_temperature": temp,
            "avg_loss": round(avg_loss, 6),
            "n_batches": n_batches,
        }
        if lt == "ce":
            row["perplexity"] = round(math.exp(avg_loss), 4)
        results.append(row)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate distillation checkpoints")
    parser.add_argument("--models", nargs="+", required=True,
                        help="HuggingFace model IDs or local save_pretrained paths")
    parser.add_argument("--loss_types", nargs="+", default=["ce"],
                        choices=["ce", "kl", "mse"])
    parser.add_argument("--teacher_model", type=str, default=None,
                        help="Required for kl/mse")
    parser.add_argument("--teacher_revision", type=str, default=None,
                        help="HF revision for teacher (e.g. step79000)")
    parser.add_argument("--teacher_temperatures", nargs="+", type=float, default=[1.0])
    parser.add_argument("--data_path", type=str, required=True,
                        help="Directory containing .bin shard files")

    # Eval data mode (mutually exclusive)
    data_mode = parser.add_mutually_exclusive_group()
    data_mode.add_argument("--clean_eval", action="store_true",
                           help="Use DistributedSampler replay to select truly unused chunks")
    data_mode.add_argument("--eval_shard_offset", type=int, default=None,
                           help="Use shards [N:] (fast fallback; ~13%% contamination)")

    # Clean-eval training parameters (needed to replay the sampler)
    parser.add_argument("--train_steps", type=int, default=79000)
    parser.add_argument("--train_world_size", type=int, default=8)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--sampler_seed", type=int, default=0,
                        help="DistributedSampler seed (default 0, PyTorch default)")

    parser.add_argument("--n_eval_tokens", type=int, default=5_000_000)
    parser.add_argument("--context_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"],
                        default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--out_file", type=str, default="eval_results.jsonl")
    parser.add_argument("--append", action="store_true",
                        help="Append to out_file instead of overwriting")
    args = parser.parse_args()

    # Default to clean_eval if neither mode is specified
    if not args.clean_eval and args.eval_shard_offset is None:
        args.clean_eval = True

    needs_teacher = any(lt in ("kl", "mse") for lt in args.loss_types)
    if needs_teacher and args.teacher_model is None:
        parser.error("--teacher_model is required for kl/mse loss types")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
             "float16": torch.float16}[args.dtype]
    n_eval_chunks = args.n_eval_tokens // args.context_length

    # Build eval dataset
    if args.clean_eval:
        full_dataset = PileMemmap(args.data_path, args.context_length)
        clean_indices = get_clean_eval_indices(
            total_chunks=len(full_dataset),
            world_size=args.train_world_size,
            batch_size=args.train_batch_size,
            steps=args.train_steps,
            sampler_seed=args.sampler_seed,
        )
        subset_indices = clean_indices[:n_eval_chunks].tolist()
        dataset = Subset(full_dataset, subset_indices)
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
    print(f"Batches per model: {len(loader)}\n")

    # Load teacher once
    teacher = None
    if needs_teacher:
        print(f"Loading teacher: {args.teacher_model} (revision={args.teacher_revision})")
        teacher = load_model(args.teacher_model, dtype, device,
                             revision=args.teacher_revision)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_file)), exist_ok=True)

    with open(args.out_file, "a" if args.append else "w") as out_f:
        for model_path in args.models:
            print(f"Evaluating: {model_path}")
            t0 = time.time()

            student = load_model(model_path, dtype, device)
            results = evaluate_model(
                student, teacher, loader,
                loss_types=args.loss_types,
                teacher_temperatures=args.teacher_temperatures,
                device=device,
                dtype=dtype,
            )
            elapsed = time.time() - t0

            for r in results:
                row = {
                    "model": model_path,
                    **r,
                    "n_eval_tokens": len(dataset) * args.context_length,
                    "eval_mode": eval_mode_tag,
                    "teacher_model": args.teacher_model,
                    "teacher_revision": args.teacher_revision,
                    "elapsed_s": round(elapsed, 1),
                }
                tag = f"{r['loss_type']} temp={r['teacher_temperature']}"
                ppl = f" ppl={r['perplexity']:.2f}" if "perplexity" in r else ""
                print(f"  {tag}: loss={r['avg_loss']:.4f}{ppl}")
                out_f.write(json.dumps(row) + "\n")
            out_f.flush()

            del student
            torch.cuda.empty_cache()

    print(f"\nResults written to {args.out_file}")


if __name__ == "__main__":
    main()
