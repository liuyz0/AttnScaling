"""
Training script. Run with:
    python train.py [--arg value ...]              # single GPU
    torchrun --nproc_per_node=N train.py [...]    # multi-GPU DDP

Every field in TrainConfig is exposed as a CLI argument automatically.
"""

import contextlib
import dataclasses
import json
import math
import os
import time
import types
import typing
import argparse

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoConfig, AutoModelForCausalLM

from config import TrainConfig
from data import make_dataset
from loss import compute_loss, kl_loss


# ---------------------------------------------------------------------------
# Argparse: auto-generated from TrainConfig fields
# ---------------------------------------------------------------------------

def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Distillation training")
    hints = typing.get_type_hints(TrainConfig)

    for f in dataclasses.fields(TrainConfig):
        hint = hints[f.name]
        origin = typing.get_origin(hint)
        args = typing.get_args(hint)

        if origin is typing.Literal:
            parser.add_argument(f"--{f.name}", type=type(args[0]), default=f.default, choices=list(args))
        elif origin in (types.UnionType, typing.Union):
            inner = next(a for a in args if a is not type(None))
            def _nullable(t):
                def _parse(v):
                    return None if v.lower() == "none" else t(v)
                _parse.__name__ = f"nullable_{t.__name__}"
                return _parse
            parser.add_argument(f"--{f.name}", type=_nullable(inner), default=f.default)
        elif hint is bool:
            # bool("False") == True in argparse, so use store_true flag instead.
            parser.add_argument(f"--{f.name}", action="store_true", default=f.default)
        else:
            parser.add_argument(f"--{f.name}", type=hint, default=f.default)

    return parser


# ---------------------------------------------------------------------------
# LR schedule: linear warmup + cosine decay
# ---------------------------------------------------------------------------

def get_lr(step: int, config: TrainConfig) -> float:
    decay_steps = config.lr_decay_steps or config.max_steps
    if step < config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    if step > decay_steps:
        return config.learning_rate * 0.1
    progress = (step - config.warmup_steps) / (decay_steps - config.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.learning_rate * 0.1 + coeff * config.learning_rate * 0.9


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config: TrainConfig):
    # ---- DDP setup ----
    # torchrun sets LOCAL_RANK; absent means single-GPU mode.
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_ddp = local_rank != -1
    if is_ddp:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_main = (rank == 0)

    # Same model init seed on all ranks; DDP syncs gradients so weights stay in sync.
    torch.manual_seed(config.seed)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[config.dtype]

    save_steps = {int(s) for s in config.save_steps.split(",") if s.strip()}

    if is_main:
        os.makedirs(config.out_dir, exist_ok=True)
        metrics_path = os.path.join(config.out_dir, "metrics.jsonl")
        if config.use_wandb:
            import wandb
            wandb.init(
                project="distill-scaling",
                name=config.run_name or None,
                config=dataclasses.asdict(config),
            )

    # ---- student (randomly initialized) ----
    if is_main:
        print(f"Initializing student from config: {config.student_model}")
    student_cfg = AutoConfig.from_pretrained(config.student_model)
    student_cfg.use_cache = False
    student = AutoModelForCausalLM.from_config(
        student_cfg, attn_implementation="flash_attention_2",
        torch_dtype=dtype 
    ).to(device)
    student.train()

    if is_ddp:
        student = torch.nn.parallel.DistributedDataParallel(student, device_ids=[local_rank])
    # raw_student: unwrapped handle used for optimizer param groups and saving.
    raw_student = student.module if is_ddp else student

    # ---- teacher (optional, frozen — no DDP needed: no backward, no gradient sync) ----
    teacher = None
    if config.teacher_model is not None:
        if is_main:
            print(f"Loading teacher: {config.teacher_model}")
        teacher = AutoModelForCausalLM.from_pretrained(
            config.teacher_model, torch_dtype=dtype, attn_implementation="flash_attention_2",
            cache_dir="./cache", revision="step79000"
        ).to(device)
        teacher.config.use_cache = False
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

    # ---- data ----
    dataset = make_dataset(config)
    sampler = None
    data_restarts = 0  # used as epoch counter to reshuffle DistributedSampler

    if config.data_source == "local":
        # DistributedSampler assigns non-overlapping index subsets to each rank.
        # drop_last=True in the DataLoader ensures all ranks always have the same
        # number of batches per pass, preventing deadlock on the DDP all-reduce.
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            pin_memory=True,
            # PERFORMANCE: num_workers controls prefetch parallelism. Each worker
            # independently reads from the memmap file while the GPU is computing.
            # Start with 4; if you see GPU utilization dips between steps (check
            # nvidia-smi or torch profiler), increase to 8. Diminishing returns
            # beyond ~8 workers on NVMe. On spinning HDD, shuffled access may be
            # the real bottleneck — consider shuffle=False or pre-shuffling shards.
            num_workers=4,
            # Keeps workers alive across iterator restarts (avoids re-fork overhead).
            persistent_workers=True,
            # Each worker prefetches this many batches ahead. 2 is usually enough
            # to fully hide I/O latency with num_workers=4. Increase if profiling
            # shows the DataLoader is still the bottleneck.
            prefetch_factor=2,
            drop_last=True,
        )
    else:
        # RandomTokenDataset is pure in-memory; num_workers=0 is fine.
        # Give each rank a different seed so they generate independent token streams.
        torch.manual_seed(config.seed + rank)
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            pin_memory=True,
            num_workers=0,
        )

    # ---- optimizer (biases and LayerNorm params get no weight decay, matching GPT-NeoX) ----
    no_decay_types = (torch.nn.LayerNorm,)
    lm_head_param_ids = (
        {id(p) for p in raw_student.get_output_embeddings().parameters()}
        if config.lm_head_wd is not None else set()
    )
    decay_params, no_decay_params, lm_head_params = [], [], []
    for _, module in raw_student.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            if isinstance(module, no_decay_types) or param_name == "bias":
                no_decay_params.append(param)
            elif id(param) in lm_head_param_ids:
                lm_head_params.append(param)
            else:
                decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    if lm_head_params:
        param_groups.append({"params": lm_head_params, "weight_decay": config.lm_head_wd})

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        fused=(device.type == "cuda"),
    )

    # ---- training loop ----
    step = 0
    running_loss = torch.zeros((), device=device)
    running_kl = torch.zeros((), device=device)
    t0 = time.time()
    data_iter = iter(loader)

    while step < config.max_steps:
        lr = get_lr(step, config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)

        for micro_step in range(config.gradient_accumulation_steps):
            try:
                input_ids, labels = next(data_iter)
            except StopIteration:
                data_restarts += 1
                if sampler is not None:
                    # Change the shuffle seed so the new pass has a different order.
                    sampler.set_epoch(data_restarts)
                data_iter = iter(loader)
                input_ids, labels = next(data_iter)

            # non_blocking=True lets the CPU keep preparing the next batch while
            # this transfer runs, which is only effective with pin_memory=True.
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # no_sync() suppresses the DDP all-reduce on intermediate micro-steps,
            # accumulating gradients locally. The all-reduce fires only on the last
            # micro-step when the context is not active.
            is_last_micro = (micro_step == config.gradient_accumulation_steps - 1)
            sync_ctx = contextlib.nullcontext() if (not is_ddp or is_last_micro) else student.no_sync()

            with sync_ctx:
                with torch.autocast(device_type=device.type, dtype=dtype):
                    student_logits = student(input_ids=input_ids, use_cache=False).logits  # (B, T, vocab)

                    teacher_logits = None
                    if teacher is not None:
                        with torch.no_grad():
                            teacher_logits = teacher(input_ids=input_ids, use_cache=False).logits

                    loss = compute_loss(
                        student_logits,
                        labels,
                        config.loss_type,
                        teacher_logits=teacher_logits,
                        teacher_temperature=config.teacher_temperature,
                    ) / config.gradient_accumulation_steps

                loss.backward()

            if is_main:
                running_loss += loss.detach()

        torch.nn.utils.clip_grad_norm_(student.parameters(), config.grad_clip)
        optimizer.step()
        step += 1

        # ---- logging (rank 0 only) ----
        if is_main and step % config.log_interval == 0:
            if teacher_logits is not None and config.loss_type == "mse":
                with torch.no_grad():
                    running_kl = kl_loss(
                        student_logits.detach(), teacher_logits.detach(), temperature=config.teacher_temperature
                    )
            elapsed = time.time() - t0
            avg_loss = (running_loss / config.log_interval).item()
            row = {"step": step, "loss": avg_loss, "lr": lr, "elapsed_s": round(elapsed, 2)}
            log = f"step {step:6d} | loss {avg_loss:.4f} | lr {lr:.2e} | {elapsed:.1f}s"
            if teacher is not None and config.loss_type == "mse":
                avg_kl = running_kl.item()
                row["eval_kl"] = avg_kl
                log += f" | eval_kl {avg_kl:.4f}"
            print(log, flush=True)
            with open(metrics_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            if config.use_wandb:
                wandb.log(row, step=step)
            running_loss.zero_()
            running_kl.zero_()
            t0 = time.time()

        # ---- checkpoint (rank 0 only) ----
        if is_main and step in save_steps:
            ckpt_path = os.path.join(config.out_dir, f"step_{step:07d}")
            raw_student.save_pretrained(ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}", flush=True)

    if is_main:
        print("Training complete.")
        if config.use_wandb:
            wandb.finish()

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    config = TrainConfig(**vars(args))
    main(config)
