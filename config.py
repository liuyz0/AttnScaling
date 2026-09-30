from dataclasses import dataclass, field
from typing import Literal


@dataclass
class TrainConfig:
    # Model
    student_model: str = "EleutherAI/pythia-160m"
    teacher_model: str | None = None  # None = CE-only run, set to e.g. "EleutherAI/pythia-160m" for distillation

    # Loss
    loss_type: Literal["ce", "kl", "mse"] = "ce"
    teacher_temperature: float = 1.0  # applied to teacher logits before softmax

    # Data
    data_source: Literal["random", "local"] = "random"
    local_data_path: str = "/path/to/pile_deduped_shards/"  # directory containing .bin shard files
    context_length: int = 2048

    # Training
    batch_size: int = 16          # sequences per GPU
    gradient_accumulation_steps: int = 1
    max_steps: int = 1000         # short for testing; Pythia full = 143_000
    learning_rate: float = 1e-3   # Pythia-160m default
    weight_decay: float = 0.1
    lm_head_wd: float | None = None  # if set, overrides weight_decay for lm_head weight
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # LR schedule (cosine with warmup, same as Pythia)
    warmup_steps: int = 100
    lr_decay_steps: int | None = None  # defaults to max_steps

    # Logging / checkpointing
    log_interval: int = 10
    eval_interval: int = 100
    save_steps: str = "128,256,512,1000,2000,3000,6000,12000,16000,32000,48000,79000"
    out_dir: str = "checkpoints"
    run_name: str = ""      # used as wandb run name; empty = wandb auto-generates
    use_wandb: bool = False

    # System
    dtype: Literal["float32", "bfloat16", "float16"] = "float16"
    seed: int = 42
