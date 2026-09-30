# Emergent One-Third Scaling Law as Attention Tries to Concentrate

Code for the paper **"Emergent One-Third Scaling Law as Attention Tries to Concentrate"** by Yizhou Liu, Sara Kangaslahti, and Jeff Gore.

> We show that any softmax function learning peaked distributions develops logit magnitude growing as a power law with exponent $1/3$ and contributes a loss term decaying as a power law with exponent $1/3$. In LLMs, attention heads—not the LM head—may be the bottleneck driving this 1/3 loss scaling.

**Paper link**: https://arxiv.org/abs/2609.32100

![Video abstract](figures/attention_draft.gif)

---

## Repository Structure

```
AttnScaling/
├── toy/
│   ├── exp/           # toy model training scripts (cluster jobs)
│   ├── analyze/       # analysis notebooks for toy model experiments
│   └── tests/         # auxiliary exploration notebooks
├── analyze/           # analysis notebooks for LLM experiments
├── olmo/              # OLMo-2-specific evaluation scripts
├── runs/              # SLURM job scripts
├── train.py           # distillation training (MSE experiment, Appendix C.2)
├── loss.py            # CE / KL / MSE loss functions
├── config.py          # TrainConfig dataclass
├── data.py            # dataset utilities
├── eval.py            # general evaluation helper
├── eval_attn_logit_std.py   # Pythia: per-head attention logit std & entropy
├── eval_energy_gap.py       # Pythia: energy gap distribution
├── eval_bos_norms.py        # BOS token norm evaluation
└── eval_stats.py            # general statistics evaluation
```

> **Note:** Scripts for computing Pythia loss and LM-head logit coldness (`pythia-logit-1.py`) and the analogous OLMo-2 scripts (`olmo-logit-{0,1,2}.py`, `olmo-step-tau.py`) are maintained in a companion repository: https://github.com/liuyz0/TimeScaling. In those filenames, the suffix is a model-size index (0 = 1B, 1 = 7B, 2 = 13B).

---

## Part 1: Toy Model Experiments (Sections 2–3, Appendix A)

The toy experiments study gate-head (GH) models, which abstract the attention + LM-head structure of LLMs. Three variants are studied:

| Model | Gate architecture | Key result |
|-------|------------------|------------|
| **GH-I** (main) | Linear softmax gate | 1/3 scaling whenever gate or head is cold (Fig. 1b, 2, 3) |
| **GH-II** | Sigmoid / Tanh gate (GLU-style) | Same 1/3 scaling extends to non-softmax non-linearities (Fig. 9) |
| **GH-III** | Real self-attention gate | 1/3 scaling persists with full attention (Fig. 10) |

### Training scripts (`toy/exp/`)

These are intended to be run as SLURM arrays (see `runs/exp-*.sh`). Each script takes a task ID and total-task count as positional arguments.

| Script | Model | Optimizer | Figures |
|--------|-------|-----------|---------|
| `toy/exp/exp-0-1.py` | GH-I | SGD | 1b, 2 |
| `toy/exp/exp-1-1.py` | G-I (gate only) | SGD | 3 |
| `toy/exp/exp-0.py` | GH-I | Adam | 6, 7 |
| `toy/exp/exp-1.py` | G-I (gate only) | Adam | 8 |

Example (local, single task):
```bash
python toy/exp/exp-0-1.py 0 192
```

SLURM array submission:
```bash
sbatch runs/exp-0-1.sh   # GH-I SGD, 192 jobs
sbatch runs/exp-1-1.sh   # G-I SGD, 64 jobs
sbatch runs/exp-0.sh     # GH-I Adam, 192 jobs
sbatch runs/exp-1.sh     # G-I Adam, 64 jobs
```

### Analysis notebooks (`toy/analyze/`)

Each notebook loads the `.pt` result files saved by the training scripts and generates the figures.

| Notebook | Experiment | Figures |
|----------|-----------|---------|
| `toy/analyze/exp-0-1.ipynb` | GH-I SGD parameter scan | 1b, 2 |
| `toy/analyze/exp-1-1.ipynb` | G-I SGD parameter scan | 3 |
| `toy/analyze/exp-0.ipynb` | GH-I Adam | 6, 7 |
| `toy/analyze/exp-1.ipynb` | G-I Adam | 8 |

### Auxiliary notebooks (`toy/tests/`)

These notebooks explore model variants referenced in Appendix A.

| Notebook | Description | Figure |
|----------|-------------|--------|
| `toy/tests/test-2-1.ipynb` | GH-II with Sigmoid gate | 9 (panels a–b) |
| `toy/tests/test-2-2.ipynb` | GH-II with Tanh gate | 9 (panels c–d) |
| `toy/tests/test-3-3.ipynb` | GH-III with real attention gate | 10 |

---

## Part 2: LLM Experiments on Pythia (Section 4, Appendix C.1)

We evaluate the Pythia model family (70M–12B) from EleutherAI, loaded from Hugging Face at logarithmically spaced training checkpoints, using text from the Pile (deduped).

### Step 1: Compute loss and LM-head logit coldness

Use `pythia-logit-1.py` from https://github.com/liuyz0/TimeScaling. This produces per-checkpoint cross-entropy loss and LM-head logit standard deviation.

### Step 2: Compute attention logit std and entropy

```bash
python eval_attn_logit_std.py \
    --model EleutherAI/pythia-12b \
    --revisions step128 step256 step512 step1000 step2000 step3000 \
               step6000 step12000 step16000 step32000 step48000 step79000 \
    --data_path data/pythia_deduped_pile_idxmaps/ \
    --clean_eval --n_eval_tokens 200000 \
    --out_file results/eval_attn_logit_std_12b.jsonl
```

Pre-configured SLURM scripts are in `runs/run_eval_attn_logit_std*.sh` for each model size.

### Step 3: Compute energy gap distributions

```bash
python eval_energy_gap.py \
    --model EleutherAI/pythia-12b \
    --revisions step128 step256 ... step79000 \
    --data_path data/pythia_deduped_pile_idxmaps/ \
    --clean_eval --n_eval_tokens 2000000 \
    --out_dir results/energy_gap/12b
```

Pre-configured SLURM scripts are in `runs/run_eval_energy_gap*.sh` for each model size.

### Step 4: Analysis notebooks

| Notebook | Model | Content | Figures |
|----------|-------|---------|---------|
| `analyze/pythia-logit-0to7-v3.ipynb` | All 8 sizes (70M–12B) | Loss fitting L(τ) = c τ^{−α} + L^∞; collapse after subtracting L^∞ | 4c, 15 |
| `analyze/attn_logit_std_12b.ipynb` | Pythia-12B | Per-head attention logit std vs. τ; all 36 layers | 4d, 13 |
| `analyze/attn_logit_std_410m.ipynb` | Pythia-410M | Per-head entropy (final ckpt); attention logit std vs. τ | 11a, 14 (panels a–b) |
| `analyze/attn_logit_std.ipynb` | Pythia-160M | Per-head entropy (final ckpt); attention logit std vs. τ | 11b, 14 (panels c–d) |
| `analyze/energy_gap_12b.ipynb` | Pythia-12B | Energy gap density for all 36 layers | 4b, 12 |
| `analyze/energy_gap.ipynb` | Pythia-12B and 410M | Energy gap density | 4b, 14 (panel a) |
| `analyze/energy_gap_160m.ipynb` | Pythia-160M | Energy gap density | 14 (panel c) |

> **Note on Figure 4a:** The per-head entropy scatter plot (Pythia-12B, final checkpoint) and the entropy thresholds are produced inside `analyze/attn_logit_std_12b.ipynb`.

---

## Part 3: LLM Experiments on OLMo-2 (Appendix C.1)

We evaluate OLMo-2 (1B, 7B, 13B) from AllenAI, using text from FineWeb.

### Step 1: Compute loss and LM-head logit coldness

Use `olmo-logit-{0,1,2}.py` and `olmo-step-tau.py` from https://github.com/liuyz0/TimeScaling (suffixes 0 = 1B, 1 = 7B, 2 = 13B).

### Step 2: Compute attention logit std and entropy

```bash
python olmo/eval_attn_logit_std.py \
    --model allenai/OLMo-2-1124-13B \
    --revisions stage1-step1100-tokens10B ... stage1-step596057-tokens5001B \
    --n_eval_tokens 2000000 \
    --out_file results/olmo2_13b_attn_logit_std.jsonl
```

Pre-configured SLURM scripts: `runs/run_eval_attn_logit_std_olmo2_*.sh`.

### Step 3: Compute energy gap distributions

```bash
python olmo/eval_energy_gap.py \
    --model allenai/OLMo-2-1124-13B \
    --revisions stage1-step1100-tokens10B ... stage1-step596057-tokens5001B \
    --n_eval_tokens 2000000 \
    --out_dir results/energy_gap/olmo2_13b
```

Pre-configured SLURM scripts: `runs/run_eval_energy_gap_olmo2_*.sh`.

### Step 4: Analysis notebooks

| Notebook | Model | Content | Figure |
|----------|-------|---------|--------|
| `analyze/attn_logit_std_olo13b.ipynb` | OLMo-2-13B | Per-head entropy at final ckpt; logit std vs. τ for highlighted head | 16 (panels a, d) |
| `analyze/energy_gap_olmo13b.ipynb` | OLMo-2-13B | Energy gap density for highlighted head | 16 (panel b) |
| `analyze/attn_logit_std_olo7b.ipynb` | OLMo-2-7B | Per-head attention logit std vs. τ | — |
| `analyze/attn_logit_std_olo1b.ipynb` | OLMo-2-1B | Per-head attention logit std vs. τ | — |
| `analyze/energy_gap_olmo7b.ipynb` | OLMo-2-7B | Energy gap density | — |
| `analyze/energy_gap_olmo1b.ipynb` | OLMo-2-1B | Energy gap density | — |

> **Figure 16c** (loss vs. τ for OLMo-2-1B, 7B, 13B with fitted exponents) uses loss and LM-head logit data from `olmo-logit-{0,1,2}.py` in https://github.com/liuyz0/TimeScaling. The τ axis is computed by `olmo-step-tau.py` from the same repo.

---

## Part 4: MSE Distillation Experiment (Section 4, Appendix C.2)

To test whether the LM head is the scaling bottleneck, we train a Pythia-160M student using MSE between mean-centered logits instead of the standard cross-entropy or KL objective. The teacher is a frozen pretrained Pythia-160M.

### Training

```bash
# Single GPU
python train.py \
    --student_model EleutherAI/pythia-160m \
    --teacher_model EleutherAI/pythia-160m \
    --loss_type mse \
    --learning_rate 6e-4 \
    --max_steps 79000 \
    --warmup_steps 1430 \
    --lr_decay_steps 143000 \
    --batch_size 256 \
    --local_data_path /path/to/pile_deduped/ \
    --out_dir checkpoints/mse_160m

# Multi-GPU (8 GPUs)
torchrun --nproc_per_node=8 train.py [same args]
```

See `runs/run_full*.sh` for SLURM job scripts.

### Analysis notebooks

| Notebook | Content | Figure |
|----------|---------|--------|
| `analyze/attn_logit_std_mse1.ipynb` | Attention and LM-head logit coldness across checkpoints; identifies saturation of LM head | 5a |
| `analyze/mse_1.ipynb` | MSE loss fitting after LM-head saturation; L(τ) ≈ c τ^{−1/3} | 5b |

---

## Data

- **Pythia evaluations**: require the Pile (deduped) index maps, available via the Pythia repository instructions. Set `--data_path` to the directory containing `.bin` shard files.
- **OLMo-2 evaluations**: use the HuggingFace `HuggingFaceFW/fineweb` dataset, streamed automatically.
- **MSE training**: requires the Pile (deduped) shard files. Set `--local_data_path` in `config.py` or via CLI.

## Citation

If you use this code, please cite our paper:

```bibtex
@article{liu2026emergent,
  title={Emergent One-Third Scaling Law as Attention Tries to Concentrate},
  author={Liu, Yizhou and Kangaslahti, Sara and Gore, Jeff},
  journal={arXiv preprint arXiv:2609.32100},
  year={2026}
}
```
