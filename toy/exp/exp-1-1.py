# scan G-I model with MSE
# from test-1.ipynb
# 8 beta_G, 8 lr
# comparing to exp-1, we use SGD here

import torch
import torch.nn.functional as F
import torch.nn as nn
from dataclasses import dataclass
import sys
import time

# ---- Hyperparameters ----
task_id = int(sys.argv[1]) # from 0 to 63
num_tasks = int(sys.argv[2]) # should be 64
assert num_tasks == 64, "num_tasks should be 64"

lr_id, beta_G_id = task_id % 8, task_id // 8

lrs = [0.3, 1, 3, 10, 15, 25, 33, 100]
lr = lrs[lr_id]

beta_Gs = torch.logspace(0, 3, 8)
beta_G = beta_Gs[beta_G_id].item()

batch_size = 2048
test_num_batch = 4
num_steps = 10000
log_interval = 10

# ---- Config ----
@dataclass
class Config:
    vocab_size: int = 128     # vocab size (n)
    n_embd: int = 32         # model width (m)
    beta: float = 1.0              # scaling factor for the logits
    beta_g: float = 1.0            # scaling factor for the gating
    teacher: bool = False             # whether to use teacher model
    d_gate: int = 8             # number of classes in gate

# ---- Utils ----
def rmsnorm(x: torch.Tensor):
    return F.rms_norm(x, (x.size(-1),))

# ---- Model ----
class GATE_I(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        if cfg.teacher:
            self.K = torch.nn.Parameter(torch.randn(cfg.d_gate, cfg.n_embd) * cfg.beta_g / (cfg.n_embd ** 0.5))
            self.V = torch.nn.Parameter(torch.randn(cfg.d_gate, cfg.n_embd, cfg.n_embd) / (cfg.n_embd ** 0.5))
        else:
            self.K = torch.nn.Parameter(torch.zeros(cfg.d_gate, cfg.n_embd))
            self.V = torch.nn.Parameter(torch.randn(cfg.d_gate, cfg.n_embd, cfg.n_embd) / (cfg.n_embd ** 0.5) / 4)
            #self.V = torch.nn.Parameter(torch.zeros(cfg.d_gate, cfg.n_embd, cfg.n_embd))

    def forward(self, x: torch.Tensor):
        # x: (batch_size, n_embd)
        # Compute gate logits
        gate_logits = F.linear(x, self.K)  # (batch_size, d_gate)
        gate_probs = F.softmax(gate_logits, dim=-1)  # (batch_size, d_gate)
        # Compute value
        Values = torch.einsum('bm,dam->bda', x, self.V)  # (batch_size, d_gate, n_embd)
        # Compute output as weighted sum of values
        output = torch.einsum('bd,bdm->bm', gate_probs, Values)
        return output
    
t_cfg = Config(beta = 1.0, beta_g = beta_G, teacher = True)
teacher = GATE_I(t_cfg)
s_cfg = Config()
student = GATE_I(s_cfg)
init_V = student.V.detach().clone()

optimizer = torch.optim.SGD(student.parameters(), lr=lr)

train_losses = torch.zeros(num_steps)
test_losses = torch.zeros(num_steps // log_interval)
s_beta_G = torch.zeros(num_steps // log_interval)
V_diff = torch.zeros(num_steps // log_interval)

start_time = time.time()
for step in range(num_steps):
    x = torch.randn(batch_size, s_cfg.n_embd)
    with torch.no_grad():
        teacher_output = teacher(x)
    student_output = student(x)
    loss = F.mse_loss(student_output, teacher_output)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_losses[step] = loss.item()

    if (step + 1) % log_interval == 0:
        with torch.no_grad():
            test_loss = 0.0
            for _ in range(test_num_batch):
                test_x = torch.randn(batch_size, s_cfg.n_embd)
                t_out = teacher(test_x)  # (B, vocab_size)
                s_out = student(test_x)  # (B, vocab_size)
                test_loss += F.mse_loss(s_out, t_out).item()
            test_loss /= test_num_batch
            test_losses[step // log_interval] = test_loss
            s_beta_G[step // log_interval] = student.K.norm().item() / (s_cfg.d_gate ** 0.5)
            V_diff[step // log_interval] = (student.V - init_V).norm().item()

print(f'Run time {time.time() - start_time:.2f} seconds for {num_steps} steps.')
torch.save({
    'train_losses': train_losses,
    'test_losses': test_losses,
    's_beta_G': s_beta_G,
    'V_diff': V_diff},
    f'/orcd/compute/gore/001/DistillScaling/toy/outputs/exp-1-1-{task_id}.pt')