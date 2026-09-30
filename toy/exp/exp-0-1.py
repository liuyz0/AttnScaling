# scan GH-I model
# architecture from test-1-4.ipynb
# 8 beta_H, 8 beta_G, 3 lr, 64x3 = 192 CPUs
# comparing to exp-0, we use SGD here
# lr 3, 10, 30

import torch
import torch.nn.functional as F
import torch.nn as nn
from dataclasses import dataclass
import sys
import time

# ---- Hyperparameters ----
task_id = int(sys.argv[1]) # from 0 to 191
num_tasks = int(sys.argv[2]) # should be 192
assert num_tasks == 192, "num_tasks should be 192"

lr_id, beta_H_id, beta_G_id = task_id % 3, (task_id // 3) % 8, (task_id // 3) // 8

lrs = [3.0, 10.0, 30.0]
lr = lrs[lr_id]

beta_Hs = torch.logspace(0, 3, 8)
beta_H = beta_Hs[beta_H_id].item()

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
    
class GH_I(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.gate = GATE_I(cfg)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.teacher:
            self.head.weight.data.copy_(torch.randn(cfg.vocab_size, cfg.n_embd) / (cfg.n_embd ** 0.5))
        else:
            self.head.weight.data.zero_()
            
    def forward(self, x: torch.Tensor):
        # x: (batch_size, n_embd)
        x = rmsnorm(x)
        x = self.gate(x) + x  # residual connection
        x = rmsnorm(x)
        logits = self.head(x) * self.cfg.beta  # scale logits
        return logits
    
t_cfg = Config(beta = beta_H, beta_g = beta_G, teacher = True)
teacher = GH_I(t_cfg)
s_cfg = Config()
student = GH_I(s_cfg)
init_V = student.gate.V.detach().clone()

optimizer = torch.optim.SGD(student.parameters(), lr=lr)

train_losses = torch.zeros(num_steps)
test_losses = torch.zeros(num_steps // log_interval)
s_beta_H = torch.zeros(num_steps // log_interval)
s_beta_G = torch.zeros(num_steps // log_interval)
V_diff = torch.zeros(num_steps // log_interval)

start_time = time.time()
for step in range(num_steps):
    x = torch.randn(batch_size, s_cfg.n_embd)
    with torch.no_grad():
        teacher_logits = teacher(x)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
    student_logits = student(x)
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    loss = F.kl_div(student_log_probs, teacher_log_probs, reduction='batchmean', log_target=True)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_losses[step] = loss.item()

    if (step + 1) % log_interval == 0:
        with torch.no_grad():
            test_loss = 0.0
            for _ in range(test_num_batch):
                test_x = torch.randn(batch_size, s_cfg.n_embd)
                t_logits = teacher(test_x)  # (B, vocab_size)
                teacher_log_probs = F.log_softmax(t_logits, dim=-1)  # (B, vocab_size)
                s_logits = student(test_x)  # (B, vocab_size)
                student_log_probs = F.log_softmax(s_logits, dim=-1)  # (B, vocab_size)
                test_loss += F.kl_div(student_log_probs, teacher_log_probs, reduction='batchmean', log_target=True).item()
            test_loss /= test_num_batch
            test_losses[step // log_interval] = test_loss
            s_beta_H[step // log_interval] = student.head.weight.norm().item() / (s_cfg.vocab_size ** 0.5)
            s_beta_G[step // log_interval] = student.gate.K.norm().item() / (s_cfg.d_gate ** 0.5)
            V_diff[step // log_interval] = (student.gate.V - init_V).norm().item()

print(f'Run time {time.time() - start_time:.2f} seconds for {num_steps} steps.')
torch.save({
    'train_losses': train_losses,
    'test_losses': test_losses,
    's_beta_H': s_beta_H,
    's_beta_G': s_beta_G,
    'V_diff': V_diff},
    f'/orcd/compute/gore/001/DistillScaling/toy/outputs/exp-0-1-{task_id}.pt')