"""
Loss functions for distillation experiments.

  ce  : standard cross-entropy on ground-truth labels (baseline)
  kl  : KL divergence between student and (temperature-scaled) teacher
  mse : mean squared error between student and teacher logits
"""

import torch
import torch.nn.functional as F


def ce_loss(student_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Standard language-model cross-entropy."""
    return F.cross_entropy(student_logits.view(-1, student_logits.size(-1)), labels.view(-1))


def kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """
    Forward KL: KL(teacher || student), averaged over tokens.
    Dividing logits by T then computing KL is equivalent to CE with soft labels.
    """
    T = temperature
    # reshape to (B*T, vocab) so batchmean divides by number of tokens, not just batch size
    student_logits = student_logits.view(-1, student_logits.size(-1))
    teacher_logits = teacher_logits.view(-1, teacher_logits.size(-1))
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_probs = F.softmax(teacher_logits / T, dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")


def mse_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """MSE on mean-centered logits. Centering removes the mean which is irrelevant
    to the output distribution (softmax is translation-invariant)."""
    student_logits = student_logits - student_logits.mean(dim=-1, keepdim=True)
    teacher_logits = teacher_logits - teacher_logits.mean(dim=-1, keepdim=True)
    return F.mse_loss(student_logits, teacher_logits / temperature)


def compute_loss(
    student_logits: torch.Tensor,
    labels: torch.Tensor,
    loss_type: str,
    teacher_logits: torch.Tensor | None = None,
    teacher_temperature: float = 1.0,
) -> torch.Tensor:
    if loss_type == "ce":
        return ce_loss(student_logits, labels)
    elif loss_type == "kl":
        assert teacher_logits is not None
        return kl_loss(student_logits, teacher_logits, teacher_temperature)
    elif loss_type == "mse":
        assert teacher_logits is not None
        return mse_loss(student_logits, teacher_logits, teacher_temperature)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type!r}")
