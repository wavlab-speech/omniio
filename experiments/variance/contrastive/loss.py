"""Symmetric InfoNCE (NT-Xent) loss for contrastive learning."""
import torch
import torch.nn.functional as F


def infonce_loss(img_proj: torch.Tensor, txt_proj: torch.Tensor, temp: torch.Tensor) -> torch.Tensor:
    """
    img_proj, txt_proj: (B, D) L2-normalised embeddings.
    Returns scalar loss.
    """
    B = img_proj.size(0)
    sim = (img_proj @ txt_proj.T) * temp          # (B, B)
    labels = torch.arange(B, device=img_proj.device)
    loss_i2t = F.cross_entropy(sim, labels)
    loss_t2i = F.cross_entropy(sim.T, labels)
    return (loss_i2t + loss_t2i) * 0.5
