"""
Lightweight projection heads for contrastive fine-tuning.

Input: pre-extracted frozen encoder embeddings (e.g. from CLIP or CLAP).
Trainable: two 2-layer MLP projection heads + log-temperature scalar.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int = 512, out_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class ContrastiveModel(nn.Module):
    def __init__(self, in_dim: int = 512, out_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.vision_proj = ProjectionHead(in_dim, out_dim, hidden_dim)
        self.text_proj = ProjectionHead(in_dim, out_dim, hidden_dim)
        # log(1/tau); clamped to [0.01, 100]
        self.log_temp = nn.Parameter(torch.zeros(1))

    @property
    def temp(self) -> torch.Tensor:
        return self.log_temp.exp().clamp(0.01, 100.0)

    def forward(self, img_emb: torch.Tensor, txt_emb: torch.Tensor):
        return self.vision_proj(img_emb), self.text_proj(txt_emb)
