"""
Audio and text encoders trained from random initialization.

AudioEncoder: log-Mel (80 bins, 10 ms) computed on the GPU -> 2-layer Conv2d
subsampling (x4 in time) -> pre-LN Transformer encoder -> masked mean pool ->
projection. TextEncoder: character embeddings -> Transformer encoder ->
masked mean pool -> projection. Both ends are L2-normalised; a learned
temperature scales the similarity.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


@dataclass
class Config:
    d_model: int = 384
    audio_layers: int = 6
    text_layers: int = 4
    n_heads: int = 6
    ff_mult: int = 4
    dropout: float = 0.1
    out_dim: int = 256
    n_mels: int = 80


CONFIGS = {
    "tiny": Config(d_model=256, audio_layers=4, text_layers=2, n_heads=4, out_dim=128),
    "small": Config(),
    "base": Config(d_model=768, audio_layers=12, text_layers=6, n_heads=12, out_dim=512),
}


def sinusoid(n: int, d: int, device) -> torch.Tensor:
    pos = torch.arange(n, device=device).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2, device=device) * (-math.log(10000.0) / d))
    pe = torch.zeros(n, d, device=device)
    pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
    return pe


def masked_mean(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    mask = torch.arange(x.size(1), device=x.device)[None, :] < lengths[:, None]
    return (x * mask.unsqueeze(-1)).sum(1) / lengths.clamp(min=1).unsqueeze(-1)


def encoder_stack(cfg: Config, n_layers: int) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(cfg.d_model, cfg.n_heads, cfg.ff_mult * cfg.d_model, cfg.dropout,
                                       activation="gelu", batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)


class AudioEncoder(nn.Module):
    def __init__(self, cfg: Config, sample_rate: int = 16000):
        super().__init__()
        self.cfg = cfg
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate, n_fft=400, win_length=400, hop_length=160, n_mels=cfg.n_mels)
        d = cfg.d_model
        self.conv = nn.Sequential(nn.Conv2d(1, d, 3, 2), nn.GELU(), nn.Conv2d(d, d, 3, 2), nn.GELU())
        f = ((cfg.n_mels - 1) // 2 - 1) // 2
        self.proj_in = nn.Linear(d * f, d)
        self.encoder = encoder_stack(cfg, cfg.audio_layers)
        self.norm = nn.LayerNorm(d)
        self.proj = nn.Linear(d, cfg.out_dim)

    def forward(self, wave: torch.Tensor, wave_len: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=wave.device.type, enabled=False):
            x = torch.log(self.mel(wave.float()) + 1e-6).transpose(1, 2)  # (B, T, mels)
            x = (x - x.mean(dim=(1, 2), keepdim=True)) / (x.std(dim=(1, 2), keepdim=True) + 1e-5)
        lens = wave_len // 160 + 1
        x = self.conv(x.unsqueeze(1))  # (B, d, T', f)
        b, d, t, f = x.shape
        x = self.proj_in(x.transpose(1, 2).reshape(b, t, d * f))
        lens = ((lens - 1) // 2 - 1) // 2
        lens = lens.clamp(min=1, max=t)
        x = x + sinusoid(t, d, x.device)[None]
        pad = torch.arange(t, device=x.device)[None, :] >= lens[:, None]
        x = self.encoder(x, src_key_padding_mask=pad)
        return F.normalize(self.proj(self.norm(masked_mean(x, lens))), dim=-1)


class TextEncoder(nn.Module):
    def __init__(self, cfg: Config, vocab: int):
        super().__init__()
        self.emb = nn.Embedding(vocab, cfg.d_model, padding_idx=0)
        self.encoder = encoder_stack(cfg, cfg.text_layers)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.out_dim)

    def forward(self, ids: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
        x = self.emb(ids) * math.sqrt(self.emb.embedding_dim)
        x = x + sinusoid(x.size(1), x.size(2), x.device)[None]
        pad = torch.arange(x.size(1), device=x.device)[None, :] >= lens[:, None]
        x = self.encoder(x, src_key_padding_mask=pad)
        return F.normalize(self.proj(self.norm(masked_mean(x, lens))), dim=-1)


class ScratchCLAP(nn.Module):
    def __init__(self, cfg: Config, vocab: int):
        super().__init__()
        self.audio = AudioEncoder(cfg)
        self.text = TextEncoder(cfg, vocab)
        self.log_temp = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    @property
    def temp(self) -> torch.Tensor:
        return self.log_temp.exp().clamp(max=100.0)

    def forward(self, wave, wave_len, ids, ids_len):
        return self.audio(wave, wave_len), self.text(ids, ids_len)
