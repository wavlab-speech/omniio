"""The frozen CLAP encoder (laion/clap-htsat-unfused) of the ESC-50 experiment: `load_encoder`
returns the batch encoder, the projection width and the per-item feature extractor. Audio is
stored at 48 kHz, the rate the CLAP feature extractor requires."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_common import features_tensor  # noqa: E402

MODEL = "laion/clap-htsat-unfused"


class _CLAPBatchEncoder:
    """(audio feature dict, list_of_str, labels) -> normalised audio/text features on device.
    Mel features already computed in the loader workers (Preprocessor); the sampling
    rate is checked there by the feature extractor."""

    def __init__(self, model, processor):
        self.model = model
        self.tokenizer = processor.tokenizer
        self.dim = model.config.projection_dim

    @torch.no_grad()
    def __call__(self, batch, device):
        feats, texts, labels = batch
        feats = {k: v.to(device, non_blocking=True) for k, v in feats.items()}
        ae = features_tensor(self.model.get_audio_features(**feats), self.dim)
        txt_in = self.tokenizer(list(texts), return_tensors="pt", padding=True).to(device)
        te = features_tensor(self.model.get_text_features(**txt_in), self.dim)
        return ae, te, labels.to(device)


def load_encoder(device: str):
    from transformers import ClapModel, ClapProcessor

    from run_common import Preprocessor

    print(f"Loading {MODEL} onto {device} ...")
    model = ClapModel.from_pretrained(MODEL).eval().to(device)
    processor = ClapProcessor.from_pretrained(MODEL)
    fe = processor.feature_extractor
    print(f"  feature extractor expects {fe.sampling_rate} Hz audio")
    return (_CLAPBatchEncoder(model, processor), model.config.projection_dim,
            Preprocessor("audio", fe))

