"""The frozen CLIP encoder (openai/clip-vit-base-patch32) of the Flickr8k experiment:
`load_encoder` returns the batch encoder, the projection width and the per-item image
processor."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_common import features_tensor  # noqa: E402

MODEL = "openai/clip-vit-base-patch32"


class _CLIPBatchEncoder:
    """(image feature dict, list_of_str, labels) -> normalised image/text features on device.
    Image preprocessing already happened in the loader workers (Preprocessor)."""

    def __init__(self, model, processor):
        self.model = model
        self.tokenizer = processor.tokenizer
        self.dim = model.config.projection_dim

    @torch.no_grad()
    def __call__(self, batch, device):
        feats, texts, labels = batch
        feats = {k: v.to(device, non_blocking=True) for k, v in feats.items()}
        ie = features_tensor(self.model.get_image_features(**feats), self.dim)
        txt_in = self.tokenizer(list(texts), return_tensors="pt", padding=True,
                                truncation=True, max_length=77).to(device)
        te = features_tensor(self.model.get_text_features(**txt_in), self.dim)
        return ie, te, labels.to(device)


def load_encoder(device: str):
    from transformers import CLIPModel, CLIPProcessor

    from run_common import Preprocessor

    print(f"Loading {MODEL} onto {device} ...")
    model = CLIPModel.from_pretrained(MODEL).eval().to(device)
    processor = CLIPProcessor.from_pretrained(MODEL)
    return (_CLIPBatchEncoder(model, processor), model.config.projection_dim,
            Preprocessor("image", processor.image_processor))

