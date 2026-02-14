from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List
import numpy as np

import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

@dataclass
class EncoderConfig:
    model_name: str = "resnet50"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    image_size: int = 224

class ImageEncoder:
    def __init__(self, cfg: EncoderConfig):
        self.cfg = cfg
        if cfg.model_name.lower() != "resnet50":
            raise ValueError("Only resnet50 is implemented in this template (simple + reliable).")

        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        model.fc = nn.Identity()
        model.eval()
        self.model = model.to(cfg.device)

        self.preprocess = transforms.Compose([
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=weights.transforms().mean, std=weights.transforms().std),
        ])

    @torch.inference_mode()
    def encode_pil_batch(self, images: List[Image.Image], batch_size: int = 32) -> np.ndarray:
        device = self.cfg.device
        feats = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            x = torch.stack([self.preprocess(im.convert("RGB")) for im in batch]).to(device)
            y = self.model(x)  # (B, 2048)
            y = torch.nn.functional.normalize(y, p=2, dim=1)
            feats.append(y.detach().cpu().numpy().astype("float32"))
        return np.vstack(feats) if feats else np.zeros((0, 2048), dtype="float32")

    @torch.inference_mode()
    def encode_paths(self, paths: List[str], batch_size: int = 32) -> np.ndarray:
        images = [Image.open(p) for p in paths]
        try:
            return self.encode_pil_batch(images, batch_size=batch_size)
        finally:
            for im in images:
                try:
                    im.close()
                except Exception:
                    pass

    @torch.inference_mode()
    def encode_single(self, path: str) -> np.ndarray:
        im = Image.open(path)
        try:
            x = self.preprocess(im.convert("RGB")).unsqueeze(0).to(self.cfg.device)
            y = self.model(x)
            y = torch.nn.functional.normalize(y, p=2, dim=1)
            return y.detach().cpu().numpy().astype("float32")[0]
        finally:
            try:
                im.close()
            except Exception:
                pass
