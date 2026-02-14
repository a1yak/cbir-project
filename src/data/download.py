from __future__ import annotations

import argparse
from pathlib import Path
import random
import shutil

from torchvision.datasets import OxfordIIITPet

from src.utils.seed import set_seed
from src.utils.paths import ensure_dir

def main():
    parser = argparse.ArgumentParser(description="Download Oxford-IIIT Pet dataset using torchvision.")
    parser.add_argument("--out_dir", type=str, default="data", help="Where to download dataset (root folder).")
    parser.add_argument("--make_samples", type=int, default=1, help="If 1, copy a few random images to data/sample.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = ensure_dir(args.out_dir)

    # This will download into: <out_dir>/oxford-iiit-pet/...
    OxfordIIITPet(root=str(out_dir), split="trainval", download=True)
    OxfordIIITPet(root=str(out_dir), split="test", download=True)

    images_dir = out_dir / "oxford-iiit-pet" / "images"
    print(f"✅ Dataset ready. Images folder: {images_dir}")

    if args.make_samples == 1:
        sample_dir = ensure_dir(out_dir / "sample")
        all_imgs = sorted([p for p in images_dir.glob("*.jpg")])
        if len(all_imgs) == 0:
            raise RuntimeError(f"No .jpg images found in {images_dir}")
        picks = random.sample(all_imgs, k=min(3, len(all_imgs)))
        for i, p in enumerate(picks, start=1):
            dst = sample_dir / f"query{i}.jpg"
            shutil.copy2(p, dst)
        print(f"✅ Sample queries copied to: {sample_dir}")

if __name__ == "__main__":
    main()
