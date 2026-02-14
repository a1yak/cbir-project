from __future__ import annotations

import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import joblib

from src.features.encoder import ImageEncoder, EncoderConfig
from src.utils.seed import set_seed
from src.utils.paths import ensure_dir
from src.utils.labels import label_from_filename

def main():
    parser = argparse.ArgumentParser(description="Build CBIR index: embeddings + kNN.")
    parser.add_argument("--images_dir", type=str, required=True, help="Folder with dataset images (*.jpg).")
    parser.add_argument("--out_dir", type=str, default="results/index", help="Where to save the index.")
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    images_dir = Path(args.images_dir)
    out_dir = ensure_dir(args.out_dir)

    paths = sorted([str(p) for p in images_dir.glob("*.jpg")])
    if len(paths) == 0:
        raise RuntimeError(f"No .jpg files found in {images_dir}")

    labels = [label_from_filename(p) for p in paths]
    meta = pd.DataFrame({"path": paths, "label": labels})
    meta_path = out_dir / "metadata.csv"
    meta.to_csv(meta_path, index=False)

    encoder = ImageEncoder(EncoderConfig(model_name=args.model))
    print(f"Encoding {len(paths)} images on device: {encoder.cfg.device} ...")

    # Encode in chunks to keep memory stable
    chunk = 2000
    feats = []
    for i in tqdm(range(0, len(paths), chunk)):
        batch_paths = paths[i:i+chunk]
        emb = encoder.encode_paths(batch_paths, batch_size=args.batch_size)
        feats.append(emb)
    embeddings = np.vstack(feats).astype("float32")

    emb_path = out_dir / "embeddings.npy"
    np.save(emb_path, embeddings)

    print("Fitting NearestNeighbors (cosine) ...")
    nn = NearestNeighbors(metric="cosine", algorithm="auto")
    nn.fit(embeddings)
    index_path = out_dir / "nn_index.joblib"
    joblib.dump(nn, index_path)

    cfg = {
        "model": args.model,
        "device": encoder.cfg.device,
        "metric": "cosine",
        "num_items": int(len(paths)),
        "embedding_dim": int(embeddings.shape[1]),
        "seed": args.seed,
    }
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    print("✅ Index built!")
    print(f"- {emb_path}")
    print(f"- {meta_path}")
    print(f"- {index_path}")

if __name__ == "__main__":
    main()
