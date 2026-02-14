from __future__ import annotations

import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from src.features.encoder import ImageEncoder, EncoderConfig
from src.utils.paths import ensure_dir

def save_grid(query_path: str, result_paths: list[str], out_path: Path, title: str):
    n = 1 + len(result_paths)
    cols = min(6, n)
    rows = (n + cols - 1) // cols

    fig = plt.figure(figsize=(3*cols, 3*rows))
    ax = fig.add_subplot(rows, cols, 1)
    ax.imshow(plt.imread(query_path))
    ax.set_title("QUERY")
    ax.axis("off")

    for i, p in enumerate(result_paths, start=2):
        ax = fig.add_subplot(rows, cols, i)
        ax.imshow(plt.imread(p))
        ax.set_title(f"#{i-1}")
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Query CBIR index with an image.")
    parser.add_argument("--query_path", type=str, required=True)
    parser.add_argument("--index_dir", type=str, default="results/index")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--out_dir", type=str, default="results/queries/query1")
    args = parser.parse_args()

    index_dir = Path(args.index_dir)
    out_dir = ensure_dir(args.out_dir)

    meta = pd.read_csv(index_dir / "metadata.csv")
    embeddings = np.load(index_dir / "embeddings.npy")
    nn = joblib.load(index_dir / "nn_index.joblib")

    encoder = ImageEncoder(EncoderConfig(model_name="resnet50"))
    q_emb = encoder.encode_single(args.query_path).reshape(1, -1)

    dists, idxs = nn.kneighbors(q_emb, n_neighbors=min(args.k, len(meta)))
    dists = dists[0].tolist()
    idxs = idxs[0].tolist()

    rows = []
    result_paths = []
    for rank, (i, d) in enumerate(zip(idxs, dists), start=1):
        p = meta.iloc[i]["path"]
        lbl = meta.iloc[i]["label"]
        rows.append({"rank": rank, "path": p, "label": lbl, "cosine_distance": float(d)})
        result_paths.append(p)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "results.csv", index=False)
    (out_dir / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    save_grid(args.query_path, result_paths, out_dir / "retrieval_grid.png",
              title=f"Top-{len(result_paths)} similar images (cosine distance)")

    print("✅ Query done!")
    print(f"- {out_dir / 'results.csv'}")
    print(f"- {out_dir / 'retrieval_grid.png'}")

if __name__ == "__main__":
    main()
