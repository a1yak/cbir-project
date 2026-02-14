from __future__ import annotations

import argparse
from pathlib import Path
import random

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from src.features.encoder import ImageEncoder, EncoderConfig
from src.utils.seed import set_seed
from src.utils.labels import label_from_filename
from src.utils.paths import ensure_dir

def precision_at_k(retrieved_labels: list[str], query_label: str, k: int) -> float:
    topk = retrieved_labels[:k]
    if len(topk) == 0:
        return 0.0
    return sum(1 for l in topk if l == query_label) / float(len(topk))

def main():
    parser = argparse.ArgumentParser(description="Evaluate CBIR with Precision@K on random queries.")
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--index_dir", type=str, default="results/index")
    parser.add_argument("--out_dir", type=str, default="results/plots")
    parser.add_argument("--num_queries", type=int, default=100)
    parser.add_argument("--k_list", type=int, nargs="+", default=[1,5,10])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = ensure_dir(args.out_dir)

    images_dir = Path(args.images_dir)
    all_paths = sorted([str(p) for p in images_dir.glob("*.jpg")])
    if len(all_paths) == 0:
        raise RuntimeError(f"No .jpg images found in {images_dir}")

    index_dir = Path(args.index_dir)
    meta = pd.read_csv(index_dir / "metadata.csv")
    embeddings = np.load(index_dir / "embeddings.npy")
    nn = joblib.load(index_dir / "nn_index.joblib")

    encoder = ImageEncoder(EncoderConfig(model_name="resnet50"))

    # Pick queries
    q_paths = random.sample(all_paths, k=min(args.num_queries, len(all_paths)))
    max_k = max(args.k_list)
    scores = {k: [] for k in args.k_list}

    # For each query, retrieve max_k+1, then drop identical image if present.
    for qp in q_paths:
        q_label = label_from_filename(qp)
        q_emb = encoder.encode_single(qp).reshape(1, -1)
        dists, idxs = nn.kneighbors(q_emb, n_neighbors=min(max_k + 1, len(meta)))
        idxs = idxs[0].tolist()

        retrieved = []
        for i in idxs:
            p = meta.iloc[i]["path"]
            if Path(p).resolve() == Path(qp).resolve():
                continue
            retrieved.append(meta.iloc[i]["label"])
            if len(retrieved) >= max_k:
                break

        for k in args.k_list:
            scores[k].append(precision_at_k(retrieved, q_label, k))

    rows = [{"k": k, "precision_at_k": float(np.mean(scores[k]))} for k in args.k_list]
    df = pd.DataFrame(rows).sort_values("k")
    df.to_csv(out_dir / "metrics.csv", index=False)

    # Plot
    fig = plt.figure(figsize=(6,4))
    plt.plot(df["k"], df["precision_at_k"], marker="o")
    plt.xlabel("K")
    plt.ylabel("Precision@K")
    plt.title("CBIR Retrieval Quality (ResNet50 embeddings)")
    plt.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "precision_at_k.png", dpi=200)
    plt.close(fig)

    print("✅ Evaluation done!")
    print(f"- {out_dir / 'metrics.csv'}")
    print(f"- {out_dir / 'precision_at_k.png'}")

if __name__ == "__main__":
    main()
