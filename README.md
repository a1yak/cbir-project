# Content-Based Image Retrieval (CBIR) — ResNet50 + kNN

This repo implements a **reproducible CBIR pipeline**:
- Download a public dataset (Oxford-IIIT Pet via `torchvision`)
- Extract **AI embeddings** from a **pre-trained ResNet50**
- Build a **nearest-neighbor index** (sklearn, cosine distance)
- Query with an image → get Top‑K similar images
- Produce artifacts: **CSV/JSON + retrieval grid image + evaluation plot**

## Repo structure
- `src/` code
- `data/` dataset location + `data/sample/` sample query images
- `results/` generated index, query outputs, plots

## 0) Setup (Windows / PowerShell)

From the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

## 1) Download dataset (and create sample queries)

```powershell
python -m src.data.download --out_dir data --make_samples 1
```

After this, images will be in:
`data/oxford-iiit-pet/images`

Sample query images will be copied to:
`data/sample/query1.jpg`, `data/sample/query2.jpg`, `data/sample/query3.jpg`

## 2) Build the CBIR index (embeddings + kNN)

```powershell
python -m src.index.build --images_dir data/oxford-iiit-pet/images --out_dir results/index --batch_size 32
```

Outputs:
- `results/index/embeddings.npy`
- `results/index/metadata.csv` (paths + labels)
- `results/index/nn_index.joblib`
- `results/index/config.json`

## 3) Run a query (Top‑K retrieval + artifacts)

```powershell
python -m src.search.query --query_path data/sample/query1.jpg --index_dir results/index --k 10 --out_dir results/queries/query1
```

Outputs:
- `results/queries/query1/results.csv`
- `results/queries/query1/results.json`
- `results/queries/query1/retrieval_grid.png`

## 4) Evaluate (Precision@K on random queries)

```powershell
python -m src.eval.evaluate --images_dir data/oxford-iiit-pet/images --index_dir results/index --num_queries 100 --k_list 1 5 10 --out_dir results/plots
```

Outputs:
- `results/plots/metrics.csv`
- `results/plots/precision_at_k.png`

## Notes / Known issues
- First index build can be slow on CPU. If you have an NVIDIA GPU and CUDA PyTorch installed, it will be faster automatically.
- We use **cosine distance** on L2-normalized embeddings.
