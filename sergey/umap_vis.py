from collections import defaultdict
from pathlib import Path
import faiss
import numpy as np
from argparse import Namespace
from tqdm.auto import tqdm

import numpy as np
import umap                         # pip install umap-learn
import matplotlib.pyplot as plt

import torch

def load_yaml(yaml_path):
    import yaml
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

class Embedding:
    def __init__(self, base_path, features_path, yaml_path, faiss_path, normalize=False):
        self.features_path = Path(base_path) / features_path
        self.faiss_path = Path(base_path) / faiss_path
        self.args = Namespace(**load_yaml(Path(base_path) / yaml_path))
        print(self.args)
        self.keys = None
        self.knn_index = None
        # self.vals = None




def main():
    # path = "/ivi/ilps/projects/cnmt/knn-mt-dispersion/datastores/model_512_no_dispersion/train.features.npy"
    # yaml_path = "/ivi/ilps/projects/cnmt/knn-mt-dispersion/datastores/model_512_no_dispersion/410.roen.faiss_L2_IVFPQindex_cs64_nprobe32_d512_nc4096_metadata.yaml"
    # name = "model_512_no_dispersion"


    path = "/ivi/ilps/projects/cnmt/knn-mt-dispersion/datastores/model_512_dispersion/582.train.features.npy"
    yaml_path = "/ivi/ilps/projects/cnmt/knn-mt-dispersion/datastores/model_512_dispersion/582.roen.faiss_L2_IVFPQindex_cs64_nprobe32_d512_nc2048_meta.yaml"
    name = "model_512_dispersion"


    args = Namespace(**load_yaml(yaml_path))
    data = np.memmap(path, dtype=np.float32, mode='r', shape=(args.ntotal, args.dim))
    X = data[:100000]

    # visualize using umap with l2 distances
    X = X.astype(np.float32)            # UMAP prefers float32

    # OPTIONAL: down‑sample further if plotting 100k points is too heavy
    # idx = np.random.RandomState(42).choice(len(X), 30_000, replace=False)
    # X = X[idx]

    # ------------------------------------------------------------
    # 3.  Fit UMAP (Euclidean = default) and transform
    # ------------------------------------------------------------
    umap_2d = umap.UMAP(
        n_components=2,
        n_neighbors=50,     # ↔ local structure; tweak if blobs look too tight/loose
        min_dist=0.1,       # ↔ how close points can get; lower shows micro‑clusters
        metric="euclidean", # L2 distance, matches your embeddings
        random_state=42,
        n_jobs=8,        # parallelize
        verbose=True,
    )
    embedding_2d = umap_2d.fit_transform(X)

    # ------------------------------------------------------------
    # 4.  Plot – small grey dots work for big clouds
    # ------------------------------------------------------------
    plt.figure(figsize=(8, 8))
    plt.scatter(embedding_2d[:, 0],
                embedding_2d[:, 1],
                s=1, alpha=0.6)          # tiny markers for visibility
    plt.title("UMAP projection of 100 k vectors (L2 metric)")
    plt.xlabel("UMAP‑1")
    plt.ylabel("UMAP‑2")
    plt.tight_layout()
    plt.show()

    # save the plot
    plt.savefig(f"figures/{name}_umap_projection.png", dpi=300)



if __name__ == "__main__":
    main()