import yaml
import faiss
import numpy as np
from argparse import Namespace
import argparse

from visualize import load_yaml, Embedding

def train_knn(embedding: Embedding):
    ncentroids = 1024
    niter = 20
    verbose = True
    d = embedding.dim()
    print(f"Training KNN with {d} dimensions")
    print("ncentroids", ncentroids)
    print("niter", niter)
    print("verbose", verbose)
    kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose)
    X = embedding.keys[:1000000]
    kmeans.train(X) ######## pass this for a faster run
    # save the trained kmeans model
    kmeans_path = embedding.faiss_path.with_suffix('.kmeans')
    kmeans_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(kmeans.index, str(kmeans_path))
    print(f"Saved KMeans model to {kmeans_path}")

    # save X
    X_path = embedding.faiss_path.with_suffix('.X.npy')
    X_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(X_path, X)
    print(f"Saved X to {X_path}")


def load_knn(embedding: Embedding):
    # load the knn index
    knn_path = embedding.faiss_path.with_suffix('.kmeans')
    if not knn_path.exists():
        raise ValueError(f"KNN model not found at {knn_path}")
    print(f"Loading KNN model from {knn_path}")
    knn_index = faiss.read_index(str(knn_path))

    # load X
    X_path = embedding.faiss_path.with_suffix('.X.npy')
    if not X_path.exists():
        raise ValueError(f"X not found at {X_path}")
    print(f"Loading X from {X_path}")
    X = np.memmap(X_path, dtype=np.float32, mode='r', shape=(262144, embedding.args.dim))
    return knn_index, X


def parse_args():
    # load form yaml
    parser = argparse.ArgumentParser(description="KNN")
    parser.add_argument("--model", choices=["baseline", "dispersion"], default="baseline", help="Model to use")

    args = parser.parse_args()
    if args.model == "baseline":
        embedding = Embedding(
            base_path='/ivi/ilps/projects/cnmt/knn-mt-dispersion/datastores/model_512_no_dispersion/',
            features_path='train.features.npy',
            yaml_path="410.roen.faiss_L2_IVFPQindex_cs64_nprobe32_d512_nc4096_metadata.yaml",
            faiss_path='410.roen.faiss_L2_IVFPQindex_cs64_nprobe32_d512_nc4096'
        )
    elif args.model == "dispersion":
        embedding = Embedding(
            base_path='/ivi/ilps/projects/cnmt/knn-mt-dispersion/datastores/model_512_dispersion/',
            features_path='582.train.features.npy',
            yaml_path="582.roen.faiss_L2_IVFPQindex_cs64_nprobe32_d512_nc2048_meta.yaml",
            faiss_path='582.roen.faiss_L2_IVFPQindex_cs64_nprobe32_512_4096'
        )
    else:
        raise ValueError("Model not found")
    return embedding


def main():
    embedding = parse_args()
    # load faiss index
    embedding.load_raw()
    # embedding.load_faiss(gpu=False)
    # train knn
    train_knn(embedding)


def _extract_centroids(index):
    """
    Return the (k, d) centroid matrix from a K‑means index that was saved with
    `faiss.write_index(kmeans.index, ...)`, regardless of the Faiss version.
    """
    k, d = index.ntotal, index.d

    # Faiss ≥ 1.7.3  →  use get_xb()
    if hasattr(index, "get_xb"):
        ptr = index.get_xb()                         # float*  length k*d
        return faiss.rev_swig_ptr(ptr, k * d).reshape(k, d)

    # older Faiss  →  xb is still a C++ std::vector
    if hasattr(index, "xb"):
        return faiss.vector_to_array(index.xb).reshape(k, d)

    # portable fallback: reconstruct_n
    return index.reconstruct_n(0, k).reshape(k, d)

def calculate_kmeans_metrics(kmeans_index, train_vecs):
    """
    Parameters
    ----------
    kmeans_index : faiss.IndexFlatL2          # what faiss.write_index wrote
    train_vecs   : (N, d) float32 np.ndarray  # or memmap with same dtype

    Returns
    -------
    dict with:
      - imbalance_factor
      - inertia_total
      - radius_mean, radius_max        (intra‑cluster)
      - inter_min, inter_mean          (centroid gaps)
      - dunn_index                     (= inter_min / radius_max)
      - n_points, n_clusters           (book‑keeping)
    """
    import numpy as np
    import faiss

    # ---- 1. Assign every vector to its nearest centroid
    # Faiss returns squared distances (L2)
    D2, labels = kmeans_index.search(np.asarray(train_vecs), 1)
    labels = labels.ravel()
    D2 = D2.ravel()                      # squared within‑cluster distances

    k = kmeans_index.ntotal              # number of clusters
    n = train_vecs.shape[0]

    # ---- 2. Basic cluster statistics
    sizes = np.bincount(labels, minlength=k)
    inertia_total = float(D2.sum())      # a.k.a. within‑cluster SSE
    imbalance = (k * (sizes**2).sum()) / (sizes.sum()**2)

    # ---- 3. Intra‑cluster radii (√mean distance per cluster)
    #       radius_max   = worst‑case tightness
    #       radius_mean  = average tightness
    radii = np.zeros(k, dtype=np.float64)
    for cid in range(k):
        if sizes[cid]:
            radii[cid] = np.sqrt(D2[labels == cid].mean())
    radius_max  = float(radii.max())
    radius_mean = float(radii.mean())

    # ---- 4. Inter‑cluster centroid distances
    d = kmeans_index.d
    centroids = _extract_centroids(kmeans_index)

    # pairwise squared distances, but keep only upper triangle
    c_norm2 = (centroids**2).sum(1, keepdims=True)
    dist2 = c_norm2 + c_norm2.T - 2 * centroids @ centroids.T
    np.clip(dist2, 0.0, None, out=dist2)               # numerical safety
    iu = np.triu_indices(k, k=1)
    inter_min  = float(np.sqrt(dist2[iu].min()))
    inter_mean = float(np.sqrt(dist2[iu].mean()))

    # ---- 5. Composite quality index (Dunn)
    dunn = inter_min / radius_max if radius_max > 0 else float("inf")

    return dict(
        n_points          = int(n),
        n_clusters        = int(k),
        imbalance_factor  = float(imbalance),
        inertia_total     = inertia_total,
        radius_mean       = radius_mean,
        radius_max        = radius_max,
        inter_min         = inter_min,
        inter_mean        = inter_mean,
        dunn_index        = dunn,
    )


def eval():
    embedding = parse_args()
    # load faiss index
    embedding.load_raw()
    # embedding.load_faiss(gpu=False)
    # train knn
    knn_index, train_vecs = load_knn(embedding)


    metrics = calculate_kmeans_metrics(knn_index, train_vecs)
    print(metrics)
  

if __name__ == "__main__":
    main()
    # eval()