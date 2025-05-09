import argparse
import numpy as np
import faiss
import yaml

from visualize import Embedding

def _extract_centroids(index: faiss.Index):
    """Return the (k, d) centroid matrix whatever the Faiss version."""
    k, d = index.ntotal, index.d
    if hasattr(index, "get_xb"):                 # Faiss ≥ 1.7.3
        ptr = index.get_xb()
        return faiss.rev_swig_ptr(ptr, k * d).reshape(k, d)
    if hasattr(index, "xb"):                     # Faiss ≤ 1.7.2
        return faiss.vector_to_array(index.xb).reshape(k, d)
    return index.reconstruct_n(0, k).reshape(k, d)  # last‑resort

def calculate_kmeans_metrics(kmeans_index: faiss.Index,
                             train_vecs: np.ndarray) -> dict:
    """
    Compute intra / inter metrics for a K‑means model written with
    `faiss.write_index(kmeans.index, …)`.
    """
    # ---- 1. nearest‑centroid assignment
    D2, labels = kmeans_index.search(np.asarray(train_vecs), 1)
    labels = labels.astype(np.int64).ravel()
    D2     = D2.ravel()                # squared intra‑distances

    k = int(kmeans_index.ntotal)       # # clusters
    n = labels.size                    # # points

    # ---- 2. cluster sizes, inertia, imbalance
    sizes          = np.bincount(labels, minlength=k)
    inertia_total  = float(D2.sum())
    imbalance      = faiss.imbalance_factor(
                         n, k, faiss.swig_ptr(labels))  # Faiss utility

    # ---- 3. intra radii
    radii = np.zeros(k, dtype=np.float64)
    for cid in range(k):
        if sizes[cid]:
            radii[cid] = np.sqrt(D2[labels == cid].mean())
    radius_max, radius_mean = map(float, (radii.max(), radii.mean()))

    # ---- 4. inter‑centroid gaps
    centroids = _extract_centroids(kmeans_index)
    c_norm2   = (centroids**2).sum(1, keepdims=True)
    dist2     = c_norm2 + c_norm2.T - 2 * centroids @ centroids.T
    np.clip(dist2, 0., None, out=dist2)
    iu            = np.triu_indices(k, 1)
    inter_min     = float(np.sqrt(dist2[iu].min()))
    inter_mean    = float(np.sqrt(dist2[iu].mean()))
    dunn_index    = inter_min / radius_max if radius_max else float("inf")

    return dict(
        n_points         = n,
        n_clusters       = k,
        imbalance_factor = round(imbalance, 3),
        inertia_total    = inertia_total,
        radius_mean      = round(radius_mean, 3),
        radius_max       = round(radius_max, 3),
        inter_min        = round(inter_min, 3),
        inter_mean       = round(inter_mean, 3),
        dunn_index       = round(dunn_index, 3),
    )


def train_knn(embedding: Embedding):
    import yaml, pathlib, numpy as np, faiss

    ncentroids, niter, verbose = 1024, 20, True
    d = embedding.dim()
    print(f"Training K‑means ({d}‑D, k={ncentroids}, iter={niter})")

    # ---- K‑means fit -------------------------------------------------------
    kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose)
    X      = embedding.keys[:1_000_000]          # sample for speed
    kmeans.train(X)

    # ---- persist centroids -------------------------------------------------
    kmeans_path = embedding.faiss_path.with_suffix(".kmeans")
    kmeans_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(kmeans.index, str(kmeans_path))
    print(f"Centroids written to {kmeans_path}")

    # ---- (optional) save the training subset ------------------------------
    X_path = embedding.faiss_path.with_suffix(".X.npy")
    np.save(X_path, X)
    print(f"Training sample written to {X_path}")

    # ---- compute & display metrics ----------------------------------------
    metrics = calculate_kmeans_metrics(kmeans.index, X)
    print("\n=== K‑means training metrics ===")
    print(yaml.dump(metrics, sort_keys=False, indent=2))




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
    embedding.load_raw()
    train_knn(embedding)


if __name__ == "__main__":
    main()
