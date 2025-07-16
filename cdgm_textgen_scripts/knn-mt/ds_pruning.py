import argparse
import torch
import os
import numpy as np
import time
import faiss.contrib.torch_utils
import torch.nn.functional as F
from sklearn.cluster import kmeans_plusplus

def greedy_merging(keys, vals, knn_index, args, knn_k=8):
    merge_weights = torch.ones(knn_index.ntotal, dtype=torch.float32)
    i = 0
    for k in keys:
        dists, knn_idxs = knn_index.search(k, knn_k)
        for knn_idx in knn_idxs:
            if merge_weights[knn_idx] == 1 and vals[knn_idx[0]] == vals[i] and knn_idx != i:
                merge_weights[i] += 1
                merge_weights[knn_idx] -= 1
        i += 1

    new_total = (merge_weights > 0).int().sum()
    print(f"Old total: {knn_index.ntotal} New total after merging: {new_total}")
    mapped_features = np.memmap(f"{args.merged_path}.keys.npy", dtype='float32', mode='w+', shape=(new_total, args.dim))
    mapped_keys = np.memmap(f"{args.merged_path}.vals.npy", dtype='int32', mode='w+', shape=(new_total,))
    mapped_weights = np.memmap(f"{args.merged_path}.merge_weights.npy", dtype='int32', mode='w+', shape=(new_total,))

    j = 0
    for i, s in enumerate(merge_weights):
        if s < 0:
            continue
        else:
            mapped_features[j] = keys[i]
            mapped_keys[j] = vals[i]
            mapped_weights[j] = s
            j += 1

def sq_cdist(X, Y):
    """result[i,j] is squared eucl distance between X[i] and Y[j]"""
    diff = X.unsqueeze(dim=1) - Y.unsqueeze(dim=0)  # len(X) by len(Y)
    return diff.square().sum(dim=-1)

def gmm_merge(keys, vals, args, n_iters=10, sigma=1.0):

    total = len(vals)

    unique_vals, counts = torch.unique(vals, return_inverse=False, return_counts=True)
    print(f"Unique values: {len(unique_vals)}")
    new_total = 0
    # open memmaps for merged keys and vals, real size is not known in advance, so we can set it to total and re-save with correct size later
    mapped_features = np.memmap(f"{args.merged_path}.keys.npy", dtype='float32', mode='w+',
                                shape=(total, args.dim))
    mapped_keys = np.memmap(f"{args.merged_path}.vals.npy", dtype='int32', mode='w+', shape=(total,))
    mapped_weights = np.memmap(f"{args.merged_path}.merge_weights.npy", dtype='int32', mode='w+',
                               shape=(total,))
    new_total = 0

    for i,vocab_idx in enumerate(unique_vals):
        mask = vals == vocab_idx
        keys_subset = torch.tensor(keys[mask], device=vals.device, dtype=torch.float32)

        num_clusters = min(1, counts[i]*0.05)

        # Initialize centroids using k-means++
        centers, _ = kmeans_plusplus(keys_subset.cpu().numpy(), n_clusters=num_clusters, random_state=42)
        centers = torch.tensor(centers, dtype=torch.float32).to(keys_subset.device)

        new_keys = torch.zeros(keys_subset.shape, dtype=torch.float32, device=keys_subset.device)
        new_probs = torch.zeros((len(centers),num_clusters), counts[i], dtype=torch.float32, device=keys_subset.device)
        for _ in range(n_iters):
            sq_cdists = sq_cdist(keys_subset, centers)
            logp_xIz = -sq_cdists / (2 * sigma ** 2)
            # version 1: in log domain, then take exp
            # logq_zIx = logp_xIz - torch.logsumexp(logp_xIz, dim=1, keepdim=True)
            # q_zIx = torch.exp(logq_zIx)
            #
            # version 2 (more readable?)
            q_zIx = torch.softmax(logp_xIz, dim=1)

            new_keys[:] = centers
            new_probs[:] = q_zIx

            #elbo = (q_zIx * logp_xIz).sum() - (q_zIx * torch.log(q_zIx)).sum()

            # recalculate center means
            centers = (q_zIx.T @ keys_subset) / q_zIx.sum(dim=0).unsqueeze(dim=-1)

        new_total += len(centers)

        for j, center in enumerate(centers):
            mapped_features[new_total:new_total + j] = new_keys.cpu().numpy()
            mapped_keys[new_total:new_total+j] = vocab_idx
            mapped_weights[new_total:new_total+j] = new_probs.cpu().numpy()


if __name__=='main':
    argp = argparse.ArgumentParser()
    argp.add_argument("--keys_path", type=str, required=True)
    argp.add_argument("--values_path", type=str, required=True)
    argp.add_argument('--dstore_size', type=int, help='number of items saved in the datastore memmap')
    argp.add_argument("--faiss_index", type=str, default=None)
    argp.add_argument("--dim", type=int, default=128)
    argp.add_argument("--knn_k", type=int, default=8, help="number of nearest neighbors to consider for merging")
    argp.add_argument("--merged_path", type=str, required=True, help="path to save merged keys and vals")


    args = argp.parse_args()

    if args.faiss_index is not None:
        res = faiss.StandardGpuResources()
        co = faiss.GpuClonerOptions()
        co.useFloat16 = True
        knn_index = faiss.read_index(args.faiss_index)
        knn_index = faiss.index_cpu_to_gpu(res, 0, knn_index, co)
        ntotal = knn_index.ntotal
    else:
        ntotal = args.dstore_size
        assert ntotal is not None, "If no faiss index is provided, dstore_size must be specified"
        knn_index = None

    knn_k = args.knn_k

    print("load vals to retrieve vocab indices")
    vals_memmap = np.memmap(args.values_path, dtype=np.int32, mode='r', shape=(ntotal,))
    vals = torch.zeros(knn_index.ntotal, dtype=torch.int64, device='cuda' if torch.cuda.is_available() else 'cpu')
    vals[:] = torch.from_numpy(vals_memmap[:])

    keys = np.memmap(args.keys_path, dtype=np.float32, mode='r', shape=(ntotal, args.dim))

    gmm_merge(keys, vals, args, n_iters=10, sigma=1.0)



















