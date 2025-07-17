import argparse
import torch
import os
import numpy as np
import time
import faiss.contrib.torch_utils
import torch.nn.functional as F
from sklearn.cluster import kmeans_plusplus
import gc 

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
    return (X.detach().unsqueeze(dim=1) - Y.detach().unsqueeze(dim=0)).pow_(2).sum(dim=-1)  # len(X) by len(Y)

def sq_cdist_chunked(X, Y, chunk_size=1024):
    """
    Compute squared Euclidean distance between each pair in X and Y in chunks.
    Returns a tensor of shape [len(X), len(Y)].
    """
    X = X.detach()
    Y = Y.detach()
    results = []

    for start in range(0, X.size(0), chunk_size):
        end = min(start + chunk_size, X.size(0))
        X_chunk = X[start:end]  # shape: [chunk_size, D]
        dists = (X_chunk.unsqueeze(1) - Y.unsqueeze(0)).pow(2).sum(dim=-1)  # [chunk_size, len(Y)]
        results.append(dists)

    return torch.cat(results, dim=0)

def gmm_merge(keys, vals, args, n_iters=10, sigma=1.0):
    total = len(vals)

    unique_vals, counts = torch.unique(vals, return_inverse=False, return_counts=True)
    print(f"Unique values: {len(unique_vals)}")
    new_total = 0
    # open memmaps for merged keys and vals, real size is not known in advance, so we can set it to total and re-save with correct size later
    mapped_keys = np.memmap(f"{args.merged_path}.keys.npy", dtype='float32', mode='w+',
                                shape=(total, args.dim))
    mapped_vals = np.memmap(f"{args.merged_path}.vals.npy", dtype='int32', mode='w+', shape=(total,))
    #mapped_weights = np.memmap(f"{args.merged_path}.merge_weights.npy", dtype='int32', mode='w+',
                               #shape=(total,))

    for i,vocab_idx in enumerate(unique_vals):
        with torch.no_grad():
            mask = vals == vocab_idx

            keys_subset = torch.from_numpy(keys[mask.cpu()]).to(device=vals.device, dtype=torch.float32)

            num_clusters = torch.tensor([max(1, int(counts[i]*0.05))])

            # Initialize centroids using k-means++
            centers, _ = kmeans_plusplus(keys_subset.cpu().numpy(), n_clusters=num_clusters, random_state=42)
            centers = torch.tensor(centers, dtype=torch.float32).to(keys_subset.device)

            #new_probs = torch.zeros((centers.shape[0],num_clusters), dtype=torch.float32, device=keys_subset.device)i

            print(f"fitting gmm for index {vocab_idx} with cluster centers {num_clusters}")

            for it in range(n_iters):
                sq_cdists = sq_cdist_chunked(keys_subset, centers)
                logp_xIz = -sq_cdists / (2 * sigma ** 2)
                # version 1: in log domain, then take exp
                # logq_zIx = logp_xIz - torch.logsumexp(logp_xIz, dim=1, keepdim=True)
                # q_zIx = torch.exp(logq_zIx)
                #
                # version 2 (more readable?)
                q_zIx = torch.softmax(logp_xIz, dim=1)

                #new_probs[:] = q_zIx

                #elbo = (q_zIx * logp_xIz).sum() - (q_zIx * torch.log(q_zIx)).sum()

                # recalculate center means
                centers = ((q_zIx.T @ keys_subset) / q_zIx.sum(dim=0).unsqueeze(dim=-1)).detach()


        mapped_keys[new_total:new_total + len(centers)] = centers.cpu().numpy()
        mapped_vals[new_total:new_total+len(centers)] = vocab_idx.cpu().numpy()
         #   mapped_weights[new_total:new_total+j] = new_probs.cpu().numpy(p)
        new_total += len(centers)
    print(f"size before pruning: {total} new total size after pruning: {new_total}")
    with open(f"{args.merged_path}.ntotal.txt",'w') as tf:
        tf.write(new_total)



if __name__=='__main__':
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
    vals = torch.zeros(ntotal, dtype=torch.int64, device='cuda' if torch.cuda.is_available() else 'cpu')
    vals[:] = torch.from_numpy(vals_memmap[:])

    keys = np.memmap(args.keys_path, dtype=np.float32, mode='r', shape=(ntotal, args.dim))
    gmm_merge(keys, vals, args, n_iters=10, sigma=1.0)



















