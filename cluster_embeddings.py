import sys
import numpy as np
import torch
import torch.nn.functional as F
from cdgm_textgen.modules.riemannian_clustering import cluster_embeddings_riemannian

if __name__ == "__main__":
    embeddings_file = sys.argv[1]
    dim = int(sys.argv[2])
    n_clusters = int(sys.argv[3])
    vocab_size = int(sys.argv[4])

    wrd2idx = {}
    idx2wrd = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb_mat_norms = torch.empty((vocab_size, dim), device=device)
    i = 0

    with open(embeddings_file) as ef:
        for line in ef:
            line = line.strip()
            tok, vals = line.split()[0], line.split()[1:]
            wrd2idx[tok] = i
            idx2wrd[i] = tok
            tens = torch.tensor([float(v) for v in vals])
            emb_mat_norms[i] = F.normalize(tens, dim=0)
            i += 1

    print("Loaded embeddings for {} tokens".format(len(wrd2idx)))

    centroids = cluster_embeddings_riemannian(emb_mat_norms, n_clusters)
    np.savetxt(f"{n_clusters}_centroids_dim{dim}.txt", centroids.numpy())