import sys
import numpy as np
import torch
import torch.nn.functional as F
from cdgm_textgen.modules.riemannian_clustering import cluster_embeddings_riemannian

if __name__ == "__main__":
    embeddings_file = sys.argv[1]
    n_clusters = int(sys.argv[2])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb_mat_norms = torch.load(embeddings_file, map_location=device)
    #don't use pad and bos tokens
    emb_mat_norms = F.normalize(emb_mat_norms[2:,:], dim=-1)
    dim = emb_mat_norms.shape[-1]

    print("Loaded embeddings for {} tokens".format(emb_mat_norms.shape[0]))

    centroids = cluster_embeddings_riemannian(emb_mat_norms, n_clusters)
    np.savetxt(f"{n_clusters}_centroids_dim{dim}.txt", centroids.numpy())