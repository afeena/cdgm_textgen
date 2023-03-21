import sys

import numpy as np
import torch
import torch.nn.functional as F
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.learning.kmeans import RiemannianKMeans

embd_file = sys.argv[1]
dim = int(sys.argv[2])
n_clusters = int(sys.argv[3])
wrd2idx = {}
idx2wrd = {}
emb_mat_norms = torch.empty((27513, dim))
i = 0

with open(embd_file) as ef:
    for line in ef:
        line = line.strip()
        tok, vals = line.split()[0], line.split()[1:]
        wrd2idx[tok] = i
        idx2wrd[i] = tok
        tens = torch.tensor([float(v) for v in vals])
        emb_mat_norms[i] = F.normalize(tens, dim=0)
        i += 1

print("Loaded embeddings for {} tokens".format(len(wrd2idx)))

# tens_the = emb_mat[wrd2idx["▁the"]]
# tens_a = emb_mat[wrd2idx["▁a"]]
# tens_for = emb_mat[wrd2idx["▁for"]]
#
#
# print(F.cosine_similarity(tens_the, tens_a, dim=0))
# print(F.cosine_similarity(tens_the, tens_for, dim=0))
#
# dist1 = PowerSpherical(F.normalize(tens_the, dim=0), 1000*torch.norm(tens_the))
# dist2 = PowerSpherical(F.normalize(tens_a, dim=0), 1000*torch.norm(tens_a))
# dist3 = PowerSpherical(F.normalize(tens_for, dim=0), 1000*torch.norm(tens_for))
#
# dist_1_smaples = dist1.sample((5,))
# dist_2_smaples = dist2.sample((5,))
# dist_3_smaples = dist3.sample((5,))
#
# dist12 = F.cosine_similarity(dist_1_smaples, dist_2_smaples)
# dist13 = F.cosine_similarity(dist_1_smaples, dist_3_smaples)
# print(dist12.mean(), dist13.mean())

sphere = Hypersphere(dim=emb_mat_norms.shape[1])
kmeans = RiemannianKMeans(metric=sphere.metric,
                          n_clusters=n_clusters,
                          tol=1e-5,
                          verbose=2)
kmeans.fit(emb_mat_norms)
centroids = kmeans.centroids

np.savetxt(f"{n_clusters}_centroids_dim{dim}.txt", centroids.numpy())

# find nearest neighbor
# neighbor_ix = torch.argsort(centroids @ emb_mat_norms.T, dim=-1, descending=True)
# for ixs in neighbor_ix:
#    print(", ".join(idx2wrd[int(i.item())] for i in ixs[:5]))
