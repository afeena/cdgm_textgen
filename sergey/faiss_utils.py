import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import Tuple
import faiss
from argparse import Namespace
from tqdm.auto import tqdm
import torch

def average_k_neares_neighbour_distance(self, n=1000, k=10, use_faiss_distances=False, speedtest=False, batch_size=64):
    """
    Return the average distance of the k nearest neighbors for each vector in the dataset.
    """
    def batchify(iterable, batch_size):
        # Helper function to batchify an iterable
        for i in range(0, len(iterable), batch_size):
            yield iterable[i:i + batch_size]
    
    # Returns the average distance of the k nearest neighbors for each vector in the dataset
    n_embeddings = self.knn_index.ntotal
    n_qs = np.random.choice(n_embeddings, n, replace=False)
    Ds = defaultdict(list) # nearest-neighbour distances
    for q in tqdm(batchify(n_qs, batch_size), desc="Calculating distances", total=len(n_qs)//batch_size):
        D, I = self.retrieve(self.keys[q], k)
        if speedtest:
            continue
        # D: distances of the k nearest neighbors
        if use_faiss_distances:
            for i in range(k):
                for j in range(len(q)):
                    Ds[i].append(D[j][i])
        else:
            # calculate distances using the original vectors
            for i in range(k):
                for j in range(len(q)):
                    # I[j][i] is the index of the i-th nearest neighbor of the j-th query vector
                    # self.keys[I[j][i]] is the embedding of the i-th nearest neighbor
                    # self.keys[q] is the embedding of the query vector
                    Ds[i].append(np.linalg.norm(self.keys[q] - self.keys[I[j][i]]))
    return Ds