from collections import defaultdict
from pathlib import Path
from typing import Tuple
import faiss
import numpy as np
from argparse import Namespace
from tqdm.auto import tqdm

import torch

def load_yaml(yaml_path):
    import yaml
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

class Embedding:
    def __init__(self, base_path, features_path, targets_path, yaml_path, faiss_path, normalize=False):
        self.features_path = Path(base_path) / features_path
        self.targets_path = Path(base_path) / targets_path
        self.faiss_path = Path(base_path) / faiss_path
        self.args = Namespace(**load_yaml(Path(base_path) / yaml_path))
        print(self.args)
        self.keys = None
        self.targets = None
        self.knn_index = None
        # self.vals = None

        self.normalize = normalize

    def dim(self):
        # Returns the dimension of the embeddings
        if self.keys is None:
            self.load_raw()
        return self.keys.shape[1]
    
    def sample(self, n=1000):
        # Returns a random sample of n embeddings
        if self.keys is None:
            self.load_raw()
        if n > self.keys.shape[0]:
            n = self.keys.shape[0]
        return self.keys[np.random.choice(self.keys.shape[0], n, replace=False)]
    
    def load_raw(self):
        # /ivi/ilps/projects/cnmt/knn-mt-dispersion/datastores/model_512_no_dispersion/train.features.npy
        # large file containing the raw embeddings, need to be loaded into memory with memory mapping
        self.keys = np.memmap(self.features_path, dtype=np.float32, mode='r', shape=(self.args.ntotal, self.args.dim))
        # self.vals = np.memmap(self.args.values_path, dtype=np.int32, mode='r', shape=(self.args.ntotal,)) 
        self.targets = np.memmap(self.targets_path, dtype=np.int32, mode='r', shape=(self.args.ntotal,))

    def load_faiss(self, gpu=False):
        if gpu:
            res = faiss.StandardGpuResources()
            co = faiss.GpuClonerOptions()
            co.useFloat16 = False
        self.knn_index: faiss.Index = faiss.read_index(str(self.faiss_path))
        if gpu:
            self.knn_index = faiss.index_cpu_to_gpu(res,0,self.knn_index,co)
        self.knn_index.nprobe = self.args.nprobe
        self.knn_k = self.args.nclusters
        # self.knn_lambda = self.args.knn_lambda  # knn mt
        # self.knn_temperature = self.args.knn_temperature  # knn mt
        self.knn_metric = "L2"
        # print("load vals to retrieve vocab indices")
        # vals_memmap = np.memmap(f"{self.knn_index_file}_vals.npy", dtype=np.int32, mode='r', shape=(self.knn_index.ntotal,))
        # self.vals = torch.zeros(self.knn_index.ntotal, dtype=torch.int64)
        # self.vals[:] = torch.from_numpy(vals_memmap[:])

    def retrieve(self, q, k=1):
        # Returns distances and indices of the k nearest neighbors for each query vector q
        if q.ndim == 1:
            q = np.expand_dims(q, axis=0)
        # q to fp16
        q = q.astype(np.float32)
        # q: query vector
        # k: number of nearest neighbors to retrieve
        if self.knn_index is None:
            self.load_faiss()
        D, I = self.knn_index.search(q, k)  # distances and indices of the nearest neighbors
        return D, I
    
    def average_k_neares_neighbour_distance(self, n=1000, k=10, use_faiss_distances=False, speedtest=False, batch_size=64):
        def batchify(iterable, batch_size):
            # Helper function to batchify an iterable
            for i in range(0, len(iterable), batch_size):
                yield iterable[i:i + batch_size]
        
        # Returns the average distance of the k nearest neighbors for each vector in the dataset
        if self.knn_index is None:
            self.load_faiss()
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

    def target(self, idx):
        target = self.targets[idx]
        return target
    
    def emb(self, idx, targets=False):
        if self.normalize:
            return self.keys[idx] / np.linalg.norm(self.keys[idx])
        else:
            return self.keys[idx]



def main():
    embedding_baseline = Embedding(
        base_path='/ivi/ilps/projects/cnmt/knn-mt-dispersion/datastores/model_512_no_dispersion/',
        features_path='train.features.npy',
        yaml_path="410.roen.faiss_L2_IVFPQindex_cs64_nprobe32_d512_nc4096_metadata.yaml",
        faiss_path='410.roen.faiss_L2_IVFPQindex_cs64_nprobe32_d512_nc4096'
    )
    embedding_dispersion = Embedding(
        base_path='/ivi/ilps/projects/cnmt/knn-mt-dispersion/datastores/model_512_dispersion/',
        features_path='582.train.features.npy',
        yaml_path="582.roen.faiss_L2_IVFPQindex_cs64_nprobe32_d512_nc2048_meta.yaml",
        faiss_path='582.roen.faiss_L2_IVFPQindex_cs64_nprobe32_512_4096'
    )
    embedding_baseline.load_raw()
    embedding_dispersion.load_raw()

    if torch.cuda.is_available():
        gpu=True
    else:
        gpu=False

    gpu = False
    print(f"Loading FAISS index on {'GPU' if gpu else 'CPU'}")
    embedding_baseline.load_faiss(gpu=gpu)
    embedding_dispersion.load_faiss(gpu=gpu)

    print(embedding_dispersion.knn_index.nlist)
    print(embedding_baseline.knn_index.nlist)

    Ds_baseline = embedding_baseline.average_k_neares_neighbour_distance(n=1000, k=10, use_faiss_distances=True, speedtest=True)
    Ds_dispersion = embedding_dispersion.average_k_neares_neighbour_distance(n=1000, k=10, use_faiss_distances=True, speedtest=True)

    print(Ds_baseline)
    print(Ds_dispersion)


if __name__ == "__main__":
    main()
