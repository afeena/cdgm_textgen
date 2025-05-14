from collections import defaultdict
from pathlib import Path
from typing import Tuple
import faiss
import numpy as np
from argparse import Namespace
from tqdm.auto import tqdm
from itertools import groupby

import torch

def load_yaml(yaml_path):
    import yaml
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

class Embedding:
    def __init__(self, name, base_path, features_path, targets_path, yaml_path=None, faiss_path=None, args=None):
        self.name = name
        self.features_path = Path(base_path) / features_path
        self.targets_path = Path(base_path) / targets_path
        self.faiss_path = Path(base_path) / faiss_path

        if args is not None:
            self.args = Namespace(**args)
        elif yaml_path is not None:
            self.args = Namespace(**load_yaml(Path(base_path) / yaml_path))
        else:
            raise ValueError("Either args or yaml_path must be provided")
        print(self.args)
        self.keys = None
        self.targets = None
        self.knn_index = None
        # self.vals = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    
    def load(self):
        # /ivi/ilps/projects/cnmt/knn-mt-dispersion/datastores/model_512_no_dispersion/train.features.npy
        self.keys = np.memmap(self.features_path, dtype=np.float32, mode='r', shape=(self.args.ntotal, self.args.dim))
        self.targets = np.memmap(self.targets_path, dtype=np.int32, mode='r', shape=(self.args.ntotal,))
        self.sampled_keys = None
        self.sampled_targets = None
        self.load_faiss(self.device.type == 'cuda')

    def cache_features_and_targets(
        self,
        cache_folder: str,
        cache_name='subsampled_features_targets',
        subsample_size: int = 1000000,
        seed: int = 42,
    ):
        """
        saving a subsample of keys/targets for analysis
        """
        np.random.seed(seed)
        cache_path = Path(cache_folder) / f"{cache_name}" / f"{self.name}.npz"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        print("caching features and targets to", cache_path)
        ids = np.random.choice(self.keys.shape[0], subsample_size, replace=False)
        ids.sort() # to read in contiguous chunks

        print("size of subsample / total size", subsample_size / len(self.keys) * 100, "%")
        out_keys = self.keys[ids]
        out_targets = self.targets[ids]
        np.savez(cache_path, keys=out_keys, targets=out_targets)
        print(f"saved subsample of keys and targets to {cache_path}")

    def load_subsampled_features_and_targets(
        self,
        cache_folder: str,
        cache_name='subsampled_features_targets',
    ):
        """
        loading a subsample of keys/targets for analysis
        """
        cache_path = Path(cache_folder) / f"{cache_name}" / f"{self.name}.npz"
        print(f"loading subsample of keys and targets from {cache_path}")
        data = np.load(cache_path)
        self.sampled_keys = data['keys']
        self.sampled_targets = data['targets']
        print("size of all keys/targets", len(self.keys))
        print("size of subsample", len(self.sampled_targets))

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
        D, I = self.knn_index.search(q, k)  # distances and indices of the nearest neighbors
        return D, I

    def target(self, idx):
        target = self.targets[idx]
        return target
    
    def normalize(self, v):
        # Normalize the vector v
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

