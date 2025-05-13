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
    def __init__(self, name, base_path, features_path, targets_path, yaml_path, faiss_path):
        self.name = name
        self.features_path = Path(base_path) / features_path
        self.targets_path = Path(base_path) / targets_path
        self.faiss_path = Path(base_path) / faiss_path
        self.args = Namespace(**load_yaml(Path(base_path) / yaml_path))
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
        self.load_faiss(self.device.type == 'cuda')

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

