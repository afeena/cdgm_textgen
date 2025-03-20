import faiss
import argparse
import numpy as np
import torch

parser = argparse.ArgumentParser(description='Faiss index search')
parser.add_argument('--faiss_index', type=str, required=True)
parser.add_argument("--keys", type=str, required=True)
args = parser.parse_args()

res = faiss.StandardGpuResources()
co = faiss.GpuClonerOptions()
co.useFloat16 = True
index_cpu = faiss.read_index(args.faiss_index)
index = faiss.index_cpu_to_gpu(res, 0, index_cpu, co)

vals_memmap = np.memmap(f"{args.faiss_index}_vals.npy", dtype=np.int32, mode='r', shape=(index.ntotal,))
keys_memmap = np.memmap(f"{args.keys}", dtype=np.float32, mode='r', shape=(index.ntotal,args.dim))


test_id = 5
k = 8
dist, idx = index.search(torch.tensor(keys_memmap[test_id], device='cuda'), k)
print(dist)
print(idx)