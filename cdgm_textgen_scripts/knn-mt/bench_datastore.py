import time
import faiss.contrib.torch_utils
import torch
import sys
import numpy as np

index = sys.argv[1]
features = sys.argv[2]

k = 8

keys = np.memmap(features, dtype=np.float32, mode='r', shape=(10000000, 1024))


res = faiss.StandardGpuResources()
co = faiss.GpuClonerOptions()
co.useFloat16 = True
knn_index = faiss.read_index(index)
knn_index = faiss.index_cpu_to_gpu(res, 0, knn_index, co)


nqueries = 100000
query_idx = np.sort(np.random.choice(np.arange(keys.shape[0]), size=[nqueries], replace=False))

times = []
for i in range(query_idx):
    query = torch.tensor(keys[i], device='cuda')
    start = time.time()
    knn_index.search(query, k)
    times.append(time.time() - start)

print(index)
print("Average time for index: ", np.mean(times))





