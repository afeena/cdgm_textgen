import faiss.contrib.torch_utils
import numpy as np
import argparse

import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_size', type=int, required=True, default=1000)
    parser.add_argument('--faiss_index', type=str, required=True)
    parser.add_argument('--keys', type=str, required=True)

    args = parser.parse_args()

    keys = np.memmap(args.keys, dtype=np.float32, mode='r',shape=(21507934, 128))
    random_sample = np.random.choice(np.arange(keys.shape[0]), size=[args.sample_size], replace=False)
    tkeys = torch.from_numpy(keys[random_sample]).to(device='cuda', dtype=torch.float32)

    del keys

    args = parser.parse_args()

    res = faiss.StandardGpuResources()
    co = faiss.GpuClonerOptions()
    co.useFloat16 = True
    index_cpu = faiss.read_index(args.faiss_index)
    index = faiss.index_cpu_to_gpu(res, 0, index_cpu, co)

    k = 8

    start = 0
    end = 1000
    min_dists_batch = []
    while True:
        cosine_sim, I = index.search(tkeys, k)
        acos_dist = torch.acos(cosine_sim.clamp(-1 + 1e-7, 1 - 1e-7))
        average_row = torch.mean(acos_dist, dim=1)
        min_dist = torch.min(average_row)
        min_dists_batch.append(min_dist.item())
        print(f"Batch {start} - {end} min dist: {min_dist.item()}")
        start += 1000
        end += 1000
        if end > args.sample_size:
            break

    print(f"Mean min dist: {np.mean(min_dists_batch)}")





