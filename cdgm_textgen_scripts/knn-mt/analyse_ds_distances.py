import faiss.contrib.torch_utils
import numpy as np
import argparse

import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_size', type=int, required=True, default=1000)
    parser.add_argument('--faiss_index', type=str, required=True)

    args = parser.parse_args()

    res = faiss.StandardGpuResources()
    co = faiss.GpuClonerOptions()
    co.useFloat16 = True
    index_cpu = faiss.read_index(args.faiss_index)
    index = faiss.index_cpu_to_gpu(res, 0, index_cpu, co)

    all_ids = torch.arange(index.ntotal, device='cuda')
    perm = torch.randperm(all_ids.size(0))
    idx = perm[:args.sample_size]

    samples = torch.empty(args.sample_size, index.d, dtype=torch.float32, device='cuda')
    j=0
    for i in idx:
        _, _, sample = index.reconstruct(all_ids[i])
        samples[j] = torch.tensor(sample, dtype=torch.float32, device='cuda')
        j+=1

    k = 32

    cosine_sim, I = index.search(samples, k)
    acos_dist = torch.acos(cosine_sim)
    average_row = torch.mean(acos_dist, dim=1)
    min_dist = torch.min(average_row)

    print(f'Minimum distance: {min_dist.item()}')
    print(f'Average distance: {average_row.item()}')





