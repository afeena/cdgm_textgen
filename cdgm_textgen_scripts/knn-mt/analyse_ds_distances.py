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

    keys = np.memmap(args.keys_path, dtype=np.float32, mode='r',
                     shape=(args.dstore_size, args.dim))
    random_sample = np.random.choice(np.arange(keys.shape[0]), size=[args.samle_size], replace=False)
    tkeys = torch.zeros((args.sample_size, 1), dtype=torch.float32, device='cuda')
    tkeys = torch.from_numpy(keys[random_sample])

    args = parser.parse_args()

    res = faiss.StandardGpuResources()
    co = faiss.GpuClonerOptions()
    co.useFloat16 = True
    index_cpu = faiss.read_index(args.faiss_index)
    index = faiss.index_cpu_to_gpu(res, 0, index_cpu, co)

    all_ids = torch.arange(index.ntotal, device='cuda')
    perm = torch.randperm(all_ids.size(0))
    idx = perm[:args.sample_size]


    k = 32

    cosine_sim, I = index.search(tkeys, k)
    acos_dist = torch.acos(cosine_sim)
    average_row = torch.mean(acos_dist, dim=1)
    min_dist = torch.min(average_row)

    print(f'Minimum distance: {min_dist.item()}')
    print(f'Average distance: {average_row.item()}')





