import faiss.contrib.torch_utils
import numpy as np
import argparse
import os
import pandas as pd
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, required=True, default=512)
    parser.add_argument('--faiss_index', type=str, required=True)
    parser.add_argument('--keys', type=str, required=True)
    parser.add_argument('--vals', type=str, required=True)
    parser.add_argument('--sample_ids', type=str)
    parser.add_argument('--out_frame', type=str, default=None)
    parser.add_argument('--distance', type=str, default='l2')

    args = parser.parse_args()

    res = faiss.StandardGpuResources()
    co = faiss.GpuClonerOptions()
    co.useFloat16 = True
    index_cpu = faiss.read_index(args.faiss_index)
    index = faiss.index_cpu_to_gpu(res, 0, index_cpu, co)

    print("load vals to retrieve vocab indices")
    vals_memmap = np.memmap(f"{args.faiss_index}_vals.npy", dtype=np.int32, mode='r', shape=(index.ntotal,))
    keys_memmap = np.memmap(f"{args.keys}", dtype=np.float32, mode='r', shape=(index.ntotal,args.dim))
    sample_ids = torch.load(args.sample_ids).to(device='cuda')

    n_samples = sample_ids.size(0)
    tvals = torch.zeros((index.ntotal,), dtype=torch.int64, device='cuda')
    tvals[:] = torch.from_numpy(vals_memmap[:])

    tkeys = torch.zeros((n_samples, args.dim), dtype=torch.float32)
    tkeys[:] = torch.from_numpy(keys_memmap[sample_ids, :]).to(device='cuda')

    gold_samples_targets = tvals[sample_ids]

    k = 8
    start = 0
    num_to_retrieve = 1000
    min_dists_batch = []

    samples_avg_dist_to_correct = torch.zeros(n_samples, dtype=torch.float32, device='cuda')
    samples_correct_count = torch.zeros(n_samples, dtype=torch.int32, device='cuda')
    samples_avg_dist_to_other = torch.zeros(n_samples, dtype=torch.float32, device='cuda')
    samples_min_dist_to_other_count = torch.zeros(n_samples, dtype=torch.int32, device='cuda')

    while start < n_samples:
        end = min(gold_samples_targets.size(0), start + num_to_retrieve)
        gold_targets = gold_samples_targets[start:end].unsqueeze(1)
        
        if args.distance == 'acos':
            tkeys[start:end] = tkeys[start:end] / tkeys[start:end].norm(dim=-1, keepdim=True)
        dist, I = index.search(tkeys[start:end], k)
        if torch.numel(dist) == 0:
            break
        if args.distance == 'acos':
            print("make sure you use index with IP coarse quantizer")
            dist = torch.acos(dist.clamp(-1 + 1e-7, 1 - 1e-7))
        # check avg dist correct target(s)
        predicted_targets = tvals[I]
        correct_targets_mask = torch.eq(predicted_targets, gold_targets).to(device='cuda')
        samples_correct_count[start:end] = correct_targets_mask.sum(dim=1).to(device='cuda')
        samples_avg_dist_to_correct[start:end] = torch.mean(dist.masked_select(correct_targets_mask)).to(device='cuda')

        # check avg dist to other targets
        samples_min_dist_to_other_count[start:end] = (k - samples_correct_count[start:end]).to(device='cuda')
        samples_avg_dist_to_other[start:end] = (torch.mean(dist.masked_select(~correct_targets_mask))).to(device='cuda')
        
        average_row = torch.mean(dist, dim=1)
        min_dist = torch.min(average_row)
        min_dists_batch.append(min_dist.item())
        print(f"Batch {start} - {end} min dist: {min_dist.item()}")
        start += num_to_retrieve

    print(samples_correct_count.shape, samples_avg_dist_to_correct.shape, samples_min_dist_to_other_count.shape, gold_samples_targets.shape)
    res = {
            "target": gold_samples_targets.cpu().numpy(),
           "samples_correct_count": samples_correct_count.cpu().numpy(),
           "samples_avg_dist_to_correct": samples_avg_dist_to_correct.cpu().numpy(),
           "samples_others_count": samples_min_dist_to_other_count.cpu().numpy(),
            "samples_avg_dist_to_other": samples_avg_dist_to_other.cpu().numpy(),

    }
    if args.out_frame is None:
        args.out_frame = f"{args.faiss_index}_samples_distances.csv"
    pd.DataFrame(res).to_csv(f"{args.out_frame}", index=False)
    print(f"Mean min dist: {np.mean(min_dists_batch)}")





