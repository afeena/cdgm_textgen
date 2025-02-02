import faiss.contrib.torch_utils
import numpy as np
import argparse
import os
import pandas as pd
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_size', type=int, required=True, default=1000)
    parser.add_argument('--faiss_index', type=str, required=True)
    parser.add_argument('--keys', type=str, required=True)
    parser.add_argument('--vals', type=str, required=True)
    parser.add_argument('--sample_ids', type=str)
    parser.add_argument('--out_frame', type=str, default=None)
    parser.add_argument('--vocab_size', type=int, default=10000)

    args = parser.parse_args()

    vals = np.memmap(args.vals, dtype=np.int32, mode='r', shape=(21507934, 1))
    tvals = torch.from_numpy(vals[:]).to(device='cuda', dtype=torch.int32)

    keys = np.memmap(args.keys, dtype=np.float32, mode='r',shape=(21507934, 128))
    tkeys = torch.from_numpy(keys[:]).to(device='cuda', dtype=torch.float32)

    if not os.path.exists(args.sample_ids):
        #sample random entry for each target token in the vocabulary
        sample_ids = torch.zeros(args.vocab_size, dtype=torch.int32, device='cuda')
        for etfc in range(args.vocab_size):
            tmp = tvals.eq(etfc).squeeze(1).nonzero()
            sample_ids[etfc] = tmp[torch.randperm(tmp.size(0))[0]]
        torch.save(sample_ids, args.sample_ids)
    else:
        sample_ids = torch.load(args.sample_ids)

    tkeys = tkeys[sample_ids]
    gold_samples_targets = tvals[sample_ids]

    del keys

    args = parser.parse_args()

    res = faiss.StandardGpuResources()
    co = faiss.GpuClonerOptions()
    co.useFloat16 = True
    index_cpu = faiss.read_index(args.faiss_index)
    index = faiss.index_cpu_to_gpu(res, 0, index_cpu, co)

    k = 8

    start = 0
    num_to_retrieve = 1000
    min_dists_batch = []
    samples_avg_dist_to_correct = torch.zeros(args.sample_size, dtype=torch.float32, device='cuda')
    samples_correct_count = torch.zeros(args.sample_size, dtype=torch.int32, device='cuda')
    samples_avg_dist_to_other = torch.zeros(args.sample_size, dtype=torch.float32, device='cuda')
    samples_min_dist_to_other_count = torch.zeros(args.sample_size, dtype=torch.int32, device='cuda')

    while start < args.sample_size:
        end = min(args.sample_size, start + num_to_retrieve)
        gold_targets = gold_samples_targets[start:end]
        cosine_sim, I = index.search(tkeys[start:end], k)

        acos_dist = torch.acos(cosine_sim.clamp(-1 + 1e-7, 1 - 1e-7))
        # check avg dist correct target(s)
        predicted_targets = tvals[I]
        correct_targets_mask = torch.eq(predicted_targets, gold_targets)
        samples_correct_count[start:end] = correct_targets_mask.sum(dim=1)
        samples_avg_dist_to_correct[start:end] = torch.mean(acos_dist[correct_targets_mask.nonzero()],dim=1)

        # check avg dist to other targets
        samples_min_dist_to_other_count[start:end] = k - samples_correct_count[start:end]
        samples_avg_dist_to_other[start:end] = torch.mean(acos_dist[~correct_targets_mask],dim=1)

        average_row = torch.mean(acos_dist, dim=1)
        min_dist = torch.min(average_row)
        min_dists_batch.append(min_dist.item())
        print(f"Batch {start} - {end} min dist: {min_dist.item()}")
        start += num_to_retrieve


    res = {
           "samples_correct_count": samples_correct_count.cpu().numpy(),
           "samples_avg_dist_to_correct": samples_avg_dist_to_correct.cpu().numpy(),
           "samples_avg_dist_to_other": samples_avg_dist_to_other.cpu().numpy(),
           "samples_others_count": samples_min_dist_to_other_count.cpu().numpy(),
            "target": gold_samples_targets.cpu().numpy(),
    }
    if args.out_frame is None:
        args.out_frame = f"distances_{args.faiss_index}_samples_distances.csv"
    pd.DataFrame(res).to_csv(f"{args.out_frame}")
    print(f"Mean min dist: {np.mean(min_dists_batch)}")





