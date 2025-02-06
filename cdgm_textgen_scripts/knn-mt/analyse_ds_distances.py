import faiss.contrib.torch_utils
import numpy as np
import argparse
import os
import pandas as pd
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_size', type=int, required=True, default=100000)
    parser.add_argument('--faiss_index', type=str, required=True)
    parser.add_argument('--keys', type=str, required=True)
    parser.add_argument('--vals', type=str, required=True)
    parser.add_argument('--sample_ids', type=str)
    parser.add_argument('--out_frame', type=str, default=None)
    parser.add_argument('--vocab_size', type=int, default=10000)
    parser.add_argument('--distance', type=str, default='l2')

    args = parser.parse_args()

    res = faiss.StandardGpuResources()
    co = faiss.GpuClonerOptions()
    co.useFloat16 = True
    index_cpu = faiss.read_index(args.faiss_index)
    index = faiss.index_cpu_to_gpu(res, 0, index_cpu, co)

    vals = np.load(args.vals)
    tvals = torch.from_numpy(vals).to(dtype=torch.int64)
    del vals

    keys = np.load(args.keys)

    tkeys = torch.from_numpy(keys).to(dtype=torch.float32)
    del keys

    if not os.path.exists(args.sample_ids):
        #sample random entry for each target token in the vocabulary
        sample_ids = torch.full(args.sample_size, -1, dtype=torch.int64)
        all_vocab_idx = torch.arange(0,args.vocab_size, dtype=torch.int64)
        sample_count = 0
        for vocab_idx in all_vocab_idx:
            tmp = (tvals.flatten()==vocab_idx).nonzero().squeeze(1)
            if len(tmp.size())==0:
                sample_ids[vocab_idx]=-1
                continue
            num_per_vocab_idx = min(64, tmp.size(0))
            sample_ids[sample_count:sample_count+num_per_vocab_idx] = tmp[torch.randperm(tmp.size(0))[:num_per_vocab_idx]]
            sample_count += num_per_vocab_idx
        sample_ids = sample_ids[sample_ids!=-1]
        torch.save(sample_ids, args.sample_ids)
    else:
        sample_ids = torch.load(args.sample_ids, map_location='cpu')
    print(sample_ids.shape)
    tkeys = tkeys[sample_ids].to(device='cuda')
    gold_samples_targets = tvals[sample_ids]



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
        dist, I = index.search(tkeys[start:end], k)

        if args.distance == 'acos':
            print("make sure you use index with IP coarse quantizer")
            dist = torch.acos(dist.clamp(-1 + 1e-7, 1 - 1e-7)).cpu()
        # check avg dist correct target(s)
        predicted_targets = tvals[I.cpu()]
        correct_targets_mask = torch.eq(predicted_targets, gold_targets)
        samples_correct_count[start:end] = correct_targets_mask.sum(dim=1)
        samples_avg_dist_to_correct[start:end] = torch.mean(dist[correct_targets_mask.nonzero()],dim=1)

        # check avg dist to other targets
        samples_min_dist_to_other_count[start:end] = k - samples_correct_count[start:end]
        samples_avg_dist_to_other[start:end] = torch.mean(dist[~correct_targets_mask],dim=1)

        average_row = torch.mean(dist, dim=1)
        min_dist = torch.min(average_row)
        min_dists_batch.append(min_dist.item())
        print(f"Batch {start} - {end} min dist: {min_dist.item()}")
        start += num_to_retrieve


    res = {
            "target": gold_samples_targets.cpu().numpy(),
           "samples_correct_count": samples_correct_count.cpu().numpy(),
           "samples_avg_dist_to_correct": samples_avg_dist_to_correct.cpu().numpy(),
           "samples_others_count": samples_min_dist_to_other_count.cpu().numpy(),
            "samples_avg_dist_to_other": samples_avg_dist_to_other.cpu().numpy(),

    }
    if args.out_frame is None:
        args.out_frame = f"distances_{args.faiss_index}_samples_distances.csv"
    pd.DataFrame(res).to_csv(f"{args.out_frame}", index=False)
    print(f"Mean min dist: {np.mean(min_dists_batch)}")





