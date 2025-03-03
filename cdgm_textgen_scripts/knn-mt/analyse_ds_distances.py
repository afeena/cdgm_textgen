import faiss.contrib.torch_utils
import numpy as np
import argparse
import pandas as pd
import torch
from datetime import datetime

def add_to_bins(x, bins, out):
    idx = np.digitize(x.cpu().numpy(), bins, right=True)
    out[idx] += 1

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
    sample_ids = torch.load(args.sample_ids)

    n_samples = sample_ids.size(0)
    tvals = torch.zeros((index.ntotal,), dtype=torch.int64, device='cuda')
    tvals[:] = torch.from_numpy(vals_memmap[:])

    tkeys = torch.zeros((n_samples, args.dim), dtype=torch.float32, device='cuda')
    tkeys[:] = torch.from_numpy(keys_memmap[sample_ids, :])
    tkeys = tkeys.to(device='cuda')

    gold_samples_targets = tvals[sample_ids]

    k = 8
    start = 0
    num_to_retrieve = 1000
    min_dists_batch = []

    samples_min_dist_to_correct = torch.zeros(n_samples, dtype=torch.float32)
    samples_correct_count = torch.zeros(n_samples, dtype=torch.int32)
    samples_min_dist_to_other = torch.zeros(n_samples, dtype=torch.float32)
    samples_min_dist_to_other_count = torch.zeros(n_samples, dtype=torch.int32)
    samples_unique_count = torch.zeros(n_samples, dtype=torch.int32)
    dist_to_first_nn = torch.zeros(n_samples, dtype=torch.float32)
    dist_to_fith_nn = torch.zeros(n_samples, dtype=torch.float32)
    sample_norms = torch.zeros(n_samples, dtype=torch.float32)
    predicted_1st_nn = torch.zeros(n_samples, dtype=torch.int32)
    predicted_5th_nn = torch.zeros(n_samples, dtype=torch.int32)

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

        # check min dist correct target(s) excluding the 0th one

        dist_to_first_nn[start:end] = dist[:,1].cpu()
        dist_to_fith_nn[start:end] = dist[:,5].cpu()

        predicted_targets = tvals[I]
        predicted_1st_nn[start:end] = predicted_targets[:,1].cpu()
        predicted_5th_nn[start:end] = predicted_targets[:,5].cpu()

        for i in range(start, end):
            samples_unique_count[i] = torch.unique(predicted_targets[i]).shape[0]

        correct_targets_mask = torch.eq(predicted_targets, gold_targets)
        samples_correct_count[start:end] = correct_targets_mask.sum(dim=1).cpu()
        samples_min_dist_to_correct[start:end] = (torch.min(
            (dist.masked_select(correct_targets_mask)),dim=-1)[0]).cpu()

        # check min dist to other targets
        samples_min_dist_to_other_count[start:end] = (k - samples_correct_count[start:end]).cpu()
        samples_min_dist_to_other[start:end] = ((torch.min(dist.masked_select(~correct_targets_mask),dim=-1))[0]).cpu()

        #key norms
        sample_norms[start:end] = torch.norm(tkeys[start:end], dim=-1).cpu()

        min_dist_row = torch.min(dist[:,1:], dim=1)[0]

        min_dists_batch.append(torch.min(min_dist_row).item())
        print(f"Batch {start} - {end} min dist: {torch.min(min_dist_row).item()}")
        start += num_to_retrieve

    res = {
            "target": gold_samples_targets.cpu().numpy(),
           "samples_correct_count": samples_correct_count.cpu().numpy(),
           "samples_min_dist_to_correct": samples_min_dist_to_correct.cpu().numpy(),
           "samples_others_count": samples_min_dist_to_other_count.cpu().numpy(),
            "samples_min_dist_to_other": samples_min_dist_to_other.cpu().numpy(),
            "dist_to_first_nn": dist_to_first_nn.cpu().numpy(),
            "dist_to_fith_nn": dist_to_fith_nn.cpu().numpy(),
            "sample_norms": sample_norms.cpu().numpy(),
            "samples_unique_count": samples_unique_count.cpu().numpy(),
        "predicted 1st": predicted_1st_nn.cpu().numpy(),
        "predicted 5th": predicted_5th_nn.cpu().numpy()
    }

    if args.out_frame is None:
        datesig = datetime.today().strftime("%Y%m%d_%H%m")
        args.out_frame = f"{args.faiss_index}_samples_stats_{n_samples}_{datesig}.csv"
    pd.DataFrame(res).to_csv(f"{args.out_frame}", index=False)
    print(f"Min dist: {np.min(min_dists_batch)}")
    print(f"Norms mean: {sample_norms.mean().item()} max: {sample_norms.max().item()} min: {sample_norms.min().item()}")
    print(f"Dist to first nn mean: {dist_to_first_nn.mean().item()} max: {dist_to_first_nn.max().item()} min: {dist_to_first_nn.min().item()}")
    print(f"Dist to fith nn mean: {dist_to_fith_nn.mean().item()} max: {dist_to_fith_nn.max().item()} min: {dist_to_fith_nn.min().item()}")










