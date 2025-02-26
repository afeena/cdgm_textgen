import argparse
import os
import numpy as np
import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_size', type=int, required=True, default=100000)
    parser.add_argument("--dim", type=int, required=True, default=512)
    parser.add_argument('--faiss_index_size', type=int, required=True)
    parser.add_argument('--keys', type=str, required=True)
    parser.add_argument('--vals', type=str, required=True)
    parser.add_argument('--sample_ids', type=str)
    parser.add_argument('--vocab_size', type=int, default=10000)
    parser.add_argument('--max_keys_per_token', type=str, default='l2')

    args = parser.parse_args()

    print("load vals to retrieve vocab indices")
    vals_memmap = np.memmap(f"{args.faiss_index}_vals.npy", dtype=np.int32, mode='r', shape=(args.faiss_index_size,))
    tvals = torch.zeros((args.faiss_index_size), dtype=torch.int64)
    tvals[:] = torch.from_numpy(vals_memmap[:])

    keys_memmap = np.memmap(f"{args.keys}", dtype=np.float32, mode='r', shape=(args.faiss_index_size,args.dim))

    #sample random entry for each target token in the vocabulary
    sample_ids = torch.full((args.sample_size,), -1, dtype=torch.int64)
    all_vocab_idx = torch.arange(0,args.vocab_size, dtype=torch.int64)
    sample_count = 0
    for vocab_idx in all_vocab_idx:
        tmp = (tvals.flatten()==vocab_idx).nonzero().squeeze(1)
        if tmp.size()==torch.Size([0]):
            sample_ids[vocab_idx]=-1
            continue
        num_per_vocab_idx = min(args.max_keys_per_token, tmp.size(0))
        sample_ids[sample_count:sample_count+num_per_vocab_idx] = tmp[torch.randperm(tmp.size(0))[:num_per_vocab_idx]]
        sample_count += num_per_vocab_idx


    sample_ids = sample_ids[sample_ids!=-1]
    final_sample_size = sample_ids.size(0)
    sample_ids_path = f"{args.sample_ids}_{final_sample_size}"
    torch.save(sample_ids, sample_ids_path)

    print(f"created sampels with max {args.max_keys_per_token} keys per token")
    print(f"resulting sample sample size: {final_sample_size}")