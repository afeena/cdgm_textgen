import argparse
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--ntotal", type=int, required=True)
    parser.add_argument('--keys', type=str, required=True)
    parser.add_argument('--vals', type=str, required=True)
    parser.add_argument('--target_val', type=int, required=True)
    parser.add_argument('--out', type=str)


    args = parser.parse_args()

    print("load vals to retrieve vocab indices")
    vals_memmap = np.memmap(f"{args.faiss_index}_vals.npy", dtype=np.int32, mode='r', shape=(args.ntotal,))
    keys_memmap = np.memmap(f"{args.keys}", dtype=np.float32, mode='r', shape=(args.ntotal,args.dim))

    target_keys_ids = []

    count = 0

    for i in range(args.ntotal):
        if vals_memmap[i]==args.target_val:
            count+=1
            target_keys_ids.append(i)

    target_keys = keys_memmap[target_keys_ids]

    np.save(f"{args.out}_keys_for_val_{args.target_val}.npy", target_keys)




