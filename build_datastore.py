import argparse
import torch
import os
import numpy as np
import time
import faiss.contrib.torch_utils
import torch.nn.functional as F

if __name__ == "__main__":
    datastore_parser = argparse.ArgumentParser()
    datastore_parser.add_argument("--keys_path", type=str, required=True)
    datastore_parser.add_argument("--values_path", type=str, required=True)
    datastore_parser.add_argument('--dstore_size', type=int, help='number of items saved in the datastore memmap')
    datastore_parser.add_argument("--faiss_index", type=str, default=None)
    datastore_parser.add_argument("--dim", type=int, default=128)
    datastore_parser.add_argument("--ncentroids", type=int, default=4096)
    datastore_parser.add_argument("--code_size", type=int, default=32)
    datastore_parser.add_argument("--nprobe", type=int, default=128)
    datastore_parser.add_argument("--dry_run", action="store_true")
    datastore_parser.add_argument("--ntrain", type=int, default=1000000)
    datastore_parser.add_argument("--coarse_idx", type=str, default='l2')
    datastore_parser.add_argument("--pca", type=int, default=0, help="number of PCA components to use")

    args = datastore_parser.parse_args()

    print("read keys and vals")
    #here test ds creation
    if args.dry_run:
        args.dstore_size = 1000
        args.dim = 128
        args.faiss_index = "test"
        keys = np.random.rand(args.dstore_size, args.dim).astype(np.float32)
        vals = np.random.randint(0, 1000000, size=[args.dstore_size, 1]).astype(np.int32)

    else:

        keys = np.memmap(args.keys_path, dtype=np.float32, mode='r', shape=(args.dstore_size, args.dim))
        vals = np.memmap(args.values_path, dtype=np.int32, mode='r', shape=(args.dstore_size,))

    res = faiss.StandardGpuResources()
    co = faiss.GpuClonerOptions()
    co.useFloat16 = True
    if args.coarse_idx== 'l2':
        indexa = faiss.IndexFlatL2(args.dim)
    elif args.coarse_idx=='ip':
        indexa = faiss.IndexFlatIP(args.dim)
    cpu_idx = faiss.IndexIVFPQ(indexa, args.dim,
                               args.ncentroids,
                               args.code_size, 8)
    index = faiss.index_cpu_to_gpu(res, 0, cpu_idx, co)
    index.nprobe = args.nprobe

    if args.pca > 0:
        pca_matrix = faiss.PCAMatrix(args.dim, args.pca, 0, True)
        index = faiss.IndexPreTransform(pca_matrix, index)

    if not os.path.exists(args.faiss_index + ".trained"):
        print("create faiss index")

        print("sample vectors for training")
        np.random.seed(42)
        random_sample = np.random.choice(np.arange(vals.shape[0]), size=[min(args.ntrain, vals.shape[0])], replace=False)
        #to be able to search with gpu tensors, we need to build store with gpu tensors
        training_samples = torch.tensor(keys[random_sample], dtype=torch.float32, device='cuda')
        if args.coarse_idx=='ip':
            training_samples = F.normalize(training_samples, p=2, dim=1)

        print(f"train faiss index with {random_sample.shape[0]} samples")
        start = time.time()
        index.train(training_samples)
        print('Training took {} s'.format(time.time() - start))
        assert index.is_trained, "index is not trained"

        print('Writing index after training')
        start = time.time()
        index_cpu = faiss.index_gpu_to_cpu(index)
        faiss.write_index(index_cpu, args.faiss_index + ".trained")
        print('Writing index took {} s'.format(time.time() - start))

    print('Adding Keys')
    index_cpu = faiss.read_index(args.faiss_index + ".trained")
    start = 0
    start_time = time.time()
    num_keys_to_add_at_a_time = 1000000
    while start < args.dstore_size:
        end = min(args.dstore_size, start + num_keys_to_add_at_a_time)
        to_add = torch.tensor(keys[start:end].copy(), device='cpu', dtype=torch.float32)
        if args.coarse_idx=='ip':
            to_add = F.normalize(to_add, p=2, dim=1)
        idx_to_add = torch.arange(start, end, device='cpu', dtype=torch.int64)
        index_cpu.add_with_ids(to_add, idx_to_add)
        start += num_keys_to_add_at_a_time

        if (start % 1000000) == 0:
            print('Added %d tokens so far' % start)
            print('Writing Index', start)
            #index_cpu = faiss.index_gpu_to_cpu(index)
            faiss.write_index(index_cpu, args.faiss_index)

    print("Adding total %d keys" % end)
    print('Adding took {} s'.format(time.time() - start_time))

    print('Writing Index')
    start = time.time()
    #index_cpu = faiss.index_gpu_to_cpu(index)
    faiss.write_index(index_cpu, args.faiss_index)
    print('Writing index took {} s'.format(time.time() - start))

    print(f"total {index.ntotal} tokens are saved to a datastore")
