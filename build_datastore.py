import argparse
import faiss
import os
import numpy as np
import time

if __name__ == "__main__":
    print("start")
    datastore_parser = argparse.ArgumentParser()
    datastore_parser.add_argument("--keys_path", type=str, required=True)
    datastore_parser.add_argument("--values_path", type=str, required=True)
    datastore_parser.add_argument('--dstore_size', type=int, help='number of items saved in the datastore memmap')
    datastore_parser.add_argument("--faiss_index", type=str, default=None)
    datastore_parser.add_argument("--dim", type=int, default=128)
    datastore_parser.add_argument("--ncentroids", type=int, default=2048)
    datastore_parser.add_argument("--code_size", type=int, default=32)
    datastore_parser.add_argument("--nprobe", type=int, default=8)
    datastore_parser.add_argument("--dry_run", action="store_true")

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

        keys = np.memmap(args.keys_path, dtype=np.float32, mode='r',
                         shape=(args.dstore_size, args.dim))
        vals = np.memmap(args.values_path, dtype=np.int32, mode='r', shape=(args.dstore_size, 1))

    res = faiss.StandardGpuResources()
    config = faiss.GpuIndexFlatConfig()
    config.device = 0
    if not os.path.exists(args.faiss_index + ".trained"):
        print("training faiss index")

        indexa = faiss.IndexFlatIP(args.dim)
        cpu_idx = faiss.IndexIVFPQ(indexa, args.dim,
                                     args.ncentroids,
                                     args.code_size, 8)
        index = faiss.index_cpu_to_gpu(res,0, cpu_idx)
        index.nprobe = args.nprobe
        np.random.seed(42)
        random_sample = np.random.choice(np.arange(vals.shape[0]), size=[min(1000000, vals.shape[0])], replace=False)

        start = time.time()
        index.train(keys[random_sample].astype(np.float32))
        print('Training took {} s'.format(time.time() - start))

        print('Writing index after training')
        start = time.time()
        index_cpu = faiss.index_gpu_to_cpu(index)
        faiss.write_index(index_cpu, args.faiss_index + ".trained")
        print('Writing index took {} s'.format(time.time() - start))

    print('Adding Keys')
    index_cpu = faiss.read_index(args.faiss_index + ".trained")
    index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
    start = 0
    start_time = time.time()
    num_keys_to_add_at_a_time = 1000000
    while start < args.dstore_size:
        end = min(args.dstore_size, start + num_keys_to_add_at_a_time)
        to_add = keys[start:end].copy()
        index.add_with_ids(to_add.astype(np.float32), np.arange(start, end))
        start += num_keys_to_add_at_a_time

        if (start % 1000000) == 0:
            print('Added %d tokens so far' % start)
            print('Writing Index', start)
            faiss.write_index(index, args.faiss_index)

    print("Adding total %d keys" % end)
    print('Adding took {} s'.format(time.time() - start_time))
    print('Writing Index')
    start = time.time()
    index_cpu = faiss.index_gpu_to_cpu(index)
    faiss.write_index(index_cpu, args.faiss_index)
    print('Writing index took {} s'.format(time.time() - start))

    print(f"total {index.ntotal} tokens are saved to a datastore")
