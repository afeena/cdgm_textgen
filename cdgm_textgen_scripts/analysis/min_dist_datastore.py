import faiss
import numpy as np
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--datastore', type=str, required=True)
argparser.add_argument('--features', type=str, required=True)

args = argparser.parse_args()

datastore = faiss.read_index(args.datastore)
min_dists = []
with open(args.features, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        features = np.array([float(x) for x in line.split()])
        idx, dists = datastore.search(features.reshape(1, -1), 100)
        min_dists.append(dists[-1])

min_dist_total = np.array(min_dists).min()
print(min_dist_total)