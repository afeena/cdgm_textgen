import faiss
import numpy as np
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--datastore', type=str, required=True)
argparser.add_argument('--features', type=str, required=True)
argparser.add_argument('--targets', type=str, required=True)
argparser.add_argument('-n', type=int, default=100, help='Number of nearest neighbors to retrieve')
argparser.add_argument('-s', type=str, help="Path to save the min_dists.csv file", default='min_dists.csv')

args = argparser.parse_args()

datastore = faiss.read_index(args.datastore)
min_dists = []
min_dists_unique = []
targets = np.loadtxt(args.targets, dtype=float).astype(np.int64)
t = 0
with open(args.features, 'r') as f:
    for line_fetaure in f:
        line_fetaure = line_fetaure.strip()
        if not line_fetaure:
            continue
        features = np.array([float(x) for x in line_fetaure.split()])
        idx, dists = datastore.search(features.reshape(1, -1), args.n)
        position = np.where(idx!=targets[t])
        min_dists_unique.append(dists[position])
        min_dists.append(dists[1])
        t+=1

import pandas as pd

pd.DataFrame({'min_dist': min_dists, 'min_dist_unique': min_dists_unique, 'target': targets}).to_csv(args.s, index=False)
