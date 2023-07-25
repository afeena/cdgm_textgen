import sys

from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np

from utils import txt_to_pt

def global_similarity(emb):

    sim_mat = cosine_similarity(emb,emb)
    mean_sims = np.mean(sim_mat, axis=-1)
    max_sims = np.max(sim_mat, axis=-1)
    min_sims = np.min(sim_mat, axis=-1)

    fig, ax = plt.subplots(nrows=3)
    ind = np.arange(0, mean_sims.shape[0])
    ax[0].scatter(ind, mean_sims)
    ax[1].scatter(ind, max_sims)
    ax[2].scatter(ind, min_sims)
    plt.show()

def local_similarity(emb, k):
    """plot similarities for K nearest"""
    sim_mat = cosine_similarity(emb, emb)
    idx = np.argsort(sim_mat, axis=-1)[:,-k:]
    top_k_sim = np.take(sim_mat, indices=idx)
    print(top_k_sim)

    mean_sims = np.mean(top_k_sim, axis=-1)
    max_sims = np.max(top_k_sim, axis=-1)
    min_sims = np.min(top_k_sim, axis=-1)

    fig, ax = plt.subplots(nrows=3)
    ind = np.arange(0, top_k_sim.shape[0])
    ax[0].scatter(ind, mean_sims)
    ax[1].scatter(ind, max_sims)
    ax[2].scatter(ind, min_sims)
    plt.show()

def distr(emb):
    sim_mat = np.triu(cosine_similarity(emb, emb), k=1).ravel()
    res = plt.hist(sim_mat, bins=40, range=(-1,1))
    plt.show()

if __name__ == "__main__":
    vocab, emb = txt_to_pt(sys.argv[1])
    #global_similarity(emb)
    #local_similarity(emb, 5)
    distr(emb)