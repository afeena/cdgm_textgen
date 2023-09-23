import sys

import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np
from utils import txt_to_pt, read_mt_vocab
import torch

def global_similarity(emb, label=""):
    sim_mat = cosine_similarity(emb, emb)
    plot_emb_sims(sim_mat, 20, f"{label} global")


def local_similarity(emb, k, label=""):
    """plot similarities for K nearest"""
    sim_mat = cosine_similarity(emb, emb)

    idx = np.argsort(sim_mat, axis=-1)[:, -k:]
    top_k_sim = np.take(sim_mat, indices=idx)
    plot_emb_sims(top_k_sim, 20, f"{label} local top {k}")

def global_arc_distance(emb, label=""):
    emb = F.normalize(emb, p=2, dim=-1)
    arc_distance = torch.arccos(emb @ emb.T) / torch.pi
    plot_emb_arcdist(arc_distance.numpy(), 20, name=f"{label} global arc distance")

def local_arc_distance(emb, k, label=""):
    emb = F.normalize(emb, p=2, dim=-1)
    arc_distance = (torch.arccos(emb @ emb.T) / torch.pi).numpy()
    idx = np.argsort(arc_distance, axis=-1)
    top_k_sim = np.take(arc_distance, indices=idx)

    plot_emb_arcdist(top_k_sim, 20, name=f"{label} local arc distance")


def plot_emb_arcdist(sims, samples, name=""):
    num_emb = sims.shape[0]
    first_samples = np.arange(0, samples)
    middle_samples = np.arange(num_emb // 2 - samples // 2, num_emb // 2 + samples // 2)
    last_samples = np.arange(num_emb - samples // 2, num_emb)
    samples_ind = np.concatenate((first_samples, middle_samples, last_samples))

    sims = sims[samples_ind]

    mean_sims = np.mean(sims, axis=-1)
    max_sims = np.max(sims, axis=-1)
    min_sims = np.min(sims, axis=-1)

    fig, ax = plt.subplots(nrows=3, figsize=(10,10))
    fig.tight_layout(pad=5.0)
    ind = np.arange(0, mean_sims.shape[0]*2,2)
    ax[0].scatter(ind, mean_sims)
    ax[1].scatter(ind, max_sims)
    ax[2].scatter(ind, min_sims)
    plots_titles = [f"{name}_mean arcdist", f"{name}_max arcdist", f"{name}_min arcdist"]
    for i in range(3):
        ax[i].set(xlim=(0, ind[-1]+1),
                  xticks=ind,
                  xticklabels=samples_ind,
                  ylabel='arc distance similarity',
                  xlabel='vocab index')
        # Rotate the tick labels and set their alignment.
        plt.setp(ax[i].get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        ax[i].set_title(plots_titles[i])
    plt.savefig(f"{name}_arcdist.png")
def plot_emb_sims(sims, samples, name=""):
    num_emb = sims.shape[0]
    first_samples = np.arange(0, samples)
    middle_samples = np.arange(num_emb // 2 - samples // 2, num_emb // 2 + samples // 2)
    last_samples = np.arange(num_emb - samples // 2, num_emb)
    samples_ind = np.concatenate((first_samples, middle_samples, last_samples))

    sims = sims[samples_ind]

    mean_sims = np.mean(sims, axis=-1)
    max_sims = np.max(sims, axis=-1)
    min_sims = np.min(sims, axis=-1)

    fig, ax = plt.subplots(nrows=3, figsize=(10,10))
    fig.tight_layout(pad=5.0)
    ind = np.arange(0, mean_sims.shape[0]*2,2)
    ax[0].scatter(ind, mean_sims)
    ax[1].scatter(ind, max_sims)
    ax[2].scatter(ind, min_sims)
    plots_titles = [f"{name}_mean cosine similarity", f"{name}_max cosine similarity", f"{name}_min cosine similarity"]
    for i in range(3):
        ax[i].set(xlim=(0, ind[-1]+1), ylim=(-1.2, 1.2),
                  xticks=ind,
                  xticklabels=samples_ind,
                  yticks=np.arange(-1.2,1.2,0.2),
                  ylabel='cosine similarity',
                  xlabel='vocab index')
        # Rotate the tick labels and set their alignment.
        plt.setp(ax[i].get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        ax[i].set_title(plots_titles[i])
    plt.savefig(f"{name}_cosine_similarity.png")

def iso_score(emb):
    samples = 5000
    num_emb = emb.shape[0]
    from IsoScore import IsoScore
    first_samples = np.arange(0, samples)
    middle_samples = np.arange(num_emb // 2 - samples // 2, num_emb // 2 + samples // 2)
    last_samples = np.arange(num_emb - samples // 2, num_emb)
    full = IsoScore.IsoScore(emb)
    first_score = IsoScore.IsoScore(emb[first_samples])
    middle_score = IsoScore.IsoScore(emb[middle_samples])
    last_score = IsoScore.IsoScore(emb[last_samples])
    print(full, f"high freq score (first 5k): {first_score}, average frequency score (middle 5k): {middle_score}, low "
          f"frequency score (last 5k): {last_score}")
def distr(emb):
    sim_mat = np.triu(cosine_similarity(emb, emb), k=1).ravel()
    res = plt.hist(sim_mat, bins=40, range=(-1, 1))
    plt.show()

def index_average_sim_dist(emb, idx):
    sim_mat = cosine_similarity(emb, emb)
    sim_vec = sim_mat[idx]
    print(np.mean(sim_vec))
    plt.plot(np.arange(len(sim_vec)),sim_vec)
    plt.show()

if __name__ == "__main__":
    vocab, emb = txt_to_pt(sys.argv[1])

    label = "_".join(sys.argv[1].rsplit("/")[-1].split(".")[:-1])
    print(label)
   # global_similarity(emb, label)
   # local_similarity(emb, 5, label)
    #global_arc_distance(emb, label=label)
    #local_arc_distance(emb, 5, label=label)
    #distr(emb)
    #iso_score(emb)

    special_tokens = ["<s>", "<pad>", "</s>", "<unk>"]
    for x in [0,1,2,3]:
        print(special_tokens[x])
        index_average_sim_dist(emb, x)
