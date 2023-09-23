import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity

from utils import txt_to_pt


def global_similarity(emb, label=""):
    sim_mat = cosine_similarity(emb, emb)
    plot_sim_dists(sim_mat, 10)


def global_arc_distance(emb, label=""):
    emb = F.normalize(emb, p=2, dim=-1)
    arc_distance = 1 - torch.arccos(emb @ emb.T) / torch.pi
    plot_sim_dists(arc_distance.numpy(), 10, "arc similarity")


def plot_sim_dists(sims, n_samples, ylabel="cosine similarity"):
    # num_emb = sims.shape[0]
    # indices = np.random.choice(np.arane(num_emb),n_samples)

    indices = sorted([0, 1, 3, 158, 1013, 4195, 9644, 5563, 4252, 8643, 13663, 16865, 4504, 23663, 13392, 25000])
    #indices = sorted([27000, 27500, 26547, 24795, 23663])
    n_samples = len(indices)
    fig, ax = plt.subplots(figsize=(10, 10)
                           )

    sims = sims[indices]

    for x in range(n_samples):
        ax.scatter(np.arange(sims.shape[-1]), sims[x, :], label=indices[x], alpha=0.5)
    ax.legend()
    ax.set(xlim=(0, sims.shape[-1]+1),
           xticks=np.arange(0, stop=sims.shape[-1], step=1000, dtype='int'),
           xticklabels=np.arange(0, stop=sims.shape[-1], step=1000, dtype='int'),
           ylabel=ylabel,
           ylim=(-1.1, 1.1),
           yticks=np.arange(-1.1, stop=1.1, step=0.2, dtype='float'),
          # yticklabels=np.arange(0, stop=1, step=0.1, dtype='float'),
           xlabel='vocab index')
    ax.grid()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    vocab, emb = txt_to_pt(sys.argv[1])

    label = "_".join(sys.argv[1].rsplit("/")[-1].split(".")[:-1])
    print(label)
    global_similarity(emb, label)
    #global_arc_distance(emb)
# local_similarity(emb, 5, label)
