import argparse

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity

from utils import txt_to_pt, read_mt_vocab

def build_long_cube(num_emb, embed_dim):
    d = torch.zeros(num_emb, embed_dim)
    new = 0
    while new < num_emb:
        cand = torch.rand(embed_dim)
        cand[cand < 0.5] = -1
        cand[cand >= 0.5] = 1
        if not torch.any(torch.all(d == cand, dim=1)):
            d[new, :] = cand
            new+=1
        if new % 1000 == 0:
            print("generated {} embeddings".format(new))
    print("generated {} embeddings".format(new))
    assert new == num_emb
    return d
def build_cube_embed(num_emb, embed_dim, random=False, stretch=False, range_dim=None):
    if random:
        d = torch.randint(0, 2 ** (range_dim - 1), (num_emb * 2,)).unique()[:num_emb]
    else:
        step = 1
        if stretch:
            # stretch embeddings in the cube
            step = (2 ** (embed_dim - 1)) // num_emb
        d = torch.arange(0, num_emb) * step
    mask = 2 ** torch.arange(embed_dim - 1, -1, -1).to(d.device, d.dtype)
    d = d.unsqueeze(-1).bitwise_and(mask).ne(0).float()
    d[d == 0] = -1

    return d

def dirichlet_emb(num_emb, embed_dim, freq_vocab):
    x = freq_vocab.max() / (freq_vocab + 1)
    x = ((x - x.min()) / (x.max() - x.min())) * (10 - 0.1) + 0.01
    d = torch.distributions.Dirichlet(torch.tensor(x, dtype=torch.float))
    emb = d.sample(sample_shape=torch.Size([embed_dim])).T
    return emb

def sorted_sim(num_emb, embed_dim):
    emb_ranges = [16,32,48,63]
    probs = [0.2, 0.3, 0.3, 0.2]
    categorical = torch.distributions.Categorical(torch.tensor(probs))
    vals = torch.tensor([], dtype=torch.long)
    while len(vals) < num_emb:
        r = emb_ranges[categorical.sample().item()]
        print(r)
        d = torch.randint((2**(r-1)), (1000,)).unique()
        vals = torch.cat([vals,d], dim=0).unique()

    mask = 2 ** torch.arange(embed_dim - 1, -1, -1)
    d = vals.unsqueeze(-1).bitwise_and(mask).ne(0).float()
    d[d == 0] = -1
    sim_mat = cosine_similarity(d, d)
    mean_sims = np.mean(sim_mat, axis=-1)
    print(np.max(mean_sims), np.mean(mean_sims))
    sorted_idx = np.argsort(mean_sims)
    return d[sorted_idx]

# def gaussian_emb(embed_dim, vocab_file):
#     vocab_orig, idx2word, freq_vocab = read_mt_vocab(vocab_file)
#
#     emb = d.sample()
#     return emb
def random_emb(num_emb, embed_dim):
    return F.normalize(torch.randn(num_emb, embed_dim), dim=-1)

def save_embed(cube, fname, dictionary):
    # cdgm_textgen reuires this format as an input for pre-trained embeddings
    with open(f"{fname}.txt", "w") as out:
        out.write("{} {}".format(cube.shape[0], cube.shape[1]))
        out.write("\n")

        for tok, ind in dictionary.items():
            vals = " ".join([str(x) for x in cube[ind, :].cpu().data.numpy()])
            out.write(f"{tok} {vals}\n")


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("-t", "--type", type=str, required=True, default="cube")
    argp.add_argument("-d", "--dim", type=int, required=True)
    permute = argp.add_argument("-r", "--rand", action="store_true", default=False)
    stretch = argp.add_argument("--stretch", action="store_true", default=False)
    dict = argp.add_argument("-v", "--vocab", required=True)
    save_fn = argp.add_argument("-s", "--save", required=True)

    args = argp.parse_args()

    w2idx, idx2v, freq_vocab = read_mt_vocab(args.vocab)
    num_emb = len(w2idx)

    if args.type=="cube":
        emb_new = build_cube_embed(num_emb, args.dim, random=args.rand, stretch=args.stretch)
    elif args.type=="random":
        emb_new = random_emb(num_emb, args.dim)
    elif args.type=="long_cube":
        emb_new = build_long_cube(num_emb, args.dim)
    elif args.type=="dirichlet":
        emb_new = dirichlet_emb(num_emb, args.dim, freq_vocab)
    elif args.type=="sorted_sim":
        emb_new = sorted_sim(num_emb, args.dim)

    save_embed(emb_new, args.save, w2idx)
