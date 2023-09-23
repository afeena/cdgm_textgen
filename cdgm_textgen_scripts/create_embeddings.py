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

def lse_cube(num_emb, embed_dim):
    d = torch.rand(num_emb, 1024)
    src_dim = d.shape[1]
    A = torch.randn(embed_dim, src_dim)
    _, R = torch.linalg.qr(A, mode='r')
    tgt = d @ R.T
    tgt = torch.where(tgt > 0, 1.0, -1.0)
    print(tgt.shape[0]==torch.unique(tgt, dim=0).shape[0], torch.unique(tgt, dim=0).shape[0], tgt.shape[0])
    return tgt

def build_cube_128(num_emb, embed_dim, step=None):
    v1 = build_cube_embed(num_emb, 64, step=step)
    v2 = build_cube_embed(num_emb, 64, step=step)
    return torch.cat((v1,v2), dim=-1)

def build_cube_embed(num_emb, embed_dim, random=False, range_dim=None, step=None):
    if embed_dim==128:
        build_cube_128(num_emb, 128, step=step)
    if random:
        if range_dim is None:
            range_dim = num_emb
        d = torch.randint(0, 2 ** (range_dim - 1), (num_emb * 2,)).unique()[:num_emb]
    else:
        if step=="max":
            step = (2 ** (embed_dim - 1)) // num_emb
        else:
            step = 1 if step is None else int(step)

        d = torch.arange(0, num_emb) * step
    mask = 2 ** torch.arange(embed_dim - 1, -1, -1).to(d.device, d.dtype)
    d = d.unsqueeze(-1).bitwise_and(mask).ne(0).float()
    d[d == 0] = -1

    return d

def random_emb(num_emb, embed_dim):
    return torch.randn(num_emb, embed_dim)

def save_embed(cube, fname, dictionary):
    # cdgm_textgen reuires this format as an input for pre-trained embeddings
    with open(f"{fname}.txt", "w") as out:
        out.write("{} {}".format(cube.shape[0], cube.shape[1]))
        out.write("\n")

        for tok, ind in dictionary.items():
            vals = " ".join([str(x) for x in cube[ind, :].cpu().data.numpy()])
            out.write(f"{tok} {vals}\n")

def create_hyperbolic_emb(num_emb, embed_dim):
    import geoopt

    manifold = geoopt.manifolds.Stereographic(k=-1.0)
    return manifold.random((num_emb, embed_dim)).detach()


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("-t", "--type", type=str, required=True, default="cube")
    argp.add_argument("-d", "--dim", type=int, required=True)
    argp.add_argument("-r", "--rand", action="store_true", default=False)
    argp.add_argument("--stretch", action="store_true", default=False)
    argp.add_argument("--cube-step", type=str)
    argp.add_argument("-v", "--vocab", required=True)
    argp.add_argument("-s", "--save", required=True)

    args = argp.parse_args()

    w2idx, idx2v, freq_vocab = read_mt_vocab(args.vocab)
    num_emb = len(w2idx)

    emb_new = None
    if args.type=="cube":
        emb_new = build_cube_embed(num_emb, args.dim, random=args.rand, step=args.cube_step)
    elif args.type=="random":
        emb_new = random_emb(num_emb, args.dim)
        emb_new_norm = F.normalize(emb_new, p=2, dim=-1)
        save_embed(emb_new, args.save+"norm1", w2idx)
    elif args.type=="lse":
        emb_new = lse_cube(num_emb, args.dim)
    elif args.type=="long_cube":
        emb_new = build_long_cube(num_emb, args.dim)
    elif args.type=="hyperbolic":
        emb_new = create_hyperbolic_emb(num_emb, args.dim)

    save_embed(emb_new, args.save, w2idx)
