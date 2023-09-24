import argparse

import numpy as np
import power_spherical
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

def random_emb(num_emb, embed_dim, pad_idx=None):
    res_emb = torch.randn(num_emb, embed_dim)
    res_emb_normed = F.normalize(res_emb, dim=-1)
    if pad_idx is not None:
        torch.nn.init.constant_(res_emb[pad_idx], 0) #padding
        torch.nn.init.constant_(res_emb_normed[pad_idx],0)
    return res_emb, res_emb_normed

def trafo_init(num_emb, embed_dim, pad_idx=None):
    res_emb = torch.zeros(num_emb, embed_dim)
    torch.nn.init.normal_(res_emb, mean=0, std=embed_dim**-0.5)

    if pad_idx is not None:
        torch.nn.init.constant_(res_emb[pad_idx], 0) #padding
    return res_emb

def ps_init(num_emb, embed_dim, freq_vocab, pad_idx=None):
    freq_vocab = torch.tensor(freq_vocab)+1
    # new_rng = (10000 - 50)
    # old_rng = (freq_vocab.max() - freq_vocab.min())
    # freq_vocab = (((freq_vocab - freq_vocab.min()) * new_rng) / old_rng) + 50
    freq_vocab = freq_vocab.flip(0)

    print(freq_vocab)

    with torch.no_grad():
        ps_dist = power_spherical.PowerSpherical(F.normalize(torch.full((num_emb,embed_dim),embed_dim**-0.5), dim=-1),
                                                 scale=freq_vocab)
        res_emb = ps_dist.sample()

        if pad_idx is not None:
            torch.nn.init.constant_(res_emb[pad_idx], 0) #padding
    return res_emb
def orthogonal(num_emb, embed_dim, pad_idx=None):
    res_emb = torch.zeros(num_emb, embed_dim)
    torch.nn.init.orthogonal_(res_emb)
    if pad_idx is not None:
        torch.nn.init.constant_(res_emb[pad_idx], 0)  # padding
    return res_emb

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
    argp.add_argument("--pad", type=int, default=None)

    args = argp.parse_args()

    w2idx, idx2v, freq_vocab = read_mt_vocab(args.vocab)
    print(freq_vocab.shape)
    num_emb = len(w2idx)

    emb_new = None
    import hashlib
    import time

    hashlib.sha1().update(str(time.time()).encode("utf-8"))
    hash_str = hashlib.sha1().hexdigest()[-1:-7:-1]

    if args.type=="cube":
        emb_new = build_cube_embed(num_emb, args.dim, random=args.rand, step=args.cube_step)
    elif args.type=="random":
        emb_new, emb_new_norm = random_emb(num_emb, args.dim)
        save_embed(emb_new_norm, args.save+f".norm1.{hash_str}", w2idx)
    elif args.type=="lse":
        emb_new = lse_cube(num_emb, args.dim)
    elif args.type=="long_cube":
        emb_new = build_long_cube(num_emb, args.dim)
    elif args.type=="hyperbolic":
        emb_new = create_hyperbolic_emb(num_emb, args.dim)
    elif args.type=="trafo_init":
        emb_new = trafo_init(num_emb, args.dim, pad_idx=args.pad)
    elif args.type=="orthogonal":
        emb_new = orthogonal(num_emb, args.dim, pad_idx=args.pad)
    elif args.type=="ps_init":
        emb_new = ps_init(num_emb, args.dim, freq_vocab, pad_idx=args.pad)

    save_embed(emb_new, f"{args.save}.{hash_str}", w2idx)
