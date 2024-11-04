import sys
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np

from utils import txt_to_pt
def save_embed(emb, fname, dictionary):
    print(fname, emb.shape)
    # cdgm_textgen reuires this format as an input for pre-trained embeddings
    with open(f"{fname}.txt", "w") as out:
        out.write("{} {}".format(emb.shape[0], emb.shape[1]))
        out.write("\n")

        for tok, ind in dictionary.items():
            vals = " ".join([str(x) for x in emb[ind, :].cpu().data.numpy()])
            out.write(f"{tok} {vals}\n")


def disentangle_rare_emb(emb):
    new_emb = emb.clone()
    thr = 20000
    shape_rare = new_emb[thr:, :].shape
    new_emb[thr:, :] = torch.nn.functional.normalize(torch.randn(*shape_rare), dim=-1)
    return new_emb

def sum_random_and_pretrained(emb, lambda_rand=0.5):
    re = torch.nn.functional.normalize(torch.randn(emb.shape), dim=-1)
    resulting_emb = (1-lambda_rand)*emb+lambda_rand*re
    return torch.nn.functional.normalize(resulting_emb, dim =-1)

vocab, emb = txt_to_pt(sys.argv[1])


if sys.argv[2] == "validate":
    print(emb.norm(dim=-1))
    print(emb.shape)
if sys.argv[2] == "pad":
    torch.nn.init.constant_(emb[1], 0)  # padding
    save_embed(emb, sys.argv[1]+".pad0", vocab)
if sys.argv[2] == "norm":
    emb = emb / emb.norm(dim=-1, keepdim=True)
    save_embed(emb, sys.argv[1]+".norm1", vocab)
if sys.argv[2]=="disent":
    emb = disentangle_rare_emb(emb)
    save_embed(emb, sys.argv[1]+".disent20k", vocab)
if sys.argv[2]=="sum_emb":
    lambda_rand = float(sys.argv[3])
    emb = sum_random_and_pretrained(emb)
    save_embed(emb, sys.argv[1]+f".{lambda_rand}_rand", vocab)

