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
    sims = cosine_similarity(emb, emb)
    average_sims = np.mean(sims, axis=-1)
    for i,avg_sim in enumerate(average_sims):
        print(i, avg_sim)
        if i>15000 and avg_sim > 0.5:
            new_emb[i,:] = torch.rand(emb.shape[-1])
    return new_emb

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
    save_embed(emb, sys.argv[1]+".disent", vocab)

