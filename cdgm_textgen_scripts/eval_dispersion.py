import argparse
import torch
from ledoh_torch import minimum_acos_distance_block, circular_variance
import plotext as plt
from utils import txt_to_pt
import numpy as np

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, type=str, help="path to model checkpoint")
    parser.add_argument("--txt", default=None, type=str, help="path to txt file with sentences")

    args = parser.parse_args()

    if args.model is not None:
        model = torch.load(args.model)
        print(model.keys())
        embed = model["model"]["decoder.embed_tokens.weight"]
    elif args.txt is not None:
        vocab, embed = txt_to_pt(args.txt)
    else:
        print("Provide either model or txt")
        exit(-1)
    norms = torch.norm(embed, dim=-1)
    print(torch.allclose(norms, torch.tensor([1.0])))
    if not torch.allclose(norms, torch.tensor([1.0])):
        embed = embed / norms.unsqueeze(-1)
    md = minimum_acos_distance_block(embed, embed)
    # bins = np.arange(0,1.5,0.1)
    #
    # frq, edges = np.histogram(md, bins)
    #
    # plt.bar(edges[:-1], frq, width=0.1)
    # plt.show()
    cv = circular_variance(embed)
    print(f"min dist: {md.min()}, circular variance: {cv}")
