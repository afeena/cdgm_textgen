import argparse
import torch
from ledoh_torch import avg_acos_distance_batch, circular_variance, \
    minimum_acos_distance_batch, median_acos_distance_batch, \
    minimum_acos_distance_row

import ot
from utils import txt_to_pt
import numpy as np
import plotext as plt

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, type=str, help="path to model checkpoint")
    parser.add_argument("--txt", default=None, type=str, help="path to txt file with sentences")
    parser.add_argument("--npy", default=None, type=str, help="path to npy file with vectors")
    parser.add_argument("--bs", default=1024, type=int, help="block size to calc min dist")
    parser.add_argument("--hist", action="store_true", help="show histogram")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model is not None:
        model = torch.load(args.model)
        print(model.keys())
        embed = model["model"]["decoder.embed_tokens.weight"]
    elif args.txt is not None:
        vocab, embed = txt_to_pt(args.txt)
    elif args.npy is not None:
        embed = torch.tensor(np.load(args.npy))
    else:
        print("Provide either model or txt")
        exit(-1)
    embed = embed.to(device)
    norms = torch.norm(embed, dim=-1)
    if not torch.allclose(norms, torch.tensor([1.0], device=device)):
        embed = embed / norms.unsqueeze(-1)
    avg = avg_acos_distance_batch(embed, args.bs)
    md = minimum_acos_distance_batch(embed, args.bs)
    median = median_acos_distance_batch(embed, args.bs)
    cv = circular_variance(embed)
    ssw = ot.sliced_wasserstein_sphere_unif(embed, n_projections=100)
    print(args.model, args.txt)

    print(f"min dist: {np.round(md, decimals=2)}")
    print(f"circular variance: {np.round(cv, decimals=2)}")
    print(f"sliced wasserstein: {np.round(ssw, decimals=4)}")

    if args.hist:
        row_md = minimum_acos_distance_row(embed, embed)

        bins = np.arange(1.3,1.6,0.01)

        frq, edges = np.histogram(row_md.cpu(), bins)

        plt.bar(edges[:-1], frq, width=0.01)
        plt.show()
