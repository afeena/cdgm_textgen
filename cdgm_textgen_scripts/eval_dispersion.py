import argparse
import torch
from ledoh_torch import avg_acos_distance_batch, circular_variance
from utils import txt_to_pt
import numpy as np

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, type=str, help="path to model checkpoint")
    parser.add_argument("--txt", default=None, type=str, help="path to txt file with sentences")
    parser.add_argument("--bs", default=1024, type=int, help="block size to calc min dist")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model is not None:
        model = torch.load(args.model)
        print(model.keys())
        embed = model["model"]["decoder.embed_tokens.weight"]
    elif args.txt is not None:
        vocab, embed = txt_to_pt(args.txt)
    else:
        print("Provide either model or txt")
        exit(-1)
    embed = embed.to(device)
    norms = torch.norm(embed, dim=-1)
    if not torch.allclose(norms, torch.tensor([1.0], device=device)):
        embed = embed / norms.unsqueeze(-1)
    md = avg_acos_distance_batch(embed, args.bs)
    # bins = np.arange(0,1.5,0.1)
    #
    # frq, edges = np.histogram(md, bins)
    #
    # plt.bar(edges[:-1], frq, width=0.1)
    # plt.show()
    cv = circular_variance(embed)
    print(args.model, args.txt)
    print(f"avg dist: {md}, circular variance: {cv}")
