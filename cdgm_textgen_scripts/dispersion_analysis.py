import argparse
import torch
import torch.nn.functional as F
import pandas as pd

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", help="checkpoint emb matrix", dest="emb", type=str)
    parser.add_argument("-g", help="checkpoint emb gradient", dest="emb_grad",type=str)
    parser.add_argument("-t", help="token analysis csv", dest="tok_table", type=str)


    args = parser.parse_args()


    embeddings = torch.load(args.emb)
    embeddings_gradient = torch.load(args.emb_grad)
    token_idxs = [5, 6, 59, 183, 19503, 26597, 27413]

    with torch.no_grad():
        normalized_emb = F.normalize(embeddings, p=2, dim=-1)
        for tid in token_idxs:
            direction = embeddings[tid] @ embeddings_gradient[tid].t()
            l2n = torch.norm(embeddings_gradient[tid])
            min_dist = torch.acos((normalized_emb@normalized_emb[tid]).clamp(-1 + 1e-9, 1 - 1e-9)).topk(2, largest=False).values[1]

            print(f"{direction.item()}\t{l2n.item()}\t{min_dist.item()}")









