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
    token_idxs = [5, 6, 7, 8, 9, 26060, 25730, 27319, 27413, 27328, 34, 38, 47, 59, 183, 19503, 21703, 24858, 26597, 23579]
    tokens = ["▁the",",",".","s","▁of","▁Kilimanjaro","▁Celebrity","▁sneakers","▁vintage","▁spaghetti","▁have","▁not","▁at","▁but","▁energy","leo","675","misli","▁Volume","▁quae"]
    token_freq = [1053762,768101,584740,532100,487430,1,1,1,1,1,84735,61222,51467,37053,11838,4,2,1,1,2]

    with torch.no_grad():
        normalized_emb = F.normalize(embeddings, p=2, dim=-1)
        for tid in token_idxs:
            direction = embeddings[tid] @ embeddings_gradient[tid].t()
            l2n = torch.norm(embeddings_gradient[tid])
            min_dist = torch.acos((normalized_emb@normalized_emb[tid]).clamp(-1 + 1e-9, 1 - 1e-9)).topk(2, largest=False).values[1]

            print(f"{direction.item()}||{l2n.item()}||{min_dist.item()}")









