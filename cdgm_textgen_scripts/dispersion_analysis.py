import argparse
import torch
import torch.nn.functional as F
import pandas as pd
import geoopt.manifolds as manifolds

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", help="checkpoint emb matrix", dest="emb", type=str)
    parser.add_argument("-g", help="checkpoint emb gradient", dest="emb_grad",type=str)
    parser.add_argument("-r", help="convert to riemannian grad", dest="riemannian", action="store_true")
    parser.add_argument("-o", help="output to store norms", dest="output", type=str)


    args = parser.parse_args()

    with torch.no_grad():
        embeddings = torch.load(args.emb)

        if args.riemannian:
            embeddings_gradient_ = torch.load(args.emb_grad)
            #calculate riemannian gradient given eucl one
            embeddings_gradient = manifolds.SphereExact().erd2rgrad(embeddings, embeddings_gradient_)
        else:
            embeddings_gradient = torch.load(args.emb_grad)

        #token_idxs = [5, 6, 59, 183, 19503, 26597, 27413]

        #normalized_emb = F.normalize(embeddings, p=2, dim=-1)
            #direction = embeddings[tid] @ embeddings_gradient[tid].t()
        l2n = torch.norm(embeddings_gradient, dim=-1)
        tok_idx = torch.arange(embeddings.size(0))
        pd.DataFrame({"token_id": tok_idx, "grad_norm": l2n.numpy()}).to_csv(args.output, index=False)
            #min_dist = torch.acos((normalized_emb@normalized_emb[tid]).clamp(-1 + 1e-9, 1 - 1e-9)).topk(2, largest=False).values[1]

            #print(f"{direction.item()}\t{l2n.item()}\t{min_dist.item()}")









