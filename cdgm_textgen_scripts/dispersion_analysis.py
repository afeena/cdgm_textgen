import argparse

import ledoh_torch
import torch
import torch.nn.functional as F
import pandas as pd
import geoopt.manifolds as manifolds

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", help="csv file with the list of checkpoint label, emb paths, grad path", type=str)
    parser.add_argument("-r", help="convert to riemannian grad", dest="riemannian", action="store_true")
    parser.add_argument("-o", help="output to store norms", dest="output", type=str)

    args = parser.parse_args()

    src = pd.read_csv(args.src)
    src = src.reset_index()
    res_grad = {"token_idx":[]}
    res_dist = {"token_idx":[]}
    if args.riemannian:
        manifold = manifolds.SphereExact()

    for index, row in src.iterrows():
        with torch.no_grad():
            embeddings = torch.load(row["emb"])
            if not res_grad["token_idx"]:
                res_grad["token_idx"] = torch.arange(embeddings.size(0)).tolist()
                res_dist["token_idx"] = torch.arange(embeddings.size(0)).tolist()
            else:
                assert len(res_grad["token_idx"]) == embeddings.size(0)
                assert len(res_dist["token_idx"]) == embeddings.size(0)

            if args.riemannian:
                embeddings_gradient_ = torch.load(row["grad"])
                #calculate riemannian gradient given eucl one
                embeddings_gradient = manifold.egrad2rgrad(embeddings, embeddings_gradient_)
            else:
                embeddings_gradient = torch.load(row["grad"])

            l2n = torch.norm(embeddings_gradient, dim=-1)
            mindist = ledoh_torch.utils.minimum_acos_distance_row(embeddings)
            res_grad[row["step"]] = l2n.tolist()
            res_dist[row["step"]] = mindist.tolist()


    res_grad = pd.DataFrame(res_grad)
    res_grad.to_csv(f"{args.output}_norm.csv", index=False)

    res_dist = pd.DataFrame(res_dist)
    res_dist.to_csv(f"{args.output}_dist.csv", index=False)







