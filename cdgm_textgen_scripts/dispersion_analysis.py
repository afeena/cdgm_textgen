import argparse
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
    res = {"token_idx":[]}
    if args.riemannian:
        manifold = manifolds.SphereExact()

    for index, row in src.iterrows():
        with torch.no_grad():
            embeddings = torch.load(row["emb"])
            if not res["token_idx"]:
                res["token_idx"] = torch.arange(embeddings.size(0)).tolist()
            else:
                assert len(res["token_idx"]) == embeddings.size(0)

            if args.riemannian:
                embeddings_gradient_ = torch.load(row["grad"])
                #calculate riemannian gradient given eucl one
                embeddings_gradient = manifold.egrad2rgrad(embeddings, embeddings_gradient_)
            else:
                embeddings_gradient = torch.load(row["grad"])

            l2n = torch.norm(embeddings_gradient, dim=-1)
            res[row["step"]] = l2n.tolist()


    res = pd.DataFrame(res)
    res.to_csv(args.output, index=False)







