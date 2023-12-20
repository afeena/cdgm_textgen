import torch
import geoopt
from geoopt.optim import RiemannianAdam
from geoopt.manifolds import SphereExact

import sys
sys.path.append("..")
from cdgm_textgen.modules.sliced_sphere import compute_sliced_sphere_dispersion

def get_great_circle(d):
    PQ = torch.randn(d, 2)

    PQ, R = torch.linalg.qr(PQ)
    PQ *= R.diagonal().sign()
    p, q = PQ[:, 0], PQ[:, 1]

    return p, q

def main():
    embed_dim = 128

    X = (torch.ones(1000, embed_dim)+1e-3*torch.randn(1000, embed_dim)).requires_grad_(True)
    X = X / torch.norm(X, dim=-1, keepdim=True)


    dist = torch.acos(X.data @ X.data.T)
    dist.fill_diagonal_(float('inf'))
    min_dist = dist.view(-1).min()
    mean_v = 1 - torch.norm(torch.mean(X.data, dim=0))
    print(min_dist.detach(), mean_v.detach())


    manifold = SphereExact()
    X = geoopt.ManifoldParameter(X, manifold=manifold)
    optimizer = RiemannianAdam([X], lr=1e-2)

    for i in range(1000):
        optimizer.zero_grad()
        p,q = get_great_circle(embed_dim)
        loss = compute_sliced_sphere_dispersion(X, p, q)
        loss.backward()
        optimizer.step()

        dist = torch.acos(X.data @ X.data.T)
        dist.fill_diagonal_(float('inf'))
        min_dist = dist.view(-1).min()
        mean_v = 1 - torch.norm(torch.mean(X.data, dim=0))
        print(min_dist.detach(), mean_v.detach())


if __name__ == '__main__':
    main()