import geoopt
from geoopt.manifolds.stereographic.math import dist2plane
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_manifold_linear(num_embeddings, embed_dim, manifold, padding_idx=None):
    emb = None
    if manifold == "lorentz":
        emb = LorentzLinear(num_embeddings, embed_dim, padding_idx)
    elif manifold == "poincare":
        emb = PoincareLinear(num_embeddings, embed_dim, padding_idx)
    elif manifold == "sphere":
        emb = SphereLinear(num_embeddings, embed_dim, padding_idx)

    return emb


class ManifoldLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, manifold=None):
        bias = False
        super().__init__(in_features, out_features, bias)
        self.manifold = manifold
        self.reset_manifold_parameters()

    @torch.no_grad()
    def reset_manifold_parameters(self):
        self.weight.copy_(self.random_init(self.weight.shape,
                                           dtype=self.weight.dtype, device=self.weight.device))
        self.weight = geoopt.ManifoldParameter(self.weight, manifold=self.manifold)

    def random_init(self, shape, dtype, device):
        raise NotImplementedError

    def to_manifold(self, x):
        raise NotImplementedError

    def to_tangent_space(self, x):
        raise NotImplementedError


class LorentzLinear(ManifoldLinear):
    def __init__(self, in_features, out_features, bias=False, manifold=None):
        bias = False
        manifold = geoopt.Lorentz()
        super().__init__(in_features, out_features, bias, manifold=manifold)

    def random_init(self, shape, dtype, device):
        return self.manifold.random_normal(shape, dtype=dtype, device=device)

    def forward(self, x):
        # https://github.com/chenweize1998/fully-hyperbolic-nn/blob/main/mt/onmt/modules/hyper_nets.py#L41
        # if not self.manifold.check_point_on_manifold(x):
        #     x = self.manifold.expmap0(x)
        # x = x @ self.weight.T
        #
        # x_narrow = x.narrow(-1, 1, x.shape[-1] - 1)
        #
        # time = x.narrow(-1, 0, 1).sigmoid() + 1.1
        # scale = (time * time - self.manifold.k) / \
        #         (x_narrow * x_narrow).sum(dim=-1, keepdim=True)
        # x = torch.cat([time, x_narrow * scale.sqrt()], dim=-1)

        x = x @ self.manifold.logmap0(self.weight).T
        return x

    def to_manifold(self, x):
        return self.manifold.expmap0(x)

    def to_tangent_space(self, x):
        return self.manifold.logmap0(x)


class PoincareLinear(ManifoldLinear):
    def __init__(self, in_features, out_features, bias=False, manifold=None):
        bias = False
        manifold = geoopt.StereographicExact(k=1.0)
        super().__init__(in_features, out_features, bias, manifold=manifold)

    def random_init(self, shape, dtype, device):
        return self.manifold.random_normal(shape, dtype=dtype, device=device)

    def forward(self, x):
        # if not self.manifold.check_point_on_manifold(x):
        #     x = self.manifold.expmap0(x)
        # a_k = self.manifold.logmap0(self.weight),
        # dist2hyp = dist2plane(
        #     x=x,
        #     a=a_k,
        #     p=self.weight
        # )
        # mult_fac = 2*torch.norm(a_k, dim=-1, keepdim=True)/(1-torch.norm(self.weight, dim=-1, keepdim=True)**2)
        x = x @ self.manifold.logmap0(self.weight).T
        return x

    def to_manifold(self, x):
        return self.manifold.expmap0(x)

    def to_tangent_space(self, x):
        return self.manifold.logmap0(x)


class SphereLinear(ManifoldLinear):
    def __init__(self, in_features, out_features, bias=False, manifold=None):
        bias = False
        manifold = geoopt.SphereExact()
        super().__init__(in_features, out_features, bias, manifold=manifold)

    def forward(self, x):
        x = F.normalize(x @ self.weight.T)
        return x

    def random_init(self, shape, dtype, device):
        return self.manifold.random_uniform(shape, dtype=dtype, device=device)

    def to_manifold(self, x):
        x = x / x.norm(dim=-1, keepdim=True)
        return x

    def to_tangent_space(self, x):
        return x
