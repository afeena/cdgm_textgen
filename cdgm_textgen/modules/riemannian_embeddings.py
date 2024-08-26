from typing import Optional

import geoopt
import torch
import torch.nn as nn
from fairseq import utils
from torch import Tensor


def build_manifold_embeddings(num_embeddings, embed_dim, manifold, padding_idx=None):
    emb = None
    if manifold == "lorentz":
        emb = LorentzEmbeddings(num_embeddings, embed_dim, padding_idx)
    elif manifold == "poincare":
        emb = PoincareEmbeddings(num_embeddings, embed_dim, padding_idx)
    elif manifold == "sphere":
        emb = SphereEmbeddings(num_embeddings, embed_dim, padding_idx)
    return emb


class ManifoldEmbeddings(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None,
                 max_norm: Optional[float] = None, norm_type: float = 2., scale_grad_by_freq: bool = False,
                 sparse: bool = False, _weight: Optional[Tensor] = None, _freeze: bool = False,
                 device=None, dtype=None, manifold=None) -> None:
        super().__init__(num_embeddings, embedding_dim,
                         padding_idx, max_norm, norm_type,
                         scale_grad_by_freq, sparse,
                         _weight, _freeze, device, dtype)
        self.manifold = manifold
        self.reset_manifold_parameters()

    @torch.no_grad()
    def random_init(self, shape, dtype, device):
        raise NotImplementedError
    @torch.no_grad()
    def reset_manifold_parameters(self):
        self.weight.copy_(self.random_init((self.num_embeddings, self.embedding_dim),
                                               dtype=self.weight.dtype, device=self.weight.device))
        self.weight = geoopt.ManifoldParameter(self.weight, manifold=self.manifold)


    @torch.no_grad()
    def load_embeddings(self, path, dictionary):
        embed_dict = utils.parse_embedding(path)
        utils.load_embedding(embed_dict, dictionary, self.weight)
        self.weight.copy_(self.to_manifold(self.weight.data))

    def to_manifold(self, x):
        raise NotImplementedError

    def to_tangent_space(self, x):
        raise NotImplementedError


class LorentzEmbeddings(ManifoldEmbeddings):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None,
                 max_norm: Optional[float] = None, norm_type: float = 2., scale_grad_by_freq: bool = False,
                 sparse: bool = False, _weight: Optional[Tensor] = None, _freeze: bool = False,
                 device=None, dtype=None, manifold=None):
        manifold = geoopt.Lorentz()
        super().__init__(num_embeddings, embedding_dim, padding_idx, max_norm, norm_type,
                         scale_grad_by_freq, sparse, _weight, _freeze, device, dtype, manifold)

    @torch.no_grad()
    def random_init(self, shape, dtype, device):
        return self.manifold.random_normal(shape, dtype=dtype, device=device)
    def to_manifold(self, x):
        return self.manifold.expmap0(x)

    def to_tangent_space(self, x):
        return self.manifold.logmap0(x)

class PoincareEmbeddings(ManifoldEmbeddings):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None,
                 max_norm: Optional[float] = None, norm_type: float = 2., scale_grad_by_freq: bool = False,
                 sparse: bool = False, _weight: Optional[Tensor] = None, _freeze: bool = False,
                 device=None, dtype=None, manifold=None):
        manifold = geoopt.StereographicExact(k=-1.0)
        super().__init__(num_embeddings, embedding_dim, padding_idx, max_norm, norm_type,
                         scale_grad_by_freq, sparse, _weight, _freeze, device, dtype, manifold)

    @torch.no_grad()
    def random_init(self, shape, dtype, device):
        return self.manifold.random_normal(shape, dtype=dtype, device=device)
    def to_manifold(self, x):
        return self.manifold.expmap0(x)

    def to_tangent_space(self, x):
        return self.manifold.logmap0(x)


class SphereEmbeddings(ManifoldEmbeddings):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None,
                 max_norm: Optional[float] = None, norm_type: float = 2., scale_grad_by_freq: bool = False,
                 sparse: bool = False, _weight: Optional[Tensor] = None, _freeze: bool = False,
                 device=None, dtype=None, manifold=None):
        self.manifold = geoopt.SphereExact()
        super().__init__(num_embeddings, embedding_dim, padding_idx, max_norm, norm_type,
                         scale_grad_by_freq, sparse, _weight, _freeze, device, dtype, manifold)

    @torch.no_grad()
    def random_init(self, shape, dtype, device):
        return self.manifold.random_uniform(shape, dtype=dtype, device=device)

    def to_manifold(self, x):
        x = x / x.norm(dim=-1, keepdim=True)
        return x

    def to_tangent_space(self, x):
        return x
