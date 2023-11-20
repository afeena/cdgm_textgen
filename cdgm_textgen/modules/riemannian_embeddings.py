from typing import Optional

import geoopt
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from torch import Tensor


def build_manifold_embeddings(num_embeddings, embed_dim, manifold, padding_idx=None):
    if isinstance(manifold, geoopt.Lorentz):
        emb = LorentzEmbeddings(num_embeddings, embed_dim, padding_idx, manifold=manifold)
    elif (isinstance(manifold, geoopt.StereographicExact) and manifold.k == -1):
        emb = PoincareEmbeddings(num_embeddings, embed_dim, padding_idx, manifold=manifold)
    elif isinstance(manifold, geoopt.SphereExact):
        emb = SphereEmbeddings(num_embeddings, embed_dim, padding_idx, manifold=manifold)
    else:
        emb = EuclideanEmbeddings(num_embeddings, embed_dim, padding_idx, manifold=geoopt.Euclidean())

    return emb


class ManifoldEmbeddings(nn.Module):
    def __init__(self, num_embeddings,
                 embedding_dim,
                 padding_idx,
                 manifold,
                 _weight: Optional[Tensor] = None,
                 _freeze: bool = False,
                 device=None,
                 dtype=None
                 ):

        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.manifold = manifold
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < self.num_embeddings, 'Padding_idx must be within num_embeddings'
            elif padding_idx < 0:
                assert padding_idx >= -self.num_embeddings, 'Padding_idx must be within num_embeddings'
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx

        if _weight is None:
            self.weight = geoopt.ManifoldParameter(
                torch.empty((num_embeddings, embedding_dim), **factory_kwargs),
                manifold=self.manifold, requires_grad=not _freeze
            )
        else:
            assert list(_weight.shape) == [num_embeddings, embedding_dim], \
                'Shape of weight does not match num_embeddings and embedding_dim'
            self.weight = geoopt.ManifoldParameter(self.to_manifold(_weight), requires_grad=not _freeze)



    def forward(self, x):
        raise NotImplementedError

    @torch.no_grad()
    def reset_manifold_parameters(self):
        self.weight.copy_(self.manifold.random((self.num_embeddings, self.embedding_dim),
                                               dtype=self.weight.dtype, device=self.weight.device))

    def to_manifold(self, x):
        raise NotImplementedError

    def to_tangent_space(self, x):
        raise NotImplementedError


class LorentzEmbeddings(ManifoldEmbeddings):
    def __init__(self, n_in,
                 n_out,
                 padding_idx,
                 manifold
                 ):
        super().__init__(n_in, n_out, padding_idx, manifold)
        assert isinstance(self.manifold, geoopt.Lorentz), "Manifold must be Lorentz"

    def forward(self, x):
        # https://github.com/chenweize1998/fully-hyperbolic-nn/blob/main/mt/onmt/modules/hyper_nets.py#L41
        x = x.double()
        if not self.manifold.check_point_on_manifold(x):
            x = self.manifold.expmap0(x)
        x = x @ self.weight.T.double()

        x_narrow = x.narrow(-1, 1, x.shape[-1] - 1)

        time = x.narrow(-1, 0, 1).sigmoid() + 1.1
        scale = (time * time - self.manifold.k) / \
                (x_narrow * x_narrow).sum(dim=-1, keepdim=True)
        x = torch.cat([time, x_narrow * scale.sqrt()], dim=-1)

        return x.float()

    def load_embeddings(self, path, dictionary):
        embed_dict = utils.parse_embedding(path)
        utils.load_embedding(embed_dict, dictionary, self.weight)

    def to_manifold(self, x):
        return self.manifold.expmap0(x)

    def to_tangent_space(self,x):
        return self.manifold.logmap0(x)


class PoincareEmbeddings(ManifoldEmbeddings):
    def __init__(self, n_in,
                 n_out,
                 padding_idx,
                 manifold
                 ):
        super().__init__(n_in, n_out, padding_idx, manifold)
        assert (isinstance(self.manifold,
                           geoopt.StereographicExact) and self.manifold.k == -1), "Manifold must be StereographicExact with k=-1"

    def forward(self, x):
        # https://github.com/chenweize1998/fully-hyperbolic-nn/blob/main/mt/onmt/modules/hyper_nets.py#L41
        if not self.manifold.check_point_on_manifold(x):
            x = self.manifold.expmap0(x)
        x = x.double()
        x = self.manifold.mobius_matvec(self.weight.double(), x)

        return x.float()

    def to_manifold(self, x):
        return self.manifold.expmap0(x)

    def to_tangent_space(self, x):
        return self.manifold.logmap0(x)

    @torch.no_grad()
    def load_embeddings(self, path, dictionary):
        embed_dict = utils.parse_embedding(path)
        utils.load_embedding(embed_dict, dictionary, self.weight)
        self.weight.copy_(self.manifold.expmap0(self.weight))


class SphereEmbeddings(ManifoldEmbeddings):
    def __init__(self, n_in,
                 n_out,
                 padding_idx,
                 manifold
                 ):
        super().__init__(n_in, n_out, padding_idx, manifold)
        assert isinstance(self.manifold, geoopt.SphereExact), "Manifold must be SphereExact"


    def forward(self, x):
        # https://github.com/chenweize1998/fully-hyperbolic-nn/blob/main/mt/onmt/modules/hyper_nets.py#L41
        x = F.normalize(x @ self.weight.T)
        return x

    def to_manifold(self, x):
        x = x/x.norm(dim=-1, keepdim=True)
        return x

    def to_tangent_space(self, x):
        return x

    @torch.no_grad()
    def load_embeddings(self, path, dictionary):
        embed_dict = utils.parse_embedding(path)
        utils.load_embedding(embed_dict, dictionary, self.weight)
        self.weight.copy_(self.weight.data / self.weight.data.norm(dim=-1, keepdim=True))


class EuclideanEmbeddings(ManifoldEmbeddings):
    @torch.no_grad()
    def load_embeddings(self, path, dictionary):
        embed_dict = utils.parse_embedding(path)
        utils.load_embedding(embed_dict, dictionary, self.weight)

    def to_manifold(self, x):
        return x

    def to_tangent_space(self, x):
        return x
