import math
from typing import Tuple

import geoopt.manifolds.stereographic.math as pmath
import torch
import torch.nn.functional as F
from geoopt.utils import clamp_abs
from torch import Tensor


@torch.jit.script
def output_with_temperature_scriptable(out: Tensor, temperature: float) -> Tensor:
    return out / temperature


@torch.jit.script
def triple_sum_scriptable(x: Tensor, y: Tensor, z: Tensor) -> Tuple[Tensor, Tensor]:
    x = x + y
    return x, x + z


@torch.jit.script
def sum_scriptable(x: Tensor, y: Tensor) -> Tensor:
    return x + y


@torch.jit.script
def softmax_with_temperature_scriptable(out: Tensor, temperature: float) -> Tensor:
    return F.softmax(out / temperature, dim=-1)


@torch.jit.script
def log_softmax_with_temperature_scriptable(out: Tensor, temperature: float) -> Tensor:
    return F.log_softmax(out / temperature, dim=-1)


@torch.jit.script
def log_prob_scriptable(loc: Tensor, scale: Tensor, value: Tensor, m_string: str) -> Tensor:
    beta = torch.tensor([(loc.shape[-1] - 1) / 2], device=loc.device)
    alpha = beta + scale
    log_prob = -(
            (alpha + beta) * math.log(2)
            + torch.lgamma(alpha)
            - torch.lgamma(alpha + beta)
            + beta * math.log(math.pi)
    ) + scale * torch.log1p(
        torch.einsum(m_string, loc, value)
    )
    return log_prob


def parse_cluster_means(cluster_path, cluster_means_size):
    """
    parse cluster means from file
    file should contain one cluster mean per line
    header contains num_cluster, num_features
    """
    cluster_means = torch.empty(cluster_means_size)

    with open(cluster_path) as f_clusters:
        for line_nr, line in enumerate(f_clusters):
            pieces = line.rstrip().split(" ")
            cluster_means[line_nr] = torch.Tensor(
                [float(weight) for weight in pieces]
            )
    return cluster_means


@torch.jit.script
def dist2plane(x: Tensor,
               a: Tensor,
               p: Tensor,
               k: Tensor,
               signed: bool = False,
               scaled: bool = False,
               ) -> Tensor:
    """
    Function taken from geoopt.manifolds.stereographic.math.dist2plane
    https://github.com/geoopt/geoopt/blob/master/geoopt/manifolds/stereographic/math.py#L1424
    However, use einsum instead
    x: is dimension batch x seq_len x 1 x dim
    a: is dimension vocab_size x dim
    """
    diff = pmath._mobius_add(-p, x, k, dim=-1)
    diff_norm2 = diff.pow(2).sum(dim=-1).clamp_min(1e-15)
    sc_diff_a = torch.einsum('bsid,vd->bsv', diff, a)

    if not signed:
        sc_diff_a = sc_diff_a.abs()
    a_norm = a.norm(dim=-1, p=2).clamp_min(1e-15)
    num = 2.0 * sc_diff_a
    denom = clamp_abs((1 + k * diff_norm2) * a_norm)
    distance = pmath.arsin_k(num / denom, k)
    if scaled:
        distance = distance * a_norm
    return distance


def mobius_add(x: Tensor, y: Tensor, k: Tensor, dim: int = -1) -> Tensor:
    """
    mobius add but with einsum
    x: is dimension batch x seq_len x dim
    y: is dimension vocab_size x dim
    """
    x2 = x.pow(2).sum(dim=dim, keepdim=True)  # b x s x 1 x 1
    y2 = y.pow(2).sum(dim=dim, keepdim=True)  # v x 1
    xy = 2 * k * torch.einsum('bsid,vd->bsvi', x, y)  # b x s x v x 1

    return ((1 - xy - k * y2) * x + (1 + k * x2) * y) / (1 - xy + k ** 2 * x2 * y2).clamp_min(1e-15)


def dist(x: torch.Tensor,
         y: torch.Tensor,
         k: torch.Tensor,
         keepdim: bool = False,
         dim: int = -1) -> Tensor:
    return 2.0 * pmath.artan_k(
        mobius_add(-x, y, k, dim=dim).norm(dim=dim, p=2, keepdim=keepdim), k
    )


def chunked_y_dist(x: Tensor,
                   y: Tensor,
                   k: Tensor,
                   chunk_size: int = 10) -> Tensor:
    splitted_target = y.split(chunk_size, dim=0)
    res = []
    for m in splitted_target:
        res.append(dist(x, m, k))
    return torch.cat(res, dim=-1)

@torch.jit.script
def artan_k(x: torch.Tensor, k: torch.Tensor):
    k_sqrt = pmath.sabs(k).sqrt()
    scaled_x = x * k_sqrt

    return k_sqrt.reciprocal() * pmath.artanh(scaled_x)

@torch.jit.script
def stereographic_poincare_dist(x: torch.Tensor,
                                y: torch.Tensor,
                                k: torch.Tensor,
                                keepdim: bool = False,
                                dim: int = -1):
    # that is only for stereographic model with k=-1.0 !!!
    cpu_device = torch.device('cpu')
    orig_device = x.device

    x = x * -1.0
    x2 = x.pow(2).sum(dim=dim, keepdim=True)  # b x s x 1 x 1
    y2 = y.pow(2).sum(dim=dim, keepdim=True)  # v x 1
    xy = 2 * torch.einsum('bsid,vd->bsvi', x,y)  # b x s x v x 1
    denom = torch.addcmul(1 + xy, x2, y2)

    mobius_add = torch.addcmul(y.to(cpu_device)+x.to(cpu_device), xy.to(cpu_device), x.to(cpu_device))  # bsv1 x bs1d -> bsvd  very heavy!
    mobius_add = torch.addcmul(mobius_add, y2.to(cpu_device), x.to(cpu_device))  # bsv1 x bs1d -> bsvd
    mobius_add = torch.addcmul(mobius_add, x2.to(cpu_device), y.to(cpu_device), value=-1.0)  # bs11 x vd -> bsvd

    mobius_add = mobius_add / denom.to(cpu_device).clamp_min(1e-15)

    res_dist = 2.0 * artan_k(mobius_add.norm(dim=dim, p=2, keepdim=keepdim), k.to(cpu_device)).to(orig_device)

    return res_dist

@torch.jit.script
def stereographic_poincare_dist_chunked(x: torch.Tensor,
                                        y: torch.Tensor,
                                        k: torch.Tensor,
                                        keepdim: bool = False,
                                        dim: int = -1,
                                        chunk_size: int = 1000):
    # that is only for stereographic model with k=-1.0 !!!

    splitted_target = y.split(chunk_size, dim=0)
    res = []
    for m in splitted_target:
        res.append(stereographic_poincare_dist(x, m, k))
    return torch.cat(res, dim=-1)
