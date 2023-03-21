import math
from typing import Tuple

import torch
from torch import Tensor
import torch.nn.functional as F
import numpy as np

@torch.jit.script
def output_with_temperature_scriptable(out: Tensor, temperature: float) -> Tensor:
    return out / temperature

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
