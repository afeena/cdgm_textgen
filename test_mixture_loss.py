import math
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


@torch.jit.script
def get_loss_scriptable(q_loc: Tensor, q_scale: Tensor, p_loc: Tensor, p_scale: Tensor,
                        z_samples: Tensor,
                        mixture_out: Tensor,
                        mixture_temperature: float,
                        regularizer: str,
                        mixture_regularizer_weight: float,
                        log_pi: float,
                        log2: float) -> Tuple[Tensor, Tensor]:
    mixture_out_probs = F.softmax(mixture_out / mixture_temperature, dim=-1)  # prior
    log_prior_probs = torch.log(mixture_out_probs)

    # here we assume beta q = beta p since both has the same dimension
    beta = (q_loc.shape[-1] - 1) / 2
    q_alpha = (q_loc.shape[-1] - 1) / 2 + q_scale
    log_q_prob = -(
            (q_alpha + beta) * log2
            + torch.lgamma(q_alpha)
            - torch.lgamma(q_alpha + beta)
            + beta * log_pi
    ) + q_scale * torch.log1p(
        torch.einsum("ijk, ljk -> lij", q_loc, z_samples)
    )

    p_alpha = (p_loc.shape[-1] - 1) / 2 + p_scale
    log_p_prob = -(
            (p_alpha + beta) * log2
            + torch.lgamma(p_alpha)
            - torch.lgamma(p_alpha + beta)
            + beta * log_pi
    ) + p_scale * torch.log1p(
        torch.einsum("jk, ljk -> lj", p_loc, z_samples)
    )

    prob_sum = log_q_prob + log_prior_probs.transpose(0, 1).unsqueeze(0)
    log_q_out = torch.logsumexp(prob_sum, dim=1)
    loss = torch.sum(mixture_out_probs * (log_q_out.t() - log_p_prob.t()), dim=-1)
    prior_entropy = -(mixture_out_probs * log_prior_probs).mean(dim=-1)

    reg_term = torch.tensor([0.0], device=mixture_out.device)
    if regularizer == "posterior":
        lmodes_probs = log_prior_probs + log_q_out.t()
        lmodes_probs = lmodes_probs.t() - torch.logsumexp(lmodes_probs, dim=-1)
        entropy = -(mixture_out_probs * lmodes_probs.t()).mean(dim=-1)
        reg_term = (mixture_regularizer_weight * entropy).mean(dim=-1)
    if regularizer == "prior":
        entropy = -(mixture_out_probs * log_prior_probs).mean(dim=-1)
        reg_term = (mixture_regularizer_weight * entropy).mean(dim=-1)

    loss = loss.mean(dim=-1) + reg_term

    return loss, prior_entropy


if __name__ == "__main__":
    q_loc = torch.rand(128, 64, 128).cuda()
    q_scale = torch.rand(128, 64).cuda()
    p_loc = torch.rand(64, 128).cuda()
    p_scale = torch.rand(64, ).cuda()
    z_samples = torch.rand(128, 64, 128).cuda()
    mixture_out = torch.rand(64, 128).cuda()
    mixture_temperature = 1.0
    regularizer = 'none'
    mixture_regularizer_weight = 0.0
    log_pi = math.log(math.pi)
    log2 = math.log(2.0)
    with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
            ],
            profile_memory=True,
    ) as p:
        for _ in range(2):
            get_loss_scriptable(q_loc, q_scale, p_loc, p_scale, z_samples, mixture_out, mixture_temperature,
                                regularizer, mixture_regularizer_weight, log_pi, log2)
    print(p.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
