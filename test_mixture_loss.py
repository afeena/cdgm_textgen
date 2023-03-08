from typing import Tuple

import torch
from torch import Tensor
import torch.nn.functional as F
import math

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

    alphaq = (q_loc.shape[-1] - 1) / 2 + q_scale
    # here we assume beta q = beta p since both has the same dimension
    beta = (q_loc.shape[-1] - 1) / 2

    log_q_prob = -(
            (alphaq + beta) * log2
            + torch.lgamma(alphaq)
            - torch.lgamma(alphaq + beta)
            + beta * log_pi
    ) + q_scale * torch.log1p(
        (q_loc * z_samples.unsqueeze(1)).sum(-1)
    )

    alphap = (p_loc.shape[-1] - 1) / 2 + p_scale

    log_p_prob = -(
            (alphap + beta) * log2
            + torch.lgamma(alphap)
            - torch.lgamma(alphap + beta)
            + beta * log_pi
    ) + p_scale * torch.log1p(
        (p_loc * z_samples).sum(-1)
    )

    log_q_out = torch.logsumexp(log_q_prob + log_prior_probs.transpose(0, 1)[None, :, :], dim=1)
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

if __name__=="__main__":
    q_loc= torch.rand(128, 64, 128)
    q_scale= torch.rand(128, 64)
    p_loc= torch.rand(64, 128)
    p_scale= torch.rand(64,)
    z_samples= torch.rand(128, 64, 128)
    mixture_out = torch.rand(64, 128)
    mixture_temperature = 1.0
    regularizer = 'none'
    mixture_regularizer_weight: 0.0
    log_pi = math.log(math.pi)
    log2= math.log(2.0)

    get_loss_scriptable(q_loc, q_scale, p_loc, p_scale, z_samples, mixture_out, mixture_temperature, regularizer, mixture_regularizer_weight, log_pi, log2)