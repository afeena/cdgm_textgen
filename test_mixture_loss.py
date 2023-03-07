from typing import Tuple

import power_spherical
import torch
from torch import Tensor
import torch.nn.functional as F
import torch.autograd.profiler as profiler

def get_mixture_out_softmax(mixture_out: Tensor,
                            mixture_temperature: float) -> Tuple[Tensor, Tensor]:
    mixture_out_with_temp = mixture_out / mixture_temperature
    mixture_out_probs = F.softmax(mixture_out_with_temp, dim=-1)  # prior
    log_prior_probs = torch.log(mixture_out_probs)
    return mixture_out_probs, log_prior_probs


def get_posterior_regularization(
        log_prior_probs: Tensor,
        log_q_out: Tensor,
        mixture_out_probs: Tensor,
        mixture_regularizer_weight: float) -> Tensor:
    lmodes_probs = log_prior_probs + log_q_out.t()
    lmodes_probs = lmodes_probs.t() - torch.logsumexp(lmodes_probs, dim=-1)
    entropy = -(mixture_out_probs * lmodes_probs.t()).mean(dim=-1)
    reg_term = (mixture_regularizer_weight * entropy).mean(dim=-1)
    return reg_term


def get_prior_regularization(
                             mixture_out_probs: Tensor,
                             log_prior_probs: Tensor,
                             mixture_regularizer_weight: float) -> Tensor:
    entropy = -(mixture_out_probs * log_prior_probs).mean(dim=-1)
    reg_term = (mixture_regularizer_weight * entropy).mean(dim=-1)
    return reg_term


def loss_summation_scriptable(
        log_q_out_component: Tensor,
        log_prior_probs: Tensor,
        mixture_out_probs: Tensor,
        log_p_out: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    log_q_out = torch.logsumexp(log_q_out_component + log_prior_probs.transpose(0, 1)[None, :, :], dim=1)
    loss = torch.sum(mixture_out_probs * (log_q_out.t() - log_p_out.t()), dim=-1)
    prior_entropy = -(mixture_out_probs * log_prior_probs).mean(dim=-1)

    return loss, log_q_out, prior_entropy

def get_samples_and_probs_q(q, z_samples):
    with profiler.record_function("log_normalizer"):
        ln = q.log_normalizer()
    with profiler.record_function("log_prob_mult"):
        q_log_prob = q.scale * torch.log1p(
            (q.loc * z_samples.unsqueeze(1)).sum(-1)
        )
    with profiler.record_function("log_prob_sum"):
        log_q_out_component = ln+q_log_prob

    return log_q_out_component

def set_dist(out, target):
    kappa = torch.norm(out, dim=-1)
    norm_target = torch.norm(target, dim=-1)
    target_kappa = 100.0 * norm_target

    out = F.normalize(out, dim=-1)
    target = F.normalize(target, dim=-1)
    p = power_spherical.PowerSpherical(target,
                                       target_kappa)
    q = power_spherical.PowerSpherical(out,
                                       kappa)
    return q, p
    
def compute_kl():
    out = torch.randn((128, 512, 128)).cuda()
    target = torch.randn((128, 512, 128)).cuda()
    mixture_out = torch.randn((512, 128)).cuda()

    with profiler.record_function("set_dist"):
        q,p = set_dist(out, target)

    with profiler.record_function("get_mixture_out_softmax"):
        mixture_out_probs, log_prior_probs = get_mixture_out_softmax(mixture_out,
                                                                      1.0
                                                                      )

    z_samples = q.rsample()
    with profiler.record_function("get_samples_and_probs"):
        log_q_out_component = get_samples_and_probs_q(q,z_samples)

    log_p_out = p.log_prob(z_samples)

    with profiler.record_function("loss_summation_scriptable"):
        loss, log_q_out, prior_entropy = loss_summation_scriptable(log_q_out_component,
                                                                        log_prior_probs,
                                                                        mixture_out_probs,
                                                                        log_p_out)

    loss = loss.mean(dim=-1)

if __name__ == "__main__":
    with profiler.profile(use_cuda=True, with_stack=True, profile_memory=True) as prof:
        compute_kl()
    print(prof.export_chrome_trace("trace.json"))

