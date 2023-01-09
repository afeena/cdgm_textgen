from dataclasses import dataclass
from dataclasses import field
from typing import Optional

import torch
import torch.nn.functional as F
from fairseq.criterions import register_criterion

from cdgm_textgen.criterions import ARPowerSphericalRevKLCriterionConfig, \
    ARPowerSphericalRevKLCriterion


@dataclass
class MixturePSKLCriterionConfig(ARPowerSphericalRevKLCriterionConfig):
    mixture_regularizer: Optional[str] = field(
        default=None,
        metadata={"help": "type of mixture reg: posterior or prior"},
    )
    mixture_regularizer_weight: Optional[float] = field(
        default=0.1,
        metadata={"help": "weight of mixture reg"},
    )


@register_criterion("power_spherical_klmc_mixture_loss", dataclass=MixturePSKLCriterionConfig)
class MixturePowerSphericalRevKLCriterion(ARPowerSphericalRevKLCriterion):
    def __init__(self, task, n_samples,
                 kldirection,
                 target_kappa_scale,
                 mixture_regularizer,
                 mixture_regularizer_weight):
        super().__init__(task,
                         n_samples,
                         kldirection,
                         target_kappa_scale)
        self.mixture_regularizer = mixture_regularizer
        self.mixture_regularizer_weight = 0 if mixture_regularizer is None else mixture_regularizer_weight

    def compute_loss(self, model, net_output, sample, reduce=True):
        target = model.get_targets(sample, net_output)
        mask = sample["target"].ne(self.padding_idx)
        out = net_output[0]
        mixture_out = net_output[1]["mixtures_out"]

        loss, logging = self.compute_kl(out, target, mask, mixture_out=mixture_out)

        return loss, logging

    def compute_kl(self, out, target, mask, mixture_out=None):
        # apply padding mask
        target = target[mask]

        # transpose output to have mixture component M as a first dim
        # Mx(B*T)xD
        out = out[mask].transpose(0, 1)

        kappa = torch.norm(out, dim=-1)
        p, q = self.set_dist(out, target, kappa)

        mixture_out = mixture_out[mask]
        mixture_out_probs = F.softmax(mixture_out, dim=-1)  # prior
        log_prior_probs = F.log_softmax(mixture_out, dim=-1)

        # samples are in format Mx(B*T)xD
        # get sample x_i for each mixture component i where i=1...M
        z_samples = q.rsample()
        # get mixture components log prob q_m(x_i) for each sample component x_i i=1...M and mixture component m=1...M
        log_q_out_component = q.log_prob(z_samples.unsqueeze(1))
        # get log Q(x_i) for i=1..M
        # where Q(x_i) = log \sum_m=1^M (w_m*q_m(x_i)) --> apply logsumexp
        log_q_out = torch.logsumexp(log_q_out_component + log_prior_probs.transpose(0, 1)[None, :, :], dim=1)

        p_out = p.log_prob(z_samples)

        loss = torch.sum(mixture_out_probs * (log_q_out.t() - p_out.t()), dim=-1)

        # with torch.no_grad():
        #     from datetime import datetime
        #     lmodes_probs = F.log_softmax(mixture_out, dim=-1) + q_out
        #     lmodes_probs = lmodes_probs.t() - torch.logsumexp(lmodes_probs, dim=-1)
        #     torch.save(lmodes_probs, f"mixture_priors_{datetime.now().timestamp()}")
        #     print("post save")

        if self.mixture_regularizer == "posterior":
            lmodes_probs = F.log_softmax(mixture_out, dim=-1) + log_q_out.t()
            lmodes_probs = lmodes_probs.t() - torch.logsumexp(lmodes_probs, dim=-1)
            entropy_reg = (mixture_out_probs * lmodes_probs.t()).mean(dim=-1)
            loss = loss.mean(dim=-1) + (self.mixture_regularizer_weight * entropy_reg).mean(dim=-1)
        elif self.mixture_regularizer == "prior":
            entropy_reg = (mixture_out_probs * F.log_softmax(mixture_out, dim=-1)).mean(dim=-1)
            loss = loss.mean(dim=-1) + (self.mixture_regularizer_weight * entropy_reg).mean(dim=-1)
        else:
            loss = loss.mean(dim=-1)

        return loss, {"kappa": kappa.detach().mean()}
