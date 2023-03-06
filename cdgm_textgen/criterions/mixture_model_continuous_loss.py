import math
from dataclasses import dataclass
from dataclasses import field
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from fairseq import metrics
from fairseq.criterions import register_criterion
from torch import Tensor

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
    mixture_temperature: Optional[float] = field(default=1.0, metadata={"help": "temperature for mixture prior"})


@register_criterion("power_spherical_klmc_mixture_loss", dataclass=MixturePSKLCriterionConfig)
class MixturePowerSphericalRevKLCriterion(ARPowerSphericalRevKLCriterion):
    def __init__(self, task, n_samples,
                 kldirection,
                 target_kappa_scale,
                 mixture_regularizer,
                 mixture_regularizer_weight,
                 mixture_temperature):
        super().__init__(task,
                         n_samples,
                         kldirection,
                         target_kappa_scale)
        self.mixture_regularizer = mixture_regularizer
        self.mixture_regularizer_weight = 0 if mixture_regularizer is None else mixture_regularizer_weight
        self.mixture_temperature = mixture_temperature

    def compute_loss(self, model, net_output, sample, reduce=True):
        target = model.get_targets(sample, net_output)
        mask = sample["target"].ne(self.padding_idx)
        out = net_output[0]
        mixture_out = net_output[1]["mixtures_out"]

        loss, logging = self.compute_kl(out, target, mask, mixture_out=mixture_out)

        return loss, logging

    @torch.jit.export
    def get_mixture_out_softmax(self, mixture_out: Tensor,
                                mixture_temperature: float) -> Tuple[Tensor, Tensor]:
        mixture_out_with_temp = mixture_out / mixture_temperature
        mixture_out_probs = F.softmax(mixture_out_with_temp, dim=-1)  # prior
        log_prior_probs = torch.log(mixture_out_probs)
        return mixture_out_probs, log_prior_probs

    @torch.jit.export
    def get_posterior_regularization(self,
            log_prior_probs: Tensor,
            log_q_out: Tensor,
            mixture_out_probs: Tensor,
            mixture_regularizer_weight: float) -> Tensor:
        lmodes_probs = log_prior_probs + log_q_out.t()
        lmodes_probs = lmodes_probs.t() - torch.logsumexp(lmodes_probs, dim=-1)
        entropy = -(mixture_out_probs * lmodes_probs.t()).mean(dim=-1)
        reg_term = (mixture_regularizer_weight * entropy).mean(dim=-1)
        return reg_term

    @torch.jit.export
    def get_prior_regularization(self,
                                 mixture_out_probs: Tensor,
                                 log_prior_probs: Tensor,
                                 mixture_regularizer_weight: float) -> Tensor:
        entropy = -(mixture_out_probs * log_prior_probs).mean(dim=-1)
        reg_term = (mixture_regularizer_weight * entropy).mean(dim=-1)
        return reg_term

    @torch.jit.export
    def loss_summation_scriptable(self,
            log_q_out_component: Tensor,
            log_prior_probs: Tensor,
            mixture_out_probs: Tensor,
            log_p_out: Tensor) -> Tuple[Tensor, Tensor, Tensor]:

        log_q_out = torch.logsumexp(log_q_out_component + log_prior_probs.transpose(0, 1)[None, :, :], dim=1)
        loss = torch.sum(mixture_out_probs * (log_q_out.t() - log_p_out.t()), dim=-1)
        prior_entropy = -(mixture_out_probs * log_prior_probs).mean(dim=-1)
        return loss, log_q_out, prior_entropy

    def compute_kl(self, out: Tensor, target: Tensor, mask: Tensor, mixture_out: Tensor = None):
        # samples are in format Mx(B*T)xD
        # get sample x_i for each mixture component i where i=1...M
        # get mixture components log prob q_m(x_i) for each sample component x_i i=1...M and mixture component m=1...M
        # get log Q(x_i) for i=1..M
        # where Q(x_i) = log \sum_m=1^M (w_m*q_m(x_i)) --> apply logsumexp
        # with torch.no_grad():
        #     from datetime import datetime
        #     lmodes_probs = F.log_softmax(mixture_out, dim=-1) + q_out
        #     lmodes_probs = lmodes_probs.t() - torch.logsumexp(lmodes_probs, dim=-1)
        #     torch.save(lmodes_probs, f"mixture_priors_{datetime.now().timestamp()}")
        #     print("post save")

        # apply padding mask
        target = target[mask]

        # x is in the format of MxBxTxD --> apply mask starting from dim=1
        out = out[:, mask]

        kappa = torch.norm(out, dim=-1)
        norm_target = torch.norm(target, dim=-1)
        target_kappa = self.target_kappa_scale * norm_target
        p, q = self.set_dist(out, target, kappa, target_kappa)
        if mixture_out is not None:
            mixture_out = mixture_out[mask]

        mixture_out_probs, log_prior_probs = self.get_mixture_out_softmax(mixture_out,
                                                                          self.mixture_temperature
                                                                          )
        z_samples = q.rsample()
        log_q_out_component = q.log_prob(z_samples.unsqueeze(1))
        log_p_out = p.log_prob(z_samples)

        loss, log_q_out, prior_entropy = self.loss_summation_scriptable(log_q_out_component,
                                                                        log_prior_probs,
                                                                        mixture_out_probs,
                                                                        log_p_out)
        if self.mixture_regularizer == "posterior":
            reg_term = self.get_posterior_regularization(log_prior_probs,
                                                         log_q_out,
                                                         mixture_out_probs,
                                                         self.mixture_regularizer_weight)
            loss = loss.mean(dim=-1) + reg_term
        elif self.mixture_regularizer == "prior":
            reg_term = self.get_prior_regularization(mixture_out_probs,
                                                     log_prior_probs,
                                                     self.mixture_regularizer_weight)
            loss = loss.mean(dim=-1) + reg_term

        else:
            loss = loss.mean(dim=-1)

        return loss, {"kappa": kappa.detach().mean(),
                      "prior_entropy": prior_entropy.detach().mean()}

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        kappa = sum(log.get("kappa", 0) for log in logging_outputs)
        prior_entropy = sum(log.get("prior_entropy", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )

        metrics.log_scalar('kappa', kappa / sample_size, sample_size, round=3)
        metrics.log_scalar("prior_entropy", prior_entropy / sample_size, sample_size, round=3)
