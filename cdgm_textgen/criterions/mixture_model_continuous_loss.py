import math
from dataclasses import dataclass
from dataclasses import field
from typing import Optional, Tuple

import power_spherical
import torch
import torch.nn.functional as F
from fairseq import metrics
from fairseq.criterions import register_criterion
from torch import Tensor

from cdgm_textgen.criterions import (ARPowerSphericalRevKLCriterionConfig,
                                     ARPowerSphericalRevKLCriterion, ARPowerSphericalCriterion,
                                     ARPowerSphericalCriterionConfig)
from cdgm_textgen.modules.utils import (softmax_with_temperature_scriptable,
                                        log_softmax_with_temperature_scriptable,
                                        log_prob_scriptable)


@torch.jit.script
def get_loss_scriptable(q_loc: Tensor, q_scale: Tensor,
                        p_loc: Tensor, p_scale: Tensor,
                        z_samples: Tensor,
                        mixture_out: Tensor,
                        mixture_temperature: float,
                        ) -> Tuple[Tensor, Tensor, Tensor]:
    mixture_out_probs = softmax_with_temperature_scriptable(mixture_out, temperature=mixture_temperature)
    log_prior_probs = log_softmax_with_temperature_scriptable(mixture_out, temperature=mixture_temperature)

    log_q_prob = log_prob_scriptable(q_loc, q_scale, z_samples, "ijk, ljk -> lij")
    log_p_prob = log_prob_scriptable(p_loc, p_scale, z_samples, "jk, ljk -> lj")

    prob_sum = log_q_prob + log_prior_probs.transpose(0, 1).unsqueeze(0)
    log_q_out = torch.logsumexp(prob_sum, dim=1)
    loss = torch.sum(mixture_out_probs * (log_q_out.t() - log_p_prob.t()), dim=-1)
    prior_entropy = -(mixture_out_probs * log_prior_probs).mean(dim=-1)

    lmodes_probs = log_prior_probs + log_q_out.t()
    lmodes_probs = lmodes_probs.t() - torch.logsumexp(lmodes_probs, dim=-1)
    posterior = -(mixture_out_probs * lmodes_probs.t()).mean(dim=-1)

    return loss, prior_entropy, posterior


@register_criterion("power_spherical_mixture_loss", dataclass=ARPowerSphericalCriterionConfig)
class MixturePowerSphericalCriterion(ARPowerSphericalCriterion):
    def __init__(self, task, sentence_avg, kappa_lambda):
        super().__init__(task,
                         sentence_avg,
                         kappa_lambda)

    def compute_loss(self, model, net_output, sample, reduce=True):
        target = model.get_targets(sample, net_output)
        mask = sample["target"].ne(self.padding_idx)
        out = net_output[0]
        mixture_out = net_output[1]["mixtures_out"]
        target = target[mask]

        # out is in a shape MxBxTxD, take mask only over last 3 dims
        out = out[:, mask]
        mixture_out = mixture_out[mask]
        kappa = torch.norm(out, dim=-1)
        out = F.normalize(out, dim=-1)

        d_x = power_spherical.PowerSpherical(out, kappa)

        # Q(x) = \sum_{m=1}^M p_m Q_m(x)
        # logQ(x) = log(\sum_{m=1}^M p_m Q_m(x))

        q_outs = d_x.log_prob(target)
        p_outs = F.log_softmax(mixture_out, dim=-1).transpose(0, 1).unsqueeze(0)
        loss = -torch.logsumexp(q_outs+p_outs, dim=1)
        reg_term = self.kappa_lambda * (torch.pow(kappa, 2).mean(dim=-1))
        loss = loss.mean(dim=-1) + reg_term.mean(dim=-1)

        logging = {"kappa": kappa.detach().mean()}

        return loss, logging

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        kappa = sum(log.get("kappa", 0) for log in logging_outputs)
        prior_entropy = sum(log.get("prior_entropy", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )

        metrics.log_scalar('kappa', kappa / sample_size, sample_size, round=3)
        metrics.log_scalar("prior_entropy", prior_entropy / sample_size, sample_size, round=3)


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

    def compute_kl(self, out: Tensor, target: Tensor, mask: Tensor, mixture_out: Tensor = None):
        """
        out: Tensor(MxBxTxD) - output of the model where
        M - number of mixtures,
        B - batch size, T - sequence length, D - out embeddings dim
        target: Tensor(BxTxD) - target embeddings tensor
        mask: Tensor(BxT) - padding mask acording to target
        mixture_out: Tensor(BxTxM) - mixture logits
        return: Tuple[Tensor(1).Dict] - loss tensor and logging dict

        Calculate KL divergence between out (mixture of PS distribution) and target (PS distribution)
        1. get sample x_i for each mixture component i, where i=1...M
        2. get mixture components log probabilities q_m(x_i) for each sample component x_i i=1...M and mixture component m=1...M
        3. get log Q(x_i) for i=1..M, where Q(x_i) = log \sum_m=1^M (w_m*q_m(x_i)) --> apply logsumexp

        to save mixture priors you can use the following code:
        # with torch.no_grad():
        #     from datetime import datetime
        #     lmodes_probs = F.log_softmax(mixture_out, dim=-1) + q_out
        #     lmodes_probs = lmodes_probs.t() - torch.logsumexp(lmodes_probs, dim=-1)
        #     torch.save(lmodes_probs, f"mixture_priors_{datetime.now().timestamp()}")
        #     print("post save")
        """

        # apply padding mask
        target = target[mask]
        # x is in the format of MxBxTxD --> apply mask starting from dim=1
        out = out[:, mask]

        kappa = torch.norm(out, dim=-1)
        norm_target = torch.norm(target, dim=-1)
        target_kappa = self.target_kappa_scale * norm_target
        p, q = self.set_dist(out, target, kappa, target_kappa)
        z_samples = q.rsample()

        if mixture_out is not None:
            mixture_out = mixture_out[mask]

        loss, prior_entropy, posterior = get_loss_scriptable(q.loc, q.scale,
                                                             p.loc, p.scale,
                                                             z_samples,
                                                             mixture_out,
                                                             self.mixture_temperature
                                                             )
        loss = loss.mean(dim=-1)
        # torchscript cannot handle control flow, calculate reg term only here
        reg = None
        if self.mixture_regularizer == "prior":
            reg = (self.mixture_regularizer_weight * prior_entropy).mean(dim=-1)
        elif self.mixture_regularizer == "posterior":
            reg = (self.mixture_regularizer_weight * posterior).mean(dim=-1)

        if reg:
            loss += reg

        return loss, {"kappa": kappa.detach().mean(),
                      "prior_entropy": prior_entropy.detach().mean()}

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        kappa = sum(log.get("kappa", 0) for log in logging_outputs)
        prior_entropy = sum(log.get("prior_entropy", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )

        metrics.log_scalar('kappa', kappa / sample_size, sample_size, round=3)
        metrics.log_scalar("prior_entropy", prior_entropy / sample_size, sample_size, round=3)
