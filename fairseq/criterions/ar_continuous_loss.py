import math
from dataclasses import field
from typing import Optional

import torch.nn
import torch.nn.functional as F
from torch.distributions import multivariate_normal

from dataclasses import dataclass
from fairseq import metrics, utils
from fairseq.criterions import register_criterion, FairseqCriterion
from fairseq.criterions.cross_entropy import CrossEntropyCriterion, CrossEntropyCriterionConfig

import power_spherical

from fairseq.dataclass import FairseqDataclass


@register_criterion("cosine_ar_criterion")
class CosineARCriterion(CrossEntropyCriterion):

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, logging = self.compute_loss(model, net_output, sample)

        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": 1,
        }
        if type(logging) is dict:
            logging_output = {**logging_output, **logging}
        return loss, 1, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        target = model.get_targets(sample, net_output)
        mask = sample["target"].ne(self.padding_idx)
        out = net_output[0][mask]
        target = target[mask]
        loss = 1.0 - F.cosine_similarity(out, target, dim=-1)
        loss = loss.mean(-1)
        return loss, loss


@register_criterion("l2_ar_criterion")
class L2ARCriterion(CosineARCriterion):
    def compute_loss(self, model, net_output, sample, reduce=True):
        target = model.get_targets(sample, net_output)
        mask = sample["target"].ne(self.padding_idx)
        out = net_output[0][mask]
        target = target[mask]
        loss = torch.linalg.norm(out-target, dim=-1)
        loss = loss.mean(-1)
        return loss, loss


@dataclass
class ARPowerSphericalCriterionConfig(CrossEntropyCriterionConfig):
    kappa_lambda: float = field(
        default=0.0,
        metadata={"help": "weight for kappa reg term"},
    )


@register_criterion("ar_power_spherical_loss", dataclass=ARPowerSphericalCriterionConfig)
class ARPowerSphericalCriterion(CosineARCriterion):
    def __init__(self, task, sentence_avg, kappa_lambda):
        super().__init__(task, sentence_avg)
        self.kappa_lambda = kappa_lambda

    def compute_loss(self, model, net_output, sample, reduce=True):
        target = model.get_targets(sample, net_output)
        mask = sample["target"].ne(self.padding_idx)
        out = net_output[0]
        kappa = torch.norm(out, dim=-1)
        target = target[mask]
        kappa = kappa[mask]
        out = out[mask]

        out = F.normalize(out, dim=-1)

        d_x = power_spherical.PowerSpherical(out, kappa)

        loss = -d_x.log_prob(target)
        loss = loss.mean(-1) + (self.kappa_lambda*torch.pow(kappa, 2)).mean(-1)

        logging = {"kappa": kappa.mean(-1)}

        return loss, logging

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        kappa = sum(log.get("kappa", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar('kappa', kappa / sample_size, sample_size, round=3)

        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
            )


@register_criterion("ar_kl_criterion")
class KLCriterion(FairseqCriterion):
    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, logging = self.compute_loss(model, net_output, sample)

        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": 1,
        }
        if type(logging) is dict:
            logging_output = {**logging_output, **logging}
        return loss, 1, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        target = model.get_targets(sample, net_output)
        mask = sample["target"].ne(self.padding_idx)
        out = net_output[0]

        loss, logging = self.compute_kl(out, target, mask)

        return loss, logging


@dataclass
class ARPowerSphericalRevKLCriterionConfig(FairseqDataclass):
    n_samples: int = field(
        default=1,
        metadata={"help": "number of samples"},
    )
    kldirection: Optional[str] = field(
        default="reverse",
        metadata={"help": "forward or reverse KL"},
    )
    target_kappa_scale: Optional[float] = field(
        default=100.0,
        metadata={"help": "scaler to use for target emb norm"},
    )


@register_criterion("ar_power_spherical_klmc_loss", dataclass=ARPowerSphericalRevKLCriterionConfig)
class ARPowerSphericalRevKLCriterion(KLCriterion):
    def __init__(self, task, n_samples, kldirection, target_kappa_scale):
        super().__init__(task)
        self.n_samples = n_samples
        self.direction = kldirection
        self.target_kappa_scale = target_kappa_scale

    def set_dist(self, out, target, kappa):
        norm_target = torch.norm(target, dim=-1)
        target_scale = self.target_kappa_scale * norm_target
        p = power_spherical.PowerSpherical(target,
                                           target_scale)
        q = power_spherical.PowerSpherical(out,
                                           kappa)

        if self.direction == "reverse":
            return p, q
        else:
            return q, p

    def compute_kl(self, out, target, mask):
        target = target[mask]
        out = out[mask]

        kappa = torch.norm(out, dim=-1)
        kappa = kappa[mask]

        out = F.normalize(out, dim=-1)
        p, q = self.set_dist(out, target, kappa)
        z = q.rsample((self.n_samples,))
        log_pz = p.log_prob(z)

        entropy = q.entropy()
        neg_ce = log_pz.mean(0)
        loss = -entropy - neg_ce
        loss = loss.mean(-1)
        return loss, {"kappa": kappa.mean(-1)}


@register_criterion("ar_gaussian_kl_loss")
class ARGaussianvKLCriterion(KLCriterion):
    def compute_kl(self, out, target, mask):
        target = target[mask]
        out = out[mask]

        var_out = torch.norm(out, dim=-1)
        var_target = torch.norm(target, dim=-1)

        p = multivariate_normal.MultivariateNormal(loc=target, covariance_matrix=torch.diag(var_out))
        q = multivariate_normal.MultivariateNormal(loc=out, covariance_matrix=torch.diag(var_target))

        loss = torch.distributions.kl.kl_divergence(p, q)
        loss = loss.mean(-1)
        return loss, {}
