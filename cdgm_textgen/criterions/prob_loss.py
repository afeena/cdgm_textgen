import math
from dataclasses import dataclass
from dataclasses import field
from typing import Optional

import power_spherical
import torch
import torch.nn.functional as F
from fairseq import metrics
from fairseq.criterions import register_criterion, FairseqCriterion
from fairseq.dataclass import FairseqDataclass
from torch.distributions import multivariate_normal
from cdgm_textgen.criterions.distance_loss import DispersedDistCriterionConfig, \
    DispersedDistanceCriterion


@dataclass
class DispersedProbCriterionConfig(DispersedDistCriterionConfig):
    prob_direction: Optional[str] = field(default="out", metadata={"help": "direction of the loss, either 'out' or 'target'"})

@register_criterion("ar_power_spherical_loss", dataclass=DispersedProbCriterionConfig)
class PowerSphericalCriterion(DispersedDistanceCriterion):
    def __init__(self, task, distance_metric,
                 dispersed_weight,
                 report_dispersed,
                 dispersion_type, dispersion_features, prob_direction):
        super().__init__(task, distance_metric, dispersed_weight,
                 report_dispersed,
                 dispersion_type, dispersion_features)
        self.prob_direction = prob_direction


    def compute_loss(self, model, net_output, sample, reduce=True):
        target = model.get_targets(sample, net_output)
        mask = sample["target"].ne(self.padding_idx)
        out = net_output[0]
        kappa = model.get_kappa(out, target, self.prob_direction)
        target = target[mask]
        kappa = kappa[mask]
        out = out[mask]

        if self.prob_direction == "target":
            # If the direction is 'target', we need to compute the loss with respect to the target distribution
            loc = F.normalize(target, dim=-1)
        elif self.prob_direction == "out":
            # If the direction is 'out', we compute the loss with respect to the output distribution
            loc = F.normalize(out, dim=-1)

        d_x = power_spherical.PowerSpherical(loc, kappa)

        loss = -d_x.log_prob(target)
        logging = {"prob": loss.detach().mean(dim=-1)}
        if self.dispersion_weight > 0:
            # Compute the dispersion term
            dispersion = self.compute_dispersion(model, net_output, sample, reduce)
            logging["dispersion"] = dispersion.detach().mean(dim=-1)
            loss += self.dispersion_weight * dispersion

        loss = loss.mean(dim=-1)

        return loss, logging

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        prob_loss = sum(log.get("prob", 0) for log in logging_outputs)
        dispersion = sum(log.get("dispersion", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar("prob_loss", prob_loss / sample_size, sample_size, round=3)
        metrics.log_scalar("dispersion", dispersion / sample_size, sample_size, round=3)



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
            "loss": loss.detach(),
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": 1,
        }
        if type(logging) is dict:
            logging_output = {**logging_output, **logging}
        return loss, 1, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        raise NotImplementedError

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

    def set_dist(self, out, target, out_kappa, target_kappa):
        """
        out: model output
        target: gold reference
        out_kappa: scale parameter for model output
        """
        out = F.normalize(out, dim=-1)
        target = F.normalize(target, dim=-1)
        p = power_spherical.PowerSpherical(target,
                                           target_kappa)
        q = power_spherical.PowerSpherical(out,
                                           out_kappa)

        if self.direction == "reverse":
            return p, q
        else:
            return q, p

    def compute_loss(self, model, net_output, sample, reduce=True):
        target = model.get_targets(sample, net_output)
        mask = sample["target"].ne(self.padding_idx)
        out = net_output[0]

        loss, logging = self.compute_kl(out, target, mask)

        return loss, logging

    def compute_kl(self, out, target, mask, **kwargs):
        # apply padding mask
        target = target[mask]

        kappa = torch.norm(out, dim=-1)
        out = out[mask]
        kappa = kappa[mask]

        norm_target = torch.norm(target, dim=-1)
        target_kappa = self.target_kappa_scale * norm_target

        p, q = self.set_dist(out, target, kappa, target_kappa)
        z_samples = q.rsample((self.n_samples,))
        log_pz = p.log_prob(z_samples)

        neg_ce = log_pz.mean(0)
        entropy = -q.entropy()
        loss = entropy - neg_ce

        loss = loss.mean(dim=-1)

        return loss, {"kappa": kappa.detach().mean()}


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