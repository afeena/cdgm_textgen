import math
from dataclasses import dataclass
from dataclasses import field
from typing import Optional

import geoopt.manifolds.stereographic.math as pmath
import power_spherical
import torch
import torch.nn.functional as F
from fairseq import metrics
from fairseq.criterions import register_criterion, FairseqCriterion
from fairseq.criterions.cross_entropy import CrossEntropyCriterionConfig
from fairseq.dataclass import FairseqDataclass
from geoopt import StereographicExact
from torch.distributions import multivariate_normal


from cdgm_textgen.criterions.distance_loss import DistanceCriterion

@register_criterion("manifold_dist_criterion")
class ManifoldDistanceCriterion(DistanceCriterion):

    def compute_loss(self, model, net_output, sample, reduce=True):
        target = model.get_targets(sample, net_output)

        mask = sample["target"].ne(self.padding_idx)

        out = net_output[0][mask]
        target = target[mask]

        # sanity check
        if not model.decoder.manifold.check_point_on_manifold(out):
            if isinstance(model.decoder.manifold, StereographicExact):
                out = pmath.project(model.decoder.manifold.expmap0(out), k=model.decoder.manifold.k)
            else:
                raise ValueError(f'out is not on manifold {model.decoder.manifold}')
        if not model.decoder.manifold.check_point_on_manifold(target):
            if isinstance(model.decoder.manifold, StereographicExact):
                target = pmath.project(model.decoder.manifold.expmap0(target), k=model.decoder.manifold.k)
            else:
                raise ValueError(f'target is not on manifold {model.decoder.manifold}')

        loss = model.decoder.manifold.dist(target, out).mean()
        return loss, None


@register_criterion("manifold_dist2_criterion")
class ManifoldDistance2Criterion(DistanceCriterion):

    def compute_loss(self, model, net_output, sample, reduce=True):
        target = model.get_targets(sample, net_output)

        mask = sample["target"].ne(self.padding_idx)

        out = net_output[0][mask]
        target = target[mask]

        # sanity check
        if not model.decoder.manifold.check_point_on_manifold(out):
            if isinstance(model.decoder.manifold, StereographicExact):
                out = pmath.project(model.decoder.manifold.expmap0(out), k=model.decoder.manifold.k)
            else:
                raise ValueError(f'out is not on manifold {model.decoder.manifold}')
        if not model.decoder.manifold.check_point_on_manifold(target):
            if isinstance(model.decoder.manifold, StereographicExact):
                target = pmath.project(model.decoder.manifold.expmap0(target), k=model.decoder.manifold.k)
            else:
                raise ValueError(f'target is not on manifold {model.decoder.manifold}')

        loss = model.decoder.manifold.dist2(target, out).mean()
        return loss, None


@register_criterion("sphere_dist_criterion")
class SphereDistanceCriterion(DistanceCriterion):

    def compute_loss(self, model, net_output, sample, reduce=True):
        target = model.get_targets(sample, net_output)

        mask = sample["target"].ne(self.padding_idx)

        out = net_output[0][mask]
        target = target[mask]

        # sanity check
        if not model.decoder.manifold.check_point_on_manifold(out):
            out = F.normalize(out, dim=-1)
        if not model.decoder.manifold.check_point_on_manifold(target):
            target = F.normalize(target, dim=-1)

        loss = model.decoder.manifold.dist(target, out).mean()
        return loss, None


@register_criterion("sphere_dist2_criterion")
class SphereDistance2Criterion(DistanceCriterion):

    def compute_loss(self, model, net_output, sample, reduce=True):
        target = model.get_targets(sample, net_output)

        mask = sample["target"].ne(self.padding_idx)

        out = net_output[0][mask]
        target = target[mask]

        # sanity check
        if not model.decoder.manifold.check_point_on_manifold(out):
            out = F.normalize(out, dim=-1)
        if not model.decoder.manifold.check_point_on_manifold(target):
            target = F.normalize(target, dim=-1)

        loss = model.decoder.manifold.dist2(target, out).mean()
        return loss, None


@register_criterion("l2_ar_criterion")
class L2ARCriterion(DistanceCriterion):
    def compute_loss(self, model, net_output, sample, reduce=True):
        target = model.get_targets(sample, net_output)
        mask = sample["target"].ne(self.padding_idx)
        out = net_output[0][mask]
        target = target[mask]
        loss = torch.norm(out - target, p=2.0, dim=-1)
        loss = loss.mean(-1)
        return loss, None


@register_criterion("l2_squared_criterion")
class L2SquaredARCriterion(DistanceCriterion):
    def compute_loss(self, model, net_output, sample, reduce=True):
        target = model.get_targets(sample, net_output)
        mask = sample["target"].ne(self.padding_idx)
        out = net_output[0][mask]
        target = target[mask]
        loss = torch.pow(torch.norm(out - target, p=2.0, dim=-1), 2)
        loss = loss.mean(-1)
        return loss, None


@dataclass
class ARPowerSphericalCriterionConfig(CrossEntropyCriterionConfig):
    kappa_lambda: float = field(
        default=0.0,
        metadata={"help": "weight for kappa reg term"},
    )


@register_criterion("ar_power_spherical_loss", dataclass=ARPowerSphericalCriterionConfig)
class ARPowerSphericalCriterion(DistanceCriterion):
    def __init__(self, task, distance_metric, kappa_lambda):
        super().__init__(task, distance_metric)
        self.kappa_lambda = kappa_lambda

    def compute_loss(self, model, net_output, sample, reduce=True):
        target = model.get_targets(sample, net_output)
        mask = sample["target"].ne(self.padding_idx)
        out = net_output[0]
        kappa = model.forward_kappa(out)
        target = target[mask]
        kappa = kappa[mask]
        out = out[mask]

        out = F.normalize(out, dim=-1)

        d_x = power_spherical.PowerSpherical(out, kappa)

        loss = -d_x.log_prob(target)
        reg_term = (self.kappa_lambda * kappa).mean(dim=-1)
        loss = loss.mean(dim=-1) + reg_term.mean(dim=-1)

        logging = {"kappa": kappa.detach().mean(dim=-1)}

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
        metrics.log_scalar('kappa', kappa / sample_size, sample_size, round=3)


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
