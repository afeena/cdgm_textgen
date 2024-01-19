from dataclasses import dataclass, field

import ledoh_torch
import torch
import torch.nn.functional as F
from fairseq import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass.configs import FairseqDataclass
from ledoh_torch import minimum_acos_distance, circular_variance, LloydSphereDispersion, \
    compute_sliced_sphere_dispersion, KernelSphereDispersion


@dataclass
class DistanceCriterionConfig(FairseqDataclass):
    distance_metric: str = field(default="cosine",
                                 metadata={"help": "Distance metric to use for the loss"})


@register_criterion("distance_criterion", dataclass=DistanceCriterionConfig)
class DistanceCriterion(FairseqCriterion):
    def __init__(self, task, distance_metric):
        super().__init__(task)
        assert distance_metric in ["cosine", "acos"], 'Distance metric must be either "cosine" or "acos"'
        self.distance_metric = distance_metric

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
        target = model.get_targets(sample, net_output)
        mask = sample["target"].ne(self.padding_idx)

        out = F.normalize(net_output[0][mask], dim=-1)
        target = F.normalize(target[mask], dim=-1)

        dot_prod = out @ target.T

        if self.distance_metric == "cosine":
            loss = (1.0 - dot_prod).diag().mean()

        elif self.distance_metric == "acos":
            loss = torch.acos(dot_prod.clamp(-1 + 1e-4, 1 - 1e-4)).diag().mean()

        return loss, {"dotprod": dot_prod.detach().mean()}

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        dotprod = sum(log.get("dotprod", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size, sample_size, round=3
        )
        metrics.log_scalar(
            "dotprod", dotprod / sample_size, sample_size, round=3
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


@dataclass
class DispersedDistCriterionConfig(DistanceCriterionConfig):
    dispersed_weight: float = field(
        default=0.0,
        metadata={"help": "weight for dispersed loss"},
    )
    report_dispersed: bool = field(default=True,
                                   metadata={"help": "report dispersed loss"})


@register_criterion("dist_criterion_with_dispersion", dataclass=DispersedDistCriterionConfig)
class DispersedDistanceCriterion(DistanceCriterion):
    def __init__(self, task, distance_metric, dispersed_weight, report_dispersed):
        super().__init__(task, distance_metric)
        self.dispersed_weight = dispersed_weight
        self.report_dispersed = report_dispersed

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, logging = self.compute_loss(model, net_output, sample)
        reg = None
        if self.dispersed_weight > 0:
            reg = (self.dispersed_weight * self.compute_dispersed_reg(model)).mean()
            loss = loss + reg

        logging_output = {
            "loss": loss.detach(),
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": 1,
            "regularizer": reg.detach() if reg is not None else 0
        }
        if self.report_dispersed:
            acos_min_dist, var = self.compute_dispersed_metrics(model)
            logging_output["acos_min_dist"] = acos_min_dist.detach()
            logging_output["circular_variance"] = var.detach()

        if type(logging) is dict:
            logging_output = {**logging_output, **logging}
        return loss, 1, logging_output

    @staticmethod
    def compute_dispersed_metrics(model):
        with torch.no_grad():
            min_cosine_dist = minimum_acos_distance(model.decoder.output_projection.weight.detach())
            var = circular_variance(model.decoder.output_projection.weight.detach())
        return min_cosine_dist, var

    def compute_dispersed_reg(self, model):
        raise NotImplementedError

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        dotprod = sum(log.get("dotprod", 0) for log in logging_outputs)
        acos_min_dist = sum(log.get("acos_min_dist", 0) for log in logging_outputs)
        circular_variance = sum(log.get("circular_variance", 0) for log in logging_outputs)
        regularize_value = sum(log.get("regularizer", 0) for log in logging_outputs)
        n_logs = len(logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size, sample_size, round=3
        )
        metrics.log_scalar(
            "dotprod", dotprod / sample_size, sample_size, round=3
        )
        metrics.log_scalar("regularizer", regularize_value / sample_size, sample_size, round=3)

        metrics.log_scalar("acos_min_dist", acos_min_dist / n_logs, n_logs, round=3)
        metrics.log_scalar("circular_variance", circular_variance / n_logs, n_logs, round=3)


@dataclass
class DistanceLloydCriterionConfig(DispersedDistCriterionConfig):
    n_lloyd_samples: int = field(
        default=100,
        metadata={"help": "weight for dispersed loss"},
    )


@register_criterion("dist_criterion_with_lloyd_dispersion", dataclass=DistanceLloydCriterionConfig)
class LloydDispersedCriterion(DispersedDistanceCriterion):
    def __init__(self, task, distance_metric, dispersed_weight, report_dispersed, n_lloyd_samples):
        super().__init__(task, distance_metric, dispersed_weight, report_dispersed)
        self.n_lloyd_samples = n_lloyd_samples

    def compute_dispersed_reg(self, model):
        lloyd_disp = LloydSphereDispersion.forward(model.decoder.output_projection.weight,
                                                   n_samples=self.n_lloyd_samples)
        return lloyd_disp


@register_criterion("dist_criterion_with_sliced_dispersion", dataclass=DispersedDistCriterionConfig)
class SlicedDispersedCriterion(DispersedDistanceCriterion):
    def compute_dispersed_reg(self, model):
        p, q = ledoh_torch.utils.init_great_circle(
            model.decoder.embed_dim,
            dtype=model.decoder.output_projection.weight.dtype,
            device=model.decoder.output_projection.weight.device)

        sliced = compute_sliced_sphere_dispersion(model.decoder.output_projection.weight, p, q)
        return sliced


@dataclass
class DistanceKernelCriterionConfig(DispersedDistCriterionConfig):
    cosine_gamma: float = field(default=0.001, metadata={"help": "gamma for cosine kernel"})
    kernel_sample_size: int = field(default=-1, metadata={"help": "number of samples for kernel dispersion"})


@register_criterion("dist_criterion_with_kernel_dispersion", dataclass=DistanceKernelCriterionConfig)
class KernelDispersedCriterion(DispersedDistanceCriterion):
    def __init__(self, task, distance_metric, dispersed_weight, report_dispersed, cosine_gamma, kernel_sample_size):
        super().__init__(task, distance_metric, dispersed_weight, report_dispersed)
        self.cosine_gamma = cosine_gamma
        self.kernel_sample_size = kernel_sample_size

    def compute_dispersed_reg(self, model):
        kernel_disp = KernelSphereDispersion.forward(model.decoder.output_projection.weight,
                                                     gamma=self.cosine_gamma, batch_size=self.kernel_sample_size)
        return kernel_disp
