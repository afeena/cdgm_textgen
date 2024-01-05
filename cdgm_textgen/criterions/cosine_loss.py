import math
from dataclasses import field

import ledoh_torch
import torch
import torch.nn.functional as F
from fairseq import metrics
from fairseq.criterions import register_criterion
from fairseq.criterions.cross_entropy import CrossEntropyCriterion, CrossEntropyCriterionConfig
from ledoh_torch import LloydSphereDispersion, \
    KernelSphereDispersion, \
    compute_sliced_sphere_dispersion, \
    minimum_acos_distance, \
    circular_variance

from dataclasses import dataclass


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

        out = net_output[0][mask]
        target = target[mask]
        loss = 1.0 - F.cosine_similarity(out, target, dim=-1)
        loss = loss.mean(-1)
        return loss, None

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )

@dataclass
class CEDispersedCriterionConfig(CrossEntropyCriterionConfig):
    dispersed_weight: float = field(
        default=0.0,
        metadata={"help": "weight for dispersed loss"},
    )
    report_dispersed: bool = field(default=True,
                                   metadata={"help": "report dispersed loss"})
@register_criterion("cosine_dispersed_criterion", dataclass=CEDispersedCriterionConfig)
class CosineDispersedCriterion(CosineARCriterion):
    def __init__(self, task, sentence_avg, dispersed_weight, report_dispersed):
        super().__init__(task, sentence_avg)
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
        reg = self.dispersed_weight * self.compute_dispersed_reg(model)
        loss = loss + reg

        logging_output = {
            "loss": loss.detach(),
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": 1,
        }
        if self.report_dispersed:
            cosine_min_dist, var = self.compute_dispersed_metrics(model)
            logging_output["cosine_min_dist"] = cosine_min_dist.detach()
            logging_output["circular_variance"] = var.detach()

        if type(logging) is dict:
            logging_output = {**logging_output, **logging}
        return loss, 1, logging_output

    def compute_dispersed_metrics(self, model):
        with torch.no_grad():
            min_cosine_dist = minimum_acos_distance(model.decoder.embed_tokens.weight.detach())
            var = circular_variance(model.decoder.embed_tokens.weight.detach())
        return min_cosine_dist, var
    def compute_dispersed_reg(self, model):
        raise NotImplementedError
    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        dispersed_min_dist = sum(log.get("cosine_min_dist", 0) for log in logging_outputs)
        dispersed_mean_v = sum(log.get("circular_variance", 0) for log in logging_outputs)
        n_logs = len(logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar("min_dist", dispersed_min_dist / n_logs, sample_size, round=4)
        metrics.log_scalar("mean dist val", dispersed_mean_v / n_logs, sample_size, round=4)
@dataclass
class CELloydDispersedCriterionConfig(CEDispersedCriterionConfig):
    n_lloyd_samples: int = field(
        default=100,
        metadata={"help": "weight for dispersed loss"},
    )

@register_criterion("cosine_lloyd_dispersed_criterion", dataclass=CELloydDispersedCriterionConfig)
class CosineLloydDispersedCriterion(CosineDispersedCriterion):
    def __init__(self, task, sentence_avg, dispersed_weight, report_dispersed, n_lloyd_samples):
        super().__init__(task, sentence_avg, dispersed_weight, report_dispersed)
        self.n_lloyd_samples = n_lloyd_samples
    def compute_dispersed_reg(self, model):
        lloyd_disp = LloydSphereDispersion.forward(model.decoder.embed_tokens.weight,
                                                      n_samples=self.n_lloyd_samples)
        return lloyd_disp


@register_criterion("cosine_sliced_dispersed_criterion")
class CosineSlicedDispersedCriterion(CosineDispersedCriterion):
    def __init__(self, task, sentence_avg, dispersed_weight, report_dispersed):
        super().__init__(task, sentence_avg, dispersed_weight, report_dispersed)
        self.p = None
        self.q = None
    def compute_dispersed_reg(self, model):
        if self.p is None or self.q is None:
            self.p, self.q = ledoh_torch.utils.init_great_circle(
                model.decoder.embed_dim,
                dtype=model.decoder.embed_tokens.weight.dtype,
                device=model.decoder.embed_tokens.weight.device)
        sliced = compute_sliced_sphere_dispersion(model.decoder.embed_tokens.weight)
        return sliced

@dataclass
class CEKernelCriterionConfig(CEDispersedCriterionConfig):
    cosine_gamma: float = field(default=0.001, metadata={"help": "gamma for cosine kernel"})
    kernel_sample_size: int = field(default=-1, metadata={"help": "number of samples for kernel dispersion"})

@register_criterion("cosine_kernel_dispersed_criterion", dataclass=CEKernelCriterionConfig)
class CosineKernelDispersedCriterion(CosineDispersedCriterion):
    def __init__(self, task, sentence_avg, dispersed_weight, report_dispersed, cosine_gamma, kernel_sample_size):
        super().__init__(task, sentence_avg, dispersed_weight, report_dispersed)
        self.cosine_gamma = cosine_gamma
        self.kernel_sample_size = kernel_sample_size
    def compute_dispersed_reg(self, model):
        kernel_disp = KernelSphereDispersion.forward(model.decoder.embed_tokens.weight,
                                                      gamma=self.cosine_gamma, batch_size=self.kernel_sample_size)
        return kernel_disp