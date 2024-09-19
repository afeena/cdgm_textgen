import math
from dataclasses import dataclass, field

import ledoh_torch.utils
import torch
from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.criterions.cross_entropy import CrossEntropyCriterionConfig, CrossEntropyCriterion
from ledoh_torch import (
    LloydSphereDispersion,
    KernelSphereDispersion,
    AxisAlignedBatchSphereDispersion,
    SlicedSphereDispersion,
    MMADispersion,
    MMCSDispersion,
    MHEDispersion,
    KoLeoDispersion,
    minimum_acos_distance_batch,
    circular_variance
)



@dataclass
class CrossEntropyDispersedCriterionConfig(CrossEntropyCriterionConfig):
    dispersed_weight: float = field(
        default=0.0,
        metadata={"help": "weight for dispersed loss"},
    )
    report_dispersed: bool = field(default=True,
                                   metadata={"help": "report dispersed loss"})


@register_criterion("cross_entropy_dispersed", dataclass=CrossEntropyDispersedCriterionConfig)
class CrossEntropyDispersedCriterion(CrossEntropyCriterion):
    def __init__(self,
                 task,
                 sentence_avg,
                 dispersed_weight=0.0,
                 report_dispersed=True):
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
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        if self.dispersed_weight > 0.0:
            dispersed_loss = self.compute_dispersed_loss(model)
            loss = loss + self.dispersed_weight * dispersed_loss.mean()

        sample_size = 1

        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }

        if self.report_dispersed:
            cosine_min_dist, var = self.compute_dispersed_metrics(model)
            logging_output["min_acos_dist"] = cosine_min_dist.detach()
            logging_output["circular_variance"] = var.detach()

        return loss, sample_size, logging_output

    def compute_dispersed_loss(self, model):
        raise NotImplementedError

    @staticmethod
    def compute_dispersed_metrics(model):
        with torch.no_grad():
            min_acos_dist = minimum_acos_distance_batch(model.decoder.embed_tokens.weight.detach(), 4096)
            var = circular_variance(model.decoder.embed_tokens.weight.detach())
        return min_acos_dist, var

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        dispersed_min_dist = sum(log.get("min_acos_dist", 0) for log in logging_outputs)
        dispersed_mean_v = sum(log.get("circular_variance", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_scalar(
            "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_scalar("min_dist", dispersed_min_dist / sample_size, sample_size, round=4)
        metrics.log_scalar("circular_var", dispersed_mean_v / sample_size, sample_size, round=4)


@dataclass
class CrossEntropyLloydCriterionConfig(CrossEntropyDispersedCriterionConfig):
    n_lloyd_samples: int = field(
        default=100,
        metadata={"help": "weight for dispersed loss"},
    )


@register_criterion("ce_lloyd_disp", dataclass=CrossEntropyLloydCriterionConfig)
class CrossEntropyLloydCriterion(CrossEntropyDispersedCriterion):
    def __init__(self,
                 task,
                 sentence_avg,
                 dispersed_weight=0.0,
                 report_dispersed=True,
                 n_lloyd_samples=100):
        super().__init__(task, sentence_avg,
                         dispersed_weight,
                         report_dispersed)
        self.lloyd = LloydSphereDispersion(n_samples=n_lloyd_samples)

    def compute_dispersed_loss(self, model):
        # first sample n uniform points on the sphere
        lloyd_disp = self.lloyd(model.decoder.embed_tokens.weight)
        return lloyd_disp


@register_criterion("ce_sliced_disp", dataclass=CrossEntropyDispersedCriterionConfig)
class CrossEntropySlicedCriterion(CrossEntropyDispersedCriterion):
    def __init__(self,
                 task,
                 sentence_avg,
                 dispersed_weight=0.0,
                 report_dispersed=True):
        super().__init__(task, sentence_avg,
                         dispersed_weight,
                         report_dispersed)
        self.sliced = SlicedSphereDispersion()

    def compute_dispersed_loss(self, model):
        p, q = ledoh_torch.utils.init_great_circle(
            model.decoder.embed_dim,
            dtype=model.decoder.output_projection.weight.dtype,
            device=model.decoder.output_projection.weight.device)

        sliced = self.sliced(model.decoder.output_projection.weight, p, q)
        return sliced

@dataclass
class CrossEntropySlicedAxCriterionConfig(CrossEntropyDispersedCriterionConfig):
    batch_size: int = field(
        default=-1,
        metadata={"help": "batch size for slicedax"}
    )
@register_criterion("ce_slicedax_disp", dataclass=CrossEntropySlicedAxCriterionConfig)
class CrossEntropySlicedAXCriterion(CrossEntropyDispersedCriterion):
    def __init__(self,
                 task,
                 sentence_avg,
                 label_smoothing,
                 ignore_prefix_size=0,
                 report_accuracy=False,
                 dispersed_weight=0.0,
                 report_dispersed=True,
                 batch_size=-1):
        super().__init__(task, sentence_avg,
                         dispersed_weight,
                         report_dispersed)
        self.slicedax = AxisAlignedBatchSphereDispersion(batch_size=batch_size)

    def compute_dispersed_loss(self, model):
        slicedax = self.slicedax(model.decoder.output_projection.weight)
        return slicedax

@dataclass
class CrossEntropyMMDCriterionConfig(CrossEntropyDispersedCriterionConfig):
    cosine_gamma: float = field(default=0.001, metadata={"help": "gamma for cosine kernel"})
    batch_size: int = field(default=-1, metadata={"help": "number of samples for kernel dispersion"})


@register_criterion("ce_mmd_disp", dataclass=CrossEntropyMMDCriterionConfig)
class CrossEntropyMMDCriterion(CrossEntropyDispersedCriterion):
    def __init__(self,
                 task,
                 sentence_avg,
                 dispersed_weight=0.0,
                 report_dispersed=True,
                 cosine_gamma=0.001,
                 batch_size=-1):
        super().__init__(task, sentence_avg, dispersed_weight,
                         report_dispersed)

        self.mmd = KernelSphereDispersion(gamma=cosine_gamma, batch_size=batch_size)

    def compute_dispersed_loss(self, model):
        mmd_loss = self.mmd(model.decoder.embed_tokens.weight)
        return mmd_loss

@dataclass
class CrossEntropyMMACriterionConfig(CrossEntropyDispersedCriterionConfig):
    batch_szie: int = field(default=-1, metadata={"help": "number of samples for kernel dispersion"})


@register_criterion("ce_mma_disp", dataclass=CrossEntropyMMACriterionConfig)
class CrossEntropyMMACriterion(CrossEntropyDispersedCriterion):
    def __init__(self,
                 task,
                 sentence_avg,
                 dispersed_weight=0.0,
                 report_dispersed=True,
                 batch_size=-1):
        super().__init__(task, sentence_avg, dispersed_weight,
                         report_dispersed)

        self.mma = MMADispersion(batch_size=batch_size)

    def compute_dispersed_loss(self, model):
        mma_loss = self.mma(model.decoder.embed_tokens.weight)
        return mma_loss

@register_criterion("ce_mhe_disp", dataclass=CrossEntropyMMACriterionConfig)
class CrossEntropyMMACriterion(CrossEntropyDispersedCriterion):
    def __init__(self,
                 task,
                 sentence_avg,
                 dispersed_weight=0.0,
                 report_dispersed=True,
                 batch_size=-1):
        super().__init__(task, sentence_avg, dispersed_weight,
                         report_dispersed)

        self.mhe = MHEDispersion(batch_size=batch_size)

    def compute_dispersed_loss(self, model):
        mhe_loss = self.mma(model.decoder.embed_tokens.weight)
        return mhe_loss

@register_criterion("ce_koleo_disp", dataclass=CrossEntropyMMACriterionConfig)
class CrossEntropyMMACriterion(CrossEntropyDispersedCriterion):
    def __init__(self,
                 task,
                 sentence_avg,
                 dispersed_weight=0.0,
                 report_dispersed=True,
                 batch_size=-1):
        super().__init__(task, sentence_avg, dispersed_weight,
                         report_dispersed)

        self.mhe = KoLeoDispersion()

    def compute_dispersed_loss(self, model):
        koleo_loss = self.koleo(model.decoder.embed_tokens.weight)
        return koleo_loss