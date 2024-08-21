import math
from dataclasses import dataclass, field

import ledoh_torch.utils
import torch
from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import label_smoothed_nll_loss, \
    LabelSmoothedCrossEntropyCriterionConfig, LabelSmoothedCrossEntropyCriterion
from ledoh_torch import LloydSphereDispersion, \
    KernelSphereDispersion, \
    SlicedSphereDispersion, \
    minimum_acos_distance_block, \
    circular_variance, \
    AxisAlignedSlicedSphereDispersion, AxisAlignedBatchSphereDispersion


@dataclass
class LSCrossEntropyDispersedCriterionConfig(LabelSmoothedCrossEntropyCriterionConfig):
    dispersed_weight: float = field(
        default=0.0,
        metadata={"help": "weight for dispersed loss"},
    )
    report_dispersed: bool = field(default=True,
                                   metadata={"help": "report dispersed loss"})


@register_criterion("label_smoothed_cross_entropy_dispersed", dataclass=LSCrossEntropyDispersedCriterionConfig)
class LSCrossEntropyDispersedCriterion(LabelSmoothedCrossEntropyCriterion):
    def __init__(self,
                 task,
                 sentence_avg,
                 label_smoothing,
                 ignore_prefix_size=0,
                 report_accuracy=False,
                 dispersed_weight=0.0,
                 report_dispersed=True):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)
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
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)

        if self.report_dispersed:
            cosine_min_dist, var = self.compute_dispersed_metrics(model)
            logging_output["min_acos_dist"] = cosine_min_dist.detach()
            logging_output["circular_variance"] = var.detach()

        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )

        return loss, nll_loss

    def compute_dispersed_loss(self, model):
        raise NotImplementedError

    @staticmethod
    def compute_dispersed_metrics(model):
        with torch.no_grad():
            min_acos_dist = minimum_acos_distance_block(model.decoder.embed_tokens.weight.detach(), 4096)
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
class LSCrossEntropyLloydCriterionConfig(LSCrossEntropyDispersedCriterionConfig):
    n_lloyd_samples: int = field(
        default=100,
        metadata={"help": "weight for dispersed loss"},
    )


@register_criterion("ls_cross_entropy_lloyd_disp", dataclass=LSCrossEntropyLloydCriterionConfig)
class LSCrossEntropyLloydCriterion(LSCrossEntropyDispersedCriterion):
    def __init__(self,
                 task,
                 sentence_avg,
                 label_smoothing,
                 ignore_prefix_size=0,
                 report_accuracy=False,
                 dispersed_weight=0.0,
                 report_dispersed=True,
                 n_lloyd_samples=100):
        super().__init__(task, sentence_avg,
                         label_smoothing,
                         ignore_prefix_size,
                         report_accuracy,
                         dispersed_weight,
                         report_dispersed)
        self.lloyd = LloydSphereDispersion(n_samples=n_lloyd_samples)

    def compute_dispersed_loss(self, model):
        # first sample n uniform points on the sphere
        lloyd_disp = self.lloyd(model.decoder.embed_tokens.weight)
        return lloyd_disp


@register_criterion("ls_cross_entropy_sliced_disp", dataclass=LSCrossEntropyDispersedCriterionConfig)
class LSCrossEntropySlicedCriterion(LSCrossEntropyDispersedCriterion):
    def __init__(self,
                 task,
                 sentence_avg,
                 label_smoothing,
                 ignore_prefix_size=0,
                 report_accuracy=False,
                 dispersed_weight=0.0,
                 report_dispersed=True):
        super().__init__(task, sentence_avg,
                         label_smoothing,
                         ignore_prefix_size,
                         report_accuracy,
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

@register_criterion("ls_cross_entropy_axisalignedsliced_disp", dataclass=LSCrossEntropyDispersedCriterionConfig)
class LSCrossEntropySlicedCriterion(LSCrossEntropyDispersedCriterion):
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
                         label_smoothing,
                         ignore_prefix_size,
                         report_accuracy,
                         dispersed_weight,
                         report_dispersed)
        self.sliced = AxisAlignedBatchSphereDispersion(batch_size=batch_size)

    def compute_dispersed_loss(self, model):
        p, q = ledoh_torch.utils.init_great_circle(
            model.decoder.embed_dim,
            dtype=model.decoder.output_projection.weight.dtype,
            device=model.decoder.output_projection.weight.device)

        sliced = self.sliced(model.decoder.output_projection.weight)
        return sliced

@dataclass
class LSCrossEntropyKernelCriterionConfig(LSCrossEntropyDispersedCriterionConfig):
    cosine_gamma: float = field(default=0.001, metadata={"help": "gamma for cosine kernel"})
    kernel_sample_size: int = field(default=-1, metadata={"help": "number of samples for kernel dispersion"})


@register_criterion("ls_cross_entropy_kernel_disp", dataclass=LSCrossEntropyKernelCriterionConfig)
class LSCrossEntropyKernelCriterion(LSCrossEntropyDispersedCriterion):
    def __init__(self,
                 task,
                 sentence_avg,
                 label_smoothing,
                 ignore_prefix_size=0,
                 report_accuracy=False,
                 dispersed_weight=0.0,
                 report_dispersed=True,
                 cosine_gamma=0.001,
                 kernel_sample_size=-1):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy, dispersed_weight,
                         report_dispersed)

        self.kernel_disp = KernelSphereDispersion(gamma=cosine_gamma, batch_size=kernel_sample_size)

    def compute_dispersed_loss(self, model):
        kernel_disp = self.kernel_disp(model.decoder.embed_tokens.weight)
        return kernel_disp
