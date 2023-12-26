import math
from dataclasses import dataclass, field

import ledoh_torch.utils
import torch
from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import label_smoothed_nll_loss, \
    LabelSmoothedCrossEntropyCriterionConfig, LabelSmoothedCrossEntropyCriterion
from ledoh_torch import lloyd_dispersion, cosine_kernel_dispersion
from ledoh_torch.utils import minimum_cosine_distance, circular_variance

from cdgm_textgen.modules.sliced_sphere import compute_sliced_sphere_dispersion


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

        # here: since we don't want to scale regularizer with sample size,
        # we apply mean reduction directly and return sample_size=1
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
            logging_output["cosine_min_dist"] = cosine_min_dist.detach()
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
        if self.dispersed_weight > 0.0:
            dispersed_loss = self.compute_dispered_loss(model)
            loss = loss + self.dispersed_weight * dispersed_loss
        return loss, nll_loss

    def compute_dispered_loss(self, model):
        raise NotImplementedError

    def compute_dispersed_metrics(self, model):
        with torch.no_grad():
            min_cosine_dist = minimum_cosine_distance(model.decoder.embed_tokens.weight.data)
            var = circular_variance(model.decoder.embed_tokens.weight.data)
        return min_cosine_dist, var

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
        metrics.log_scalar(
            "nll_loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar("min_dist", dispersed_min_dist / n_logs, sample_size, round=4)
        metrics.log_scalar("mean dist val", dispersed_mean_v / n_logs, sample_size, round=4)


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
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy, dispersed_weight,
                         report_dispersed)
        self.n_samples = n_lloyd_samples

    def compute_dispered_loss(self, model):
        # first sample n uniform points on the sphere

        return lloyd_dispersion(model.decoder.embed_tokens.weight, self.n_samples)


@register_criterion("ls_cross_entropy_sliced_disp", dataclass=LSCrossEntropyDispersedCriterionConfig)
class LSCrossEntropySlicedCriterion(LSCrossEntropyDispersedCriterion):
    def compute_dispered_loss(self, model):
        p, q = ledoh_torch.utils.init_great_circle(
            model.decoder.embed_dim,
            dtype=model.decoder.embed_tokens.weight.dtype,
            device=model.decoder.embed_tokens.weight.device)

        return compute_sliced_sphere_dispersion(model.decoder.embed_tokens.weight, p, q)

@dataclass
class LSCrossEntropyKernelCriterionConfig(LSCrossEntropyDispersedCriterionConfig):
    kernel: str = field(
        default="cosine",
        metadata={"help": "kernel for dispersion"},
    )
    cosine_gamma: float = field(default = 0.001, metadata={"help": "gamma for cosine kernel"})

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
                 kernel="cosine",
                 cosine_gamma=0.001):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy, dispersed_weight,
                         report_dispersed)
        self.kernel = kernel
        self.cosine_gamma = cosine_gamma
    def compute_dispered_loss(self, model):
        return cosine_kernel_dispersion(model.decoder.embed_tokens.weight, self.cosine_gamma)