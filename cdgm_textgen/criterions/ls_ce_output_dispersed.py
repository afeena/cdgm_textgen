import math
import torch
from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion, LabelSmoothedCrossEntropyCriterionConfig
from ledoh_torch import (
    KernelSphereDispersion,
)

from cdgm_textgen.criterions import LSCrossEntropyDispersedCriterionConfig

@register_criterion("ls_ce_output_dispersion", dataclass=LSCrossEntropyDispersedCriterionConfig)
class LSCrossEntropyDispersedCriterion(LabelSmoothedCrossEntropyCriterion):
    def __init__(self,
                 task,
                 sentence_avg,
                 label_smoothing,
                 ignore_prefix_size=0,
                 report_accuracy=False,
                 dispersed_weight=0.0):
        super().__init__(task, sentence_avg,label_smoothing,
                 ignore_prefix_size=ignore_prefix_size,
                 report_accuracy=report_accuracy)
        self.dispersed_weight = dispersed_weight
        self.mmd = KernelSphereDispersion(gamma=0.1)

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
            assert "features" in net_output[-1]
            dispersed_loss = self.compute_dispersed_loss(net_output[-1]["features"])
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

        return loss, sample_size, logging_output

    def compute_dispersed_loss(self, features: torch.Tensor) -> torch.Tensor:
        features = features / features.norm(dim=-1, keepdim=True)
        return self.mmd(features)

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