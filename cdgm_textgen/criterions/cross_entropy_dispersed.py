from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from fairseq.criterions import register_criterion, LabelSmoothedCrossEntropyCriterion
from fairseq.criterions.label_smoothed_cross_entropy import label_smoothed_nll_loss, \
    LabelSmoothedCrossEntropyCriterionConfig
from fairseq import metrics, utils
import math

@dataclass
class LSCrossEntropyDispersedCriterionConfig(LabelSmoothedCrossEntropyCriterionConfig):
    dispersed_weight: float = field(
        default=0.0,
        metadata={"help": "weight for dispersed loss"},
    )
    report_dispersed: bool = field(default=True, metadata={"help": "report dispersed loss"})


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
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
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
            dispersed_min_dist, dispersed_mean_v = self.compute_dispered_loss(model)
            logging_output["dispersed_min_dist"] = dispersed_min_dist.detach()
            logging_output["dispersed_mean_v"] = dispersed_mean_v.detach()

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
            dist = model.decoder.manifold.dist(model.decoder.weight.data, model.decoder.weight.data)
            dist.fill_diagonal_(float('inf'))
            min_dist = dist.min(dim=-1)[0]
            mean_v = 1-torch.norm(torch.mean(model.decoder.weight.data, dim=0))
            return min_dist.detach(), mean_v.detach()

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        dispersed_min_dist = sum(log.get("dispersed_min_dist", 0) for log in logging_outputs)
        dispersed_mean_v = sum(log.get("dispersed_mean_v", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(dispersed_min_dist, 1, round=4)
        metrics.log_scalar(dispersed_mean_v, 1, round=4)


class LSCrossEntropyLloysCriterionConfig(LSCrossEntropyDispersedCriterionConfig):
    n_samples: int = field(
        default=100,
        metadata={"help": "weight for dispersed loss"},
    )

@register_criterion("label_smoothed_cross_entropy_lloyd", dataclass=LSCrossEntropyLloysCriterionConfig)
class LSCrossEntropyLloydCriterion(LabelSmoothedCrossEntropyCriterion):
    def __init__(self,
                 task,
                 sentence_avg,
                 label_smoothing,
                 ignore_prefix_size=0,
                 report_accuracy=False,
                 dispersed_weight=0.0,
                 report_dispersed=True,
                 n_samples=100):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy, dispersed_weight, report_dispersed)
        self.n_samples = n_samples

    def compute_dispered_loss(self, model):
        #first sample n uniform points on the sphere
        manifold = model.decoder.manifold
        samples = F.normalize(torch.randn(self.n_samples, 1, model.decoder.embed_dim), dim=-1)
        # then compute the distance to the nearest point on the sphere
        dist2 = manifold.dist2(samples, model.decoder.embed_tokens.weight[None,...], dim=-1)
        #choose closest center for each sample and sum results
        ll = (torch.min(dist2, dim=-1)[0]).sum()
        return ll

