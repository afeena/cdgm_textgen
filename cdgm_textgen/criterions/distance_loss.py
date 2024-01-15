import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from fairseq import metrics
from fairseq.criterions import FairseqCriterion, register_criterion


@dataclass
class DistanceCriterionConfig():
    distance_metric: str = field(default="cosine", metadata={"help": "Distance metric to use for the loss"})


@register_criterion("distance_criterion", dataclass=DistanceCriterionConfig)
class DistanceCriterion(FairseqCriterion):
    def __init__(self, task, distance_metric):
        super().__init__(task)
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
            loss = 1.0 - dot_prod
            return loss.mean(-1), {}

        elif self.distance_metric == "acos":
            loss = torch.acos(dot_prod).mean()
            return loss, {"cosine": loss.detach()}

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
