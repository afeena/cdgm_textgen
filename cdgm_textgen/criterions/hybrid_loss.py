import math
from dataclasses import dataclass
from dataclasses import field
from typing import Optional

import power_spherical
import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import register_criterion, FairseqCriterion
from fairseq.criterions.cross_entropy import CrossEntropyCriterion, CrossEntropyCriterionConfig
from fairseq.dataclass import FairseqDataclass
from torch.distributions import multivariate_normal

@dataclass
class HybridLossConfig(FairseqDataclass):
    epsilon: float = field(default=0.1,
                           metadata={"help": "probability of taking discrete step"})


@register_criterion("hybrid_discrete_cont_loss", dataclass=HybridLossConfig)
class HybridLoss(FairseqCriterion):
    def __init__(self, task, epsilon):
        super().__init__(task)
        self.epsilon = epsilon

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

        out = F.normalize(out, dim=-1)
        target = F.normalize(target, dim=-1)

        if torch.rand(1).item() < self.epsilon:
            dot_true = (out * target).sum(dim=-1)
            lse = torch.logsumexp(out @ model.decoder.target_embed_normalized.T, dim=-1)
            loss = lse - dot_true
        else:
            dot_true = (out * target.detach()).sum(dim=-1)
            # dot_true = (h * emb(y)).sum(dim=-1)
            loss = -dot_true
        return loss.mean(), None

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
