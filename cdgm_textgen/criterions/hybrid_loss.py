import math
from dataclasses import dataclass
from dataclasses import field
from typing import Optional

import torch
import torch.nn.functional as F
from fairseq import metrics
from fairseq.criterions import register_criterion, FairseqCriterion
from fairseq.dataclass import FairseqDataclass


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

        if torch.rand(1).item() < self.epsilon:
            out = torch.nn.functional.normalize(out, p=2, dim=-1)
            hV = out @ model.decoder.target_embed.weight.T

            dot_true = 1.0 - F.cosine_similarity(out, target, dim=-1)
            lse = torch.logsumexp(F.normalize(hV, dim=-1), dim=-1)
            loss = lse + dot_true
            loss = loss.mean(-1)
            logging = {"cosine_loss": 0.0, "full_loss": loss.detach(),
                       "full_loss_updates": 1.0,
                       "target_emb_sum": model.decoder.target_embed.weight.detach().sum()}
        else:
            loss = 1.0 - F.cosine_similarity(out, target.detach(), dim=-1)
            loss = loss.mean(-1)
            logging = {"cosine_loss": loss.detach(), "full_loss": 0.0}

        return loss, logging

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        cosine_loss_sum = sum(log.get("cosine_loss", 0) for log in logging_outputs)
        full_loss_sum = sum(log.get("full_loss", 0) for log in logging_outputs)
        full_loss_updates_sum = sum(log.get("full_loss_updates", 0) for log in logging_outputs)
        target_emb_sum = sum(log.get("target_emb_sum", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        cosine_loss_updates_sum = sample_size - full_loss_updates_sum
        target_emb_sum = target_emb_sum / full_loss_updates_sum if full_loss_sum !=0 else 0.0
        full_loss_sum = full_loss_sum / full_loss_updates_sum if full_loss_sum !=0 else 0.0
        cosine_loss_sum = cosine_loss_sum / cosine_loss_updates_sum if cosine_loss_sum !=0 else 0.0

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "cosine_loss", cosine_loss_sum / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "full_loss",  full_loss_sum / math.log(2), sample_size, round=3
        )
        metrics.log_scalar_sum(
            "full_loss_updates", full_loss_updates_sum, 1, round=3
        )
        metrics.log_scalar(
            "target_emb_sum",target_emb_sum, 1, round=6
        )