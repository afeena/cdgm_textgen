from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from fairseq.criterions import register_criterion, LabelSmoothedCrossEntropyCriterion
from fairseq.criterions.label_smoothed_cross_entropy import label_smoothed_nll_loss, \
    LabelSmoothedCrossEntropyCriterionConfig


@dataclass
class LSCrossEntropyDispersedCriterionConfig(LabelSmoothedCrossEntropyCriterionConfig):
    dispersed_weight: float = field(
        default=0.0,
        metadata={"help": "weight for dispersed loss"},
    )


@register_criterion("label_smoothed_cross_entropy_dispersed", dataclass=LSCrossEntropyDispersedCriterionConfig)
class LSCrossEntropyDispersedCriterion(LabelSmoothedCrossEntropyCriterion):
    def __init__(self,
                 task,
                 sentence_avg,
                 label_smoothing,
                 ignore_prefix_size=0,
                 report_accuracy=False,
                 dispersed_weight=0.0):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)
        self.dispersed_weight = dispersed_weight

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
            dispersed_loss = self.compute_dispered_loss(model, net_output, sample)
            loss = loss + self.dispersed_weight * dispersed_loss
        return loss, nll_loss

    def compute_dispered_loss(self, model, net_output, sample, reduce=True):
        raise NotImplementedError

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
                 n_samples=100):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy, dispersed_weight)
        self.n_samples = n_samples

    def compute_dispered_loss(self, model, net_output, sample, reduce=True):
        #first sample n uniform points on the sphere
        manifold = model.decoder.manifold
        samples = F.normalize(torch.randn(self.n_samples, 1, model.decoder.embed_dim), dim=-1)
        # then compute the distance to the nearest point on the sphere
        dist2 = manifold.dist2(samples, model.decoder.embed_tokens.weight[None,...], dim=-1)
        #choose closest center for each sample and sum results
        ll = (torch.min(dist2, dim=-1)[0]).sum()
        return ll

