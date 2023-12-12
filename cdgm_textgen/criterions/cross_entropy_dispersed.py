import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import label_smoothed_nll_loss, \
    LabelSmoothedCrossEntropyCriterionConfig, LabelSmoothedCrossEntropyCriterion


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
            dispersed_min_dist, dispersed_mean_v = self.compute_dispersed_metrics(model)
            logging_output["dispersed_min_dist"] = dispersed_min_dist.data
            logging_output["dispersed_mean_v"] = dispersed_mean_v.data

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
            dist = torch.acos(model.decoder.embed_tokens.weight.data @ model.decoder.embed_tokens.weight.data.T)
            dist.fill_diagonal_(float('inf'))
            min_dist = dist.view(-1).min()
            mean_v = 1 - torch.norm(torch.mean(model.decoder.embed_tokens.weight.data, dim=0))
            return min_dist.detach(), mean_v.detach()

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        dispersed_min_dist = sum(log.get("dispersed_min_dist", 0) for log in logging_outputs)
        dispersed_mean_v = sum(log.get("dispersed_mean_v", 0) for log in logging_outputs)
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


@register_criterion("label_smoothed_cross_entropy_lloyd", dataclass=LSCrossEntropyLloydCriterionConfig)
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
        samples = F.normalize(torch.randn(self.n_samples,
                                          1,
                                          model.decoder.embed_dim,
                                          device=model.decoder.embed_tokens.weight.device
                                          ),
                              dim=-1)
        # then compute the distance to the nearest point on the sphere
        dist2 = torch.acos(samples @ model.decoder.embed_tokens.weight.T) ** 2
        # choose closest center for each sample and sum results
        ll = (torch.min(dist2, dim=-1)[0]).sum()
        return ll


@register_criterion("label_smoothed_cross_entropy_sliced", dataclass=LSCrossEntropyDispersedCriterionConfig)
class LSCrossEntropySlicedCriterion(LSCrossEntropyDispersedCriterion):
    def compute_dispered_loss(self, model):
        # choose a great circle at random via orthonormal p,q
        PQ = F.normalize(torch.randn(2, model.decoder.embed_dim, dtype=torch.double), dim=-1)
        PQ, _ = torch.linalg.qr(PQ.T)
        p, q = PQ[:, 0], PQ[:, 1]

        # project all embeddings onto the great circle
        thetas = torch.arctan2(model.decoder.embed_tokens.weight @ p, model.decoder.embed_tokens.weight @ q)

        vocab_size = model.decoder.embed_tokens.weight.size(0)
        ix = torch.argsort(thetas)
        invix = torch.empty_like(ix)
        invix[ix] = torch.arange(vocab_size)

        phis = -math.pi - math.pi / vocab_size + (2 * math.pi * torch.arange(1, vocab_size + 1)) / vocab_size
        phis = phis[invix]

        # riemannian projection:
        tctr = 0.5*torch.mean(torch.pow(thetas - phis, 2))

        return tctr
