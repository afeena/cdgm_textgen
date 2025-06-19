from dataclasses import dataclass, field

import ledoh_torch
import torch
import torch.nn.functional as F
from fairseq import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass.configs import FairseqDataclass
from ledoh_torch import minimum_acos_distance_batch, circular_variance, LloydSphereDispersion, \
    SlicedSphereDispersion, KernelSphereDispersion, minimum_acos_distance


@dataclass
class DistanceCriterionConfig(FairseqDataclass):
    distance_metric: str = field(default="cosine",
                                 metadata={"help": "Distance metric to use for the loss"})



@register_criterion("distance_criterion", dataclass=DistanceCriterionConfig)
class DistanceCriterion(FairseqCriterion):
    def __init__(self, task, distance_metric):
        super().__init__(task)
        assert distance_metric in ["cosine", "acos"], 'Distance metric must be either "cosine" or "acos"'
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
            loss = (1.0 - dot_prod).diag().mean()

        elif self.distance_metric == "acos":
            loss = torch.acos(dot_prod.clamp(-1 + 1e-4, 1 - 1e-4)).diag().mean()

        return loss, {"dotprod": dot_prod.detach().mean()}

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        dotprod = sum(log.get("dotprod", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size, sample_size, round=3
        )
        metrics.log_scalar(
            "dotprod", dotprod / sample_size, sample_size, round=3
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True

@dataclass
class DistanceCriterionMaxMarginConfig(DistanceCriterionConfig):
    margin_lambda: float = field(default=1.0,
                                 metadata={"help": "Margin for max margin loss"})
    negative: str = field(default="projection", metadata={"help": "Negative sampling method (projection or diff)"})
@register_criterion("distance_criterion_max_margin", dataclass=DistanceCriterionMaxMarginConfig)
class DistanceCriterionMaxMargin(DistanceCriterion):
    def __init__(self, task, distance_metric, margin_lambda=0.1, negative="projection"):
        super().__init__(task, distance_metric)
        self.margin_lambda = margin_lambda
        self.negative = negative

    def negative_by_projection(self, out, target, dim=-1, eps=1e-4):
        z = (out * target).sum(dim)
        theta = z.clip(-1 + eps, 1 - eps).arccos()

        diff = out - target * torch.cos(theta).unsqueeze(-1)
        return diff / torch.norm(diff, dim=-1, keepdim=True)
    def negative_by_diff(self, out, target):
        return (out-target)/torch.norm(out-target, dim=-1, keepdim=True)

    def get_negative(self, out, target):
        out = out.clone().detach()
        target = target.clone().detach()
        if self.negative == "projection":
            negative = self.negative_by_projection(out, target)
        elif self.negative == "diff":
            negative = self.negative_by_diff(out, target)
        return negative.detach()

    def compute_loss(self, model, net_output, sample, reduce=True):
        #lambda_4 = 1
        #lambda_3=0
        #loss = max{0, self.margin+<u_orth,u_predicted>-<u,u_predicted>}

        target = model.get_targets(sample, net_output)
        mask = sample["target"].ne(self.padding_idx)

        out = F.normalize(net_output[0][mask], dim=-1)
        target = F.normalize(target[mask], dim=-1)

        dot_prod_out_target = out @ target.T

        negative = self.get_negative(out, target)

        dot_prod_out_neg = out @ negative.T

        loss = self.margin_lambda + dot_prod_out_neg.diag() - dot_prod_out_target.diag()
        loss = torch.max(torch.zeros_like(loss), loss).mean()

        return loss, {}

@dataclass
class DispersedDistCriterionConfig(DistanceCriterionConfig):
    dispersed_weight: float = field(
        default=0.0,
        metadata={"help": "weight for dispersed loss"},
    )
    report_dispersed: bool = field(default=True,
                                   metadata={"help": "report dispersed loss"})
    dispersion_type: str = field(default='sliced')


@register_criterion("dist_criterion_with_dispersion", dataclass=DispersedDistCriterionConfig)
class DispersedDistanceCriterion(DistanceCriterion):
    def __init__(self, task, distance_metric, dispersed_weight, report_dispersed, dispersion_type):
        super().__init__(task, distance_metric)

        self.dispersed_weight = dispersed_weight
        self.report_dispersed = report_dispersed

        self.dispersion = self.get_dispersion_by_type(dispersion_type)

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
            "dispersion_loss": logging["dispersion_loss"]
        }
        if self.report_dispersed:
            acos_min_dist, cvar = self.compute_dispersed_metrics(model, net_output, sample)
            logging_output["acos_min_dist"] = acos_min_dist.detach()
            logging_output["circular_variance"] = cvar.detach()

        if type(logging) is dict:
            logging_output = {**logging_output, **logging}
        return loss, 1, logging_output

    def get_masked_target_and_out_normalized(self, model, net_output, sample):
        target = model.get_targets(sample, net_output)
        mask = sample["target"].ne(self.padding_idx)

        out = F.normalize(net_output[0][mask], dim=-1)
        target = F.normalize(target[mask], dim=-1)

        return out, target

    def compute_loss(self, model, net_output, sample, reduce=True):
        out, target = self.get_masked_target_and_out_normalized(model, net_output, sample)
        dot_prod = out @ target.T

        if self.distance_metric == "cosine":
            loss = (1.0 - dot_prod).diag().mean()

        elif self.distance_metric == "acos":
            loss = torch.acos(dot_prod.clamp(-1 + 1e-4, 1 - 1e-4)).diag().mean()

        if self.dispersed_weight > 0:
            disp_loss = self.dispersion(out)
            reg = (self.dispersed_weight * disp_loss).mean()
            loss = loss + reg

        return loss, {"dotprod": dot_prod.detach().mean(), "disp_loss": disp_loss.detach().mean() if self.dispersed_weight > 0 else 0}
    def get_dispersion_by_type(self, dispersion_type):
        if dispersion_type == 'mhe-gaussian-eucl':
            return ledoh_torch.KernelSphereDispersion(kernel="gaussian",
                                                      distance="eucledian",
                                                      kernel_args={"gamma":1.0}
                                                      )
        elif dispersion_type == 'sliced':
            return ledoh_torch.SlicedSphereDispersion()

        else:
            raise ValueError(f"Unknown dispersion type: {dispersion_type}")

    def compute_dispersed_metrics(self,model, net_output, sample):
        with torch.no_grad():
            out, _ = self.get_masked_target_and_out_normalized(model, net_output, sample)
            min_cosine_dist = minimum_acos_distance(out)
            cvar = circular_variance(out)
        return min_cosine_dist.detach(), cvar.detach()

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        dotprod = sum(log.get("dotprod", 0) for log in logging_outputs)
        acos_min_dist = sum(log.get("acos_min_dist", 0) for log in logging_outputs)
        circular_variance = sum(log.get("circular_variance", 0) for log in logging_outputs)
        dispersion_loss_sum = sum(log.get("dispersion_loss", 0) for log in logging_outputs)
        n_logs = len(logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size, sample_size, round=3
        )
        metrics.log_scalar(
            "dotprod", dotprod / sample_size, sample_size, round=3
        )
        metrics.log_scalar("dispersion_loss", dispersion_loss_sum / sample_size, sample_size, round=3)

        metrics.log_scalar("acos_min_dist", acos_min_dist / n_logs, n_logs, round=3)
        metrics.log_scalar("circular_variance", circular_variance / n_logs, n_logs, round=3)
