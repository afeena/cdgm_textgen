# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field
from typing import Optional

import torch.nn.functional as F
import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from torch import Tensor
import power_spherical


def _mean_ds(x: Tensor, dim=None) -> Tensor:
    return (
        x.float().mean().type_as(x)
        if dim is None
        else x.float().mean(dim).type_as(x)
    )


@dataclass
class ContinuousCriterionConfig(FairseqDataclass):
    kappa_lambda: float = field(
        default=0.0,
        metadata={"help": "weight for kappa reg term"},
    )


@register_criterion("continuous_loss", dataclass=ContinuousCriterionConfig)
class ContinuousCriterion(FairseqCriterion):
    def __init__(self, task, kappa_lambda):
        super().__init__(task)
        self.kappa_lambda = kappa_lambda

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        # B x T
        src_tokens, src_lengths = (
            sample["net_input"]["src_tokens"],
            sample["net_input"]["src_lengths"],
        )
        tgt_tokens, prev_output_tokens = sample["target"], sample["prev_target"]

        outputs = model(src_tokens, src_lengths, prev_output_tokens, tgt_tokens)

        token_loss, token_loss_logging = self.compute_loss(model, outputs, sample)
        loss = token_loss.clone()
        sample_size = 1

        # add length loss
        length_loss = None
        if "length" in outputs:
            length_factor = outputs["length"]["factor"]
            length_loss = self.compute_length_loss(outputs["length"]) * length_factor
            loss += length_loss

        logging_output = {
            "loss": loss.data,
            "nll_loss": token_loss,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "length": length_loss.data / length_factor if length_loss is not None else 0
        }

        return loss, sample_size, {**logging_output, **token_loss_logging}

    def compute_loss(self, model, net_output, sample):
        raise NotImplementedError

    def compute_length_loss(self, outputs):
        logits = F.log_softmax(outputs["out"], dim=-1)
        length_loss = F.nll_loss(logits, outputs["tgt"].to(logits.device), reduction="none")
        return _mean_ds(length_loss)

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""

        loss = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        nll_loss = utils.item(sum(log.get("nll_loss", 0) for log in logging_outputs))
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )
        length_loss = utils.item(sum(log.get("length", 0) for log in logging_outputs))

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss / sample_size, sample_size, round=3
        )
        metrics.log_scalar('token_loss', nll_loss / sample_size, sample_size, round=3
                           )

        metrics.log_scalar("length_loss", length_loss / sample_size / math.log(2), round=3)
        if logging_outputs[0].get("kappa", None) is not None:
            kappa = utils.item(sum(log.get("kappa", 0) for log in logging_outputs))
            metrics.log_scalar('kappa', kappa, 1, round=3
                               )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True

    def _parse_model_outputs(self, model, net_output, sample, normalize=True, return_norm=False):
        mask = net_output["word_ins"]["mask"]
        outs = net_output["word_ins"]["out"]
        kappa = None
        if net_output["word_ins"]["dist_params"] is not None:
            kappa = net_output["word_ins"]["dist_params"]["kappa"]
            kappa = kappa[mask]

        target = model.get_targets(sample, net_output)
        outs, target = outs[mask], target[mask]

        outs_norm, target_norm = None, None
        if return_norm:
            outs_norm = torch.norm(outs, dim=-1) + 0.0001
            target_norm = torch.norm(target, dim=-1) + 0.0001

        if normalize:
            outs = F.normalize(outs, p=2.0, dim=-1)
            target = F.normalize(target, p=2.0, dim=-1)

        return {
            "mask": mask,
            "outs": outs,
            "target": target,
            "kappa": kappa,
            "outs_norm": outs_norm,
            "target_norm": target_norm
        }


@register_criterion("cosine_distance_loss", dataclass=ContinuousCriterionConfig)
class CosineDistanceCriterion(ContinuousCriterion):
    def compute_loss(self, model, net_output, sample):
        parsed_outputs = self._parse_model_outputs(model, net_output, sample)

        loss = 1.0 - F.cosine_similarity(parsed_outputs["outs"],
                                         parsed_outputs["target"],
                                         dim=-1
                                         )
        loss = _mean_ds(loss)
        return loss, {}


@register_criterion("vmf_loss", dataclass=ContinuousCriterionConfig)
class vMFCriterion(ContinuousCriterion):
    def compute_loss(self, model, net_output, sample):
        parsed_outputs = self._parse_model_outputs(model, net_output, sample)

        from cdgm_textgen.modules import neg_log_cmk_approx
        m = torch.tensor([parsed_outputs["outs"].shape[-1]], device=parsed_outputs["kappa"].device)

        cmk_approx = neg_log_cmk_approx(m, parsed_outputs["kappa"])
        second_term = parsed_outputs["kappa"] * (parsed_outputs["outs"] * parsed_outputs["target"]).sum(-1)
        vmf = cmk_approx - second_term

        kappa_reg_term = self.kappa_lambda * (torch.pow(parsed_outputs["kappa"], 2.0)).mean()
        loss = _mean_ds(vmf) + kappa_reg_term

        logging = {"kappa": _mean_ds(parsed_outputs["kappa"])}
        return loss, logging


@register_criterion("power_spherical_loss", dataclass=ContinuousCriterionConfig)
class PowerSphericalCriterion(ContinuousCriterion):
    def compute_loss(self, model, net_output, sample):
        parsed_outputs = self._parse_model_outputs(model, net_output, sample)

        d_x = power_spherical.PowerSpherical(parsed_outputs["outs"], parsed_outputs["kappa"])
        # here the assumption is that kappa=norm() already, so do not apply norm on kappa while do reg
        loss = -d_x.log_prob(parsed_outputs["target"])
        reg_term = self.kappa_lambda * (torch.pow(parsed_outputs["kappa"], 2.0)).mean()
        loss = _mean_ds(loss) + reg_term

        logging = {"kappa": _mean_ds(parsed_outputs["kappa"])}

        return loss, logging


@register_criterion("mrt_loss", dataclass=ContinuousCriterionConfig)
class MRTCriterion(ContinuousCriterion):
    def compute_loss(self, model, net_output, sample):
        parsed_outputs = self._parse_model_outputs(model, net_output, sample)

        tau = net_output["word_ins"]["dist_params"]["tau"][parsed_outputs["mask"]]
        loss = 1.0 - F.sigmoid(tau) * (parsed_outputs["outs"] * parsed_outputs["target"]).sum(-1)
        loss = _mean_ds(loss) + self.kappa_lambda * (torch.pow(parsed_outputs["kappa"], 2.0)).mean()
        logging = {"kappa": _mean_ds(parsed_outputs["kappa"])}
        return loss, logging


@dataclass
class PowerSphericalRevKLCriterionConfig(ContinuousCriterionConfig):
    n_samples: int = field(
        default=1,
        metadata={"help": "number of samples"},
    )
    kldirection: Optional[str] = field(
        default="reverse",
        metadata={"help": "forward or reverse KL"},
    )
    target_kappa_scale: Optional[float] = field(
        default=100.0,
        metadata={"help": "scaler to use for target emb norm"},
    )


@register_criterion("power_spherical_klmc_loss", dataclass=PowerSphericalRevKLCriterionConfig)
class PowerSphericalRevKLCriterion(ContinuousCriterion):
    def __init__(self, task, kappa_lambda, n_samples, kldirection, target_kappa_scale):
        super().__init__(task, kappa_lambda)
        self.n_samples = n_samples
        self.direction = kldirection
        self.target_kappa_scale = target_kappa_scale

    def set_dist(self, outputs):
        target_scale = self.target_kappa_scale * outputs["target_norm"]
        p = power_spherical.PowerSpherical(outputs["target"],
                                           target_scale)
        q = power_spherical.PowerSpherical(outputs["outs"],
                                           outputs["kappa"])

        if self.direction == "reverse":
            return p, q
        else:
            return q, p

    def compute_loss(self, model, net_output, sample):
        parsed_outputs = self._parse_model_outputs(model, net_output, sample, return_norm=True)
        p, q = self.set_dist(parsed_outputs)
        z = q.rsample((self.n_samples,))
        log_pz = p.log_prob(z)

        entropy = q.entropy()
        neg_ce = _mean_ds(log_pz, dim=0)
        loss = -entropy-neg_ce
        loss = _mean_ds(loss)
        logging = {"kappa": _mean_ds(parsed_outputs["kappa"])}

        return loss, logging


@register_criterion("power_spherical_kl_taylorapprox_loss", dataclass=PowerSphericalRevKLCriterionConfig)
class PowerSphericalRevKLTaylorApproxCriterion(PowerSphericalRevKLCriterion):
    def taylor_approx(self, p: power_spherical.PowerSpherical, q: power_spherical.PowerSpherical):
        """Approximate E_q [log(1+<x, mu_p>]"""

        # cannot simply use q.mean due to bug in library
        q_mean = q.loc * q.base_dist.marginal_t.mean.unsqueeze(dim=-1)

        # first-order term
        z_mean = (q_mean * p.loc).sum(-1)
        taylor_first = torch.log1p(z_mean)

        # (second-order term cancels out on expectation)

        # third-order term
        alpha_q = q.base_dist.marginal_t.base_dist.concentration1
        beta_q = q.base_dist.marginal_t.base_dist.concentration0
        t_var = q.base_dist.marginal_t.variance
        ratio = (alpha_q + beta_q) / (2 * beta_q)

        dp = torch.pow((q.loc * p.loc).sum(-1), 2)

        _ps_variance_dot = t_var * ((1 - ratio) * dp + ratio)
        taylor_second = _ps_variance_dot / (2 * torch.pow(1 + z_mean, 2))
        # ce is a -log not log
        taylor = - taylor_first + taylor_second

        return taylor

    def kl(self, p: power_spherical.PowerSpherical, q: power_spherical.PowerSpherical):
        """Approximate KL(q : p)"""
        entropy = q.entropy()
        expectation_approx = self.taylor_approx(p, q)
        ce = p.scale * expectation_approx - p.log_normalizer()
        return -entropy + ce

    def compute_loss(self, model, net_output, sample):
        parsed_outputs = self._parse_model_outputs(model, net_output, sample, return_norm=True)
        p, q = self.set_dist(parsed_outputs)
        losses = self.kl(p, q)
        loss = _mean_ds(losses, dim=0)
        logging = {"kappa": _mean_ds(parsed_outputs["kappa"])}
        return loss, logging