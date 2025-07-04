import math
from dataclasses import dataclass
from dataclasses import field
from typing import Optional

import geoopt.manifolds.stereographic.math as pmath
import power_spherical
import torch
import torch.nn.functional as F
from fairseq import metrics
from fairseq.criterions import register_criterion, FairseqCriterion
from fairseq.criterions.cross_entropy import CrossEntropyCriterionConfig
from fairseq.dataclass import FairseqDataclass
from geoopt import StereographicExact
from torch.distributions import multivariate_normal


from cdgm_textgen.criterions.distance_loss import DistanceCriterion, DistanceCriterionConfig


@register_criterion("manifold_dist_criterion", dataclass=DistanceCriterionConfig)
class ManifoldDistanceCriterion(DistanceCriterion):

    def compute_loss(self, model, net_output, sample, reduce=True):
        target = model.get_targets(sample, net_output)

        mask = sample["target"].ne(self.padding_idx)

        out = net_output[0][mask]
        target = target[mask]

        # sanity check
        if not model.decoder.manifold.check_point_on_manifold(out):
            if isinstance(model.decoder.manifold, StereographicExact):
                out = pmath.project(model.decoder.manifold.expmap0(out), k=model.decoder.manifold.k)
            else:
                raise ValueError(f'out is not on manifold {model.decoder.manifold}')
        if not model.decoder.manifold.check_point_on_manifold(target):
            if isinstance(model.decoder.manifold, StereographicExact):
                target = pmath.project(model.decoder.manifold.expmap0(target), k=model.decoder.manifold.k)
            else:
                raise ValueError(f'target is not on manifold {model.decoder.manifold}')

        loss = model.decoder.manifold.dist(target, out).mean()
        return loss, None


@register_criterion("manifold_dist2_criterion")
class ManifoldDistance2Criterion(DistanceCriterion):

    def compute_loss(self, model, net_output, sample, reduce=True):
        target = model.get_targets(sample, net_output)

        mask = sample["target"].ne(self.padding_idx)

        out = net_output[0][mask]
        target = target[mask]

        # sanity check
        if not model.decoder.manifold.check_point_on_manifold(out):
            if isinstance(model.decoder.manifold, StereographicExact):
                out = pmath.project(model.decoder.manifold.expmap0(out), k=model.decoder.manifold.k)
            else:
                raise ValueError(f'out is not on manifold {model.decoder.manifold}')
        if not model.decoder.manifold.check_point_on_manifold(target):
            if isinstance(model.decoder.manifold, StereographicExact):
                target = pmath.project(model.decoder.manifold.expmap0(target), k=model.decoder.manifold.k)
            else:
                raise ValueError(f'target is not on manifold {model.decoder.manifold}')

        loss = model.decoder.manifold.dist2(target, out).mean()
        return loss, None


@register_criterion("sphere_dist_criterion")
class SphereDistanceCriterion(DistanceCriterion):

    def compute_loss(self, model, net_output, sample, reduce=True):
        target = model.get_targets(sample, net_output)

        mask = sample["target"].ne(self.padding_idx)

        out = net_output[0][mask]
        target = target[mask]

        # sanity check
        if not model.decoder.manifold.check_point_on_manifold(out):
            out = F.normalize(out, dim=-1)
        if not model.decoder.manifold.check_point_on_manifold(target):
            target = F.normalize(target, dim=-1)

        loss = model.decoder.manifold.dist(target, out).mean()
        return loss, None


@register_criterion("sphere_dist2_criterion")
class SphereDistance2Criterion(DistanceCriterion):

    def compute_loss(self, model, net_output, sample, reduce=True):
        target = model.get_targets(sample, net_output)

        mask = sample["target"].ne(self.padding_idx)

        out = net_output[0][mask]
        target = target[mask]

        # sanity check
        if not model.decoder.manifold.check_point_on_manifold(out):
            out = F.normalize(out, dim=-1)
        if not model.decoder.manifold.check_point_on_manifold(target):
            target = F.normalize(target, dim=-1)

        loss = model.decoder.manifold.dist2(target, out).mean()
        return loss, None


@register_criterion("l2_ar_criterion")
class L2ARCriterion(DistanceCriterion):
    def compute_loss(self, model, net_output, sample, reduce=True):
        target = model.get_targets(sample, net_output)
        mask = sample["target"].ne(self.padding_idx)
        out = net_output[0][mask]
        target = target[mask]
        loss = torch.norm(out - target, p=2.0, dim=-1)
        loss = loss.mean(-1)
        return loss, None


@register_criterion("l2_squared_criterion")
class L2SquaredARCriterion(DistanceCriterion):
    def compute_loss(self, model, net_output, sample, reduce=True):
        target = model.get_targets(sample, net_output)
        mask = sample["target"].ne(self.padding_idx)
        out = net_output[0][mask]
        target = target[mask]
        loss = torch.pow(torch.norm(out - target, p=2.0, dim=-1), 2)
        loss = loss.mean(-1)
        return loss, None



