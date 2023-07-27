# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from collections.abc import Collection
from dataclasses import dataclass, field
from typing import Any, List

import torch.distributed as dist
from fairseq.dataclass import FairseqDataclass
from fairseq.optim import FairseqOptimizer, register_optimizer
from omegaconf import II, OmegaConf

from geoopt import optim

logger = logging.getLogger(__name__)

# https://geoopt.readthedocs.io/en/latest/optimizers.html
#
@dataclass
class FairseqRiemannianAdamConfig(FairseqDataclass):
    stabilize: bool = field(default=1,
                            metadata={"help": "Stabilize parameters if "
                                              "they are off-manifold due to numerical reasons every stabilize steps "})
    riemann_adam_betas: Any = field(
        default=(0.9, 0.999), metadata={"help": "betas for Adam optimizer"}
    )
    riemann_adam_eps: float = field(
        default=1e-8, metadata={"help": "epsilon for Adam optimizer"}
    )
    weight_decay: float = field(default=0.0, metadata={"help": "weight decay"})

    # TODO common vars below in parent
    tpu: bool = II("common.tpu")
    lr: List[float] = II("optimization.lr")


@register_optimizer("riemannian_adam", dataclass=FairseqRiemannianAdamConfig)
class RiemannianAdamOptimizer(FairseqOptimizer):
    """Adam optimizer for fairseq.

    Important note: this optimizer corresponds to the "AdamW" variant of
    Adam in its weight decay behavior. As such, it is most closely
    analogous to torch.optim.AdamW from PyTorch.
    """

    def __init__(self, cfg: FairseqRiemannianAdamConfig, params):
        super().__init__(cfg)

        self._optimizer = optim.RiemannianAdam(params, **self.optimizer_config)

    @property
    def optimizer_config(self):
        """
        Return a kwarg dictionary that will be used to override optimizer
        args stored in checkpoints. This allows us to load a checkpoint and
        resume training using a different set of optimizer args, e.g., with a
        different learning rate.
        """
        return {
            "lr": self.cfg.lr[0]
            if isinstance(self.cfg.lr, Collection)
            else self.cfg.lr,
            "betas": eval(self.cfg.riemann_adam_betas)
            if isinstance(self.cfg.riemann_adam_betas, str)
            else OmegaConf.to_container(self.cfg.riemann_adam_betas),
            "eps": self.cfg.riemann_adam_eps,
            "weight_decay": self.cfg.weight_decay,
            "stabilize": self.cfg.stabilize,
        }

    def average_params(self):
        """Reduce Params is only used during BMUF distributed training."""
        state_dict = self.optimizer.state_dict()
        total_gpus = float(dist.get_world_size())

        for _, value in state_dict["state"].items():
            value["exp_avg"] /= total_gpus
            value["exp_avg_sq"] /= total_gpus
            dist.all_reduce(value["exp_avg"], op=dist.ReduceOp.SUM)
            dist.all_reduce(value["exp_avg_sq"], op=dist.ReduceOp.SUM)


