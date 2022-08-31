# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import _MISSING_TYPE, dataclass, field
from typing import Any, Optional
from omegaconf import II, MISSING

from fairseq.dataclass.configs import (
    FairseqDataclass,
    CommonEvalConfig,
    DistributedTrainingConfig,
    DatasetConfig,
    OptimizationConfig,
    CheckpointConfig,
    FairseqBMUFConfig,
    EvalLMConfig,
    InteractiveConfig,
    EMAConfig,
    CommonConfig,
    GenerationConfig
)

from cdgm_textgen.dataclass.constants import GENERATOR_TYPES


@dataclass
class CDGMGenerationConfig(GenerationConfig):
    gen_type: GENERATOR_TYPES = field(
        default="seq2seq",
        metadata={"help": "define type of the generator"},
    )

    # arguments for iterative refinement generator
    decoding_measure: str = field(
        default="cosine",
        metadata={"help": "measure to use for embedding search"},
    )

    decoding_masking: str = field(
        default="distance",
        metadata={"help": "criteria to use for unmasking"},
    )
    approximate_knn: bool = field(
        default=False,
        metadata={"help": "approximate knn with faiss"},
    )


@dataclass
class CDGMCommonConfig(CommonConfig):
    wandb_log_grad: Optional[bool] = field(
        default=False,
        metadata={"help": "log gradients to Weights and Biases"},
    )


@dataclass
class CDGMTextgenConfig(FairseqDataclass):
    common: CommonConfig = CDGMCommonConfig()
    common_eval: CommonEvalConfig = CommonEvalConfig()
    distributed_training: DistributedTrainingConfig = DistributedTrainingConfig()
    dataset: DatasetConfig = DatasetConfig()
    optimization: OptimizationConfig = OptimizationConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    bmuf: FairseqBMUFConfig = FairseqBMUFConfig()
    generation: GenerationConfig = CDGMGenerationConfig()
    eval_lm: EvalLMConfig = EvalLMConfig()
    interactive: InteractiveConfig = InteractiveConfig()
    model: Any = MISSING
    task: Any = None
    criterion: Any = None
    optimizer: Any = None
    lr_scheduler: Any = None
    scoring: Any = None
    bpe: Any = None
    tokenizer: Any = None
    ema: EMAConfig = EMAConfig()
