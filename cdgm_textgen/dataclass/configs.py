# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass, field
from typing import Any, Optional

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
from omegaconf import MISSING

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
    reduce_strategy: Optional[str] = field(
        default=None,
        metadata={"help": "measure to use for embedding search"},
    )
    search_strategy: str = field(
        default="beam",
        metadata={"help": "measure to use for embedding search"},
    )
    mixture_temperature: float = field(
        default=1.0,
        metadata={"help": "temperature for mixture prior"},
    )
    decoding_masking: str = field(
        default="distance",
        metadata={"help": "criteria to use for unmasking"},
    )
    knn_index_file: Optional[str] = field(default=None, metadata={"help": "knn-mt index file on train data"})
    knn_k: Optional[int] = field(default=5, metadata={"help": "number of nearest neighbors ofr knn-MT decoding"})
    knn_lambda: Optional[float] = field(default=0.5, metadata={"help": "lambda for knn-MT decoding"})
    knn_temperature: Optional[float] = field(default=1.0, metadata={"help": "temperature for knn-MT decoding"})
    knn_rare_only: Optional[int] = field(default=-1, metadata={"help": "use rare words only for knn-MT decoding starting with index"})

@dataclass
class CDGMCommonConfig(CommonConfig):
    wandb_log_grad: Optional[bool] = field(
        default=False,
        metadata={"help": "log gradients to Weights and Biases"},
    )


@dataclass
class CDGMTextgenConfig(FairseqDataclass):
    common: CDGMCommonConfig = CDGMCommonConfig()
    common_eval: CommonEvalConfig = CommonEvalConfig()
    distributed_training: DistributedTrainingConfig = DistributedTrainingConfig()
    dataset: DatasetConfig = DatasetConfig()
    optimization: OptimizationConfig = OptimizationConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    bmuf: FairseqBMUFConfig = FairseqBMUFConfig()
    generation: CDGMGenerationConfig = CDGMGenerationConfig()
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
