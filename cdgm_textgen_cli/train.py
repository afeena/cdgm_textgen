#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""
import fairseq_cli.train as train

# import cdgm_textgen.tasks.utils as cdgm_utils
from cdgm_textgen.dataclass.configs import CDGMTextgenConfig
from cdgm_textgen.dataclass.initialize import add_defaults as cdgm_add_defaults
from cdgm_textgen.dataclass.utils import convert_namespace_to_omegaconf as cdgm_convert_namespace_to_omegaconf

# path config and helper functions
train.FairseqConfig = CDGMTextgenConfig
train.add_defaults = cdgm_add_defaults
train.convert_namespace_to_omegaconf = cdgm_convert_namespace_to_omegaconf


# patch tasks for fairseq
# train.tasks.translation_lev.TranslationLevenshteinTask.build_model = cdgm_utils.build_model_levenshtein
# train.tasks.translation_lev.TranslationLevenshteinTask.build_generator = cdgm_utils.build_generator
# train.tasks.translation_lev.TranslationLevenshteinTask.valid_step = cdgm_utils.valid_step


def main(cfg: CDGMTextgenConfig) -> None:
    train.main(cfg)


if __name__ == "__main__":
    train.cli_main()
