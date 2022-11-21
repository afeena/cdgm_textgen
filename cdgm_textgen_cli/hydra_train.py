#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import fairseq_cli.hydra_train as fairseq_hydra_train
from cdgm_textgen.dataclass.utils import omegaconf_no_object_check as cdgm_omegaconf_no_object_check
from cdgm_textgen.dataclass.configs import CDGMTextgenConfig
from cdgm_textgen.dataclass.initialize import add_defaults as cdgm_add_defaults
from cdgm_textgen.dataclass.initialize import hydra_init as cdgm_hydra_init
from cdgm_textgen_cli.train import main as cdgm_pre_main

fairseq_hydra_train.omegaconf_no_object_check = cdgm_omegaconf_no_object_check
fairseq_hydra_train.FairseqConfig = CDGMTextgenConfig
fairseq_hydra_train.add_defaults = cdgm_add_defaults
fairseq_hydra_train.hydra_init = cdgm_hydra_init
fairseq_hydra_train.pre_main = cdgm_pre_main

def cli_main():
    fairseq_hydra_train.cli_main()

if __name__ == "__main__":
    fairseq_hydra_train.cli_main()
