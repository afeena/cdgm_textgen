# initialize hydra
import sys

from cdgm_textgen.dataclass.initialize import hydra_init

hydra_init()

import cdgm_textgen.criterions  # noqa
import cdgm_textgen.models  # noqa
import cdgm_textgen.modules  # noqa
import cdgm_textgen.tasks  # noqa
import cdgm_textgen.dataclass # noqa
import cdgm_textgen.logging # noqa

import fairseq.distributed  # noqa
import fairseq.optim  # noqa
import fairseq.optim.lr_scheduler  # noqa
import fairseq.pdb  # noqa
import fairseq.scoring  # noqa
import fairseq.tasks  # noqa
import fairseq.token_generation_constraints  # noqa

import fairseq.benchmark  # noqa
import fairseq.model_parallel  # noqa

