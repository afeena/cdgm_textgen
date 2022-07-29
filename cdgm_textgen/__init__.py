# initialize hydra
from fairseq.dataclass.initialize import hydra_init

hydra_init()

import cdgm_textgen.criterions  # noqa
import cdgm_textgen.models  # noqa
import cdgm_textgen.modules  # noqa
import cdgm_textgen.tasks  # noqa
import cdgm_textgen.dataclass # noqa
import cdgm_textgen.logging # noqa

