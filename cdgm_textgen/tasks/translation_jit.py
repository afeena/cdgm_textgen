import torch
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask, TranslationConfig
from omegaconf import DictConfig


@register_task("translation_jit", dataclass=TranslationConfig)
class TranslationTaskJIT(TranslationTask):
    def build_criterion(self, cfg: DictConfig):
        from fairseq import criterions
        criterion = criterions.build_criterion(cfg, self)
        return torch.jit.script_if_tracing(criterion)

