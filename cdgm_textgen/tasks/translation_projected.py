import torch
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationConfig, TranslationTask


@register_task("translation_proj", dataclass=TranslationConfig)
class TranslationTaskSphereProjections(TranslationTask):
    def optimizer_step(self, optimizer, model, update_num):
        optimizer.step()
        # here project the model parameters to the sphere by setting the norms to 1
        with torch.no_grad():
            for np, p in model.named_parameters():
                if np == 'decoder.embed_tokens.weight':
                    p /= p.norm(dim=1, keepdim=True)
