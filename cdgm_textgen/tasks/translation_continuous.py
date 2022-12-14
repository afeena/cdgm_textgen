import json
from argparse import Namespace
from dataclasses import dataclass, field

from fairseq.tasks import TranslationConfig
from fairseq.tasks import TranslationTask, register_task


@dataclass
class TranslationContinuousConfig(TranslationConfig):
    eval_nn_measure: str = field(
        default="cosine",
        metadata={
            "help": "what measure to use in evaluation for NN search"
        },
    )

@register_task("translation_cont", dataclass=TranslationContinuousConfig)
class TranslationCountinuousTask(TranslationTask):
    def build_model(self, cfg, from_checkpoint=False):
        model = super().build_model(cfg, from_checkpoint)
        if self.cfg.eval_bleu:
            detok_args = json.loads(self.cfg.eval_bleu_detok_args)
            self.tokenizer = encoders.build_tokenizer(
                Namespace(tokenizer=self.cfg.eval_bleu_detok, **detok_args)
            )

            gen_args = json.loads(self.cfg.eval_bleu_args)
            gen_args["decoding_measure"]=self.cfg.eval_nn_measure
            self.sequence_generator = self.build_generator(
                [model], Namespace(**gen_args)
            )
        return model
