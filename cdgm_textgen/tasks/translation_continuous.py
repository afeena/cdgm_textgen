import json
from argparse import Namespace
from dataclasses import dataclass, field

from fairseq.data import encoders
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationConfig
from fairseq.tasks.translation import TranslationTask

EVAL_BLEU_ORDER = 4


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
    def build_model(self, cfg: TranslationContinuousConfig, from_checkpoint=False):
        model = super().build_model(cfg, from_checkpoint)
        if self.cfg.eval_bleu:
            detok_args = json.loads(self.cfg.eval_bleu_detok_args)
            self.tokenizer = encoders.build_tokenizer(
                Namespace(tokenizer=self.cfg.eval_bleu_detok, **detok_args)
            )

            gen_args = json.loads(self.cfg.eval_bleu_args)
            gen_args["decoding_measure"] = self.cfg.eval_nn_measure
            self.sequence_generator = self.build_generator(
                [model], Namespace(**gen_args)
            )
        return model

    def build_generator(self, models, args, seq_gen_cls=None,
                        extra_gen_cls_kwargs=None,
                        prefix_allowed_tokens_fn=None):

        from cdgm_textgen.continuous_sequence_generator import ContinuousSequenceGenerator, MixtureContinuousSequenceGenerator
        if hasattr(models[0].decoder, "num_mixture_components"):
            seq_gen_cls = MixtureContinuousSequenceGenerator
            extra_gen_cls_kwargs = {"reduce_strategy": getattr(args, "reduce_strategy", "max")}
        else:
            seq_gen_cls = ContinuousSequenceGenerator
            extra_gen_cls_kwargs = {}

        return seq_gen_cls(
            models,
            self.target_dictionary,
            beam_size=getattr(args, "beam", 1),
            min_len=getattr(args, "min_len", 1),
            search_strategy=getattr(args, "search_strategy", "beam"),
            metric=getattr(args, "decoding_measure", "cosine"),
            **extra_gen_cls_kwargs
        )
