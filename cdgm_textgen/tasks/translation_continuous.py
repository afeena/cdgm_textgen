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
        }
    )
    generator: str = field(
        default="fairseq",
        metadata={
            "help": "generator to use for decoding"
        }
    )
    mbr_nsamples: int = field(
        default=1,
        metadata={
            "help": "number of samples for MBR decoding"
        }
    )
    ranking_metric: str = field(
        default="bertscore",
        metadata={
            "help": "metric to use for ranking. Supported metrics are [google_bleu, meteor]"
        }
    )


@register_task("translation_cont", dataclass=TranslationContinuousConfig)
class TranslationCountinuousTask(TranslationTask):
    def __init__(self, cfg: TranslationContinuousConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)
        self.cfg = cfg
        self.gen = cfg.generator

    def build_model(self, cfg: TranslationContinuousConfig, from_checkpoint=False):
        model = super().build_model(cfg, from_checkpoint)
        if self.cfg.eval_bleu:
            detok_args = json.loads(self.cfg.eval_bleu_detok_args)
            self.tokenizer = encoders.build_tokenizer(
                Namespace(tokenizer=self.cfg.eval_bleu_detok, **detok_args)
            )
            gen_args = json.loads(self.cfg.eval_bleu_args)
            self.sequence_generator = self.build_generator(
                [model], Namespace(**gen_args))

        return model

    def build_generator(self, models, args, seq_gen_cls=None,
                        extra_gen_cls_kwargs=None,
                        prefix_allowed_tokens_fn=None):

        if self.gen == "fairseq":
            return super().build_generator(
                models,
                args,
                seq_gen_cls=seq_gen_cls,
                extra_gen_cls_kwargs=extra_gen_cls_kwargs,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            )

        if self.gen == "continuous":
            from cdgm_textgen.continuous_sequence_generator import ContinuousSequenceGenerator
            seq_gen_cls = ContinuousSequenceGenerator
            extra_gen_cls_kwargs = {}

        if self.gen == "mixture":
            from cdgm_textgen.continuous_sequence_generator import MixtureContinuousSequenceGenerator
            seq_gen_cls = MixtureContinuousSequenceGenerator
            extra_gen_cls_kwargs = {"reduce_strategy": getattr(args, "reduce_strategy", "max")}

        if self.gen == "mbr":
            from cdgm_textgen.continuous_sequence_generator import MBRContinuousSequenceGenerator
            seq_gen_cls = MBRContinuousSequenceGenerator
            extra_gen_cls_kwargs = {"n_samples": self.cfg.get("mbr_nsamples", 1),
                                    "ranker": self.cfg.get("ranking_metric", "bertscore"),
                                    "token_mbr":  self.cfg.get("token_mbr", False),
                                    }

        return seq_gen_cls(
            models,
            self.target_dictionary,
            beam_size=getattr(args, "beam", 1),
            min_len=getattr(args, "min_len", 1),
            search_strategy=getattr(args, "search_strategy", "beam"),
            metric=getattr(args, "search_strategy", "cosine"),
            **extra_gen_cls_kwargs
        )
