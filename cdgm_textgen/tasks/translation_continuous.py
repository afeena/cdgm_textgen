import json
from argparse import Namespace
from dataclasses import dataclass, field

import torch
from fairseq.data import encoders
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationConfig
from fairseq.tasks.translation import TranslationTask
from fairseq import search

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
    def build_model(self, cfg, from_checkpoint=False):
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
            from cdgm_textgen.continuous_sequence_generator import ContinuousSequenceGenerator
            search_strategy = search.BeamSearch(self.target_dictionary)


            extra_gen_cls_kwargs = extra_gen_cls_kwargs or {}

            return ContinuousSequenceGenerator(
                models,
                self.target_dictionary,
                beam_size=getattr(args, "beam", 1),
                min_len=getattr(args, "min_len", 1),
                normalize_scores=(not getattr(args, "unnormalized", False)),
                search_strategy=search_strategy,
                metric=getattr(args, "decoding_measure", "cosine"),
                **extra_gen_cls_kwargs,
            )

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = criterion(model, sample)

            if self.cfg.eval_bleu:
                bleu = self._inference_with_bleu(self.sequence_generator, sample, model)
                logging_output["_bleu_sys_len"] = bleu.sys_len
                logging_output["_bleu_ref_len"] = bleu.ref_len
                # we split counts into separate entries so that they can be
                # summed efficiently across workers using fast-stat-sync
                assert len(bleu.counts) == EVAL_BLEU_ORDER
                for i in range(EVAL_BLEU_ORDER):
                    logging_output["_bleu_counts_" + str(i)] = bleu.counts[i]
                    logging_output["_bleu_totals_" + str(i)] = bleu.totals[i]
            return loss, sample_size, logging_output
