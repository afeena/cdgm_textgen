import json
from argparse import Namespace
from dataclasses import dataclass, field

from fairseq.data import encoders
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationConfig
from fairseq.tasks.translation import TranslationTask

EVAL_BLEU_ORDER = 4


@dataclass
class KnnTranslationConfig(TranslationConfig):
    knn_k: int = field(default=32, metadata={"help": "number of nearest neighbors to retrieve"})
    knn_lambda: float = field(default=0.5, metadata={"help": "lambda for knn search"})
    knn_temperature: float = field(default=1.0, metadata={"help": "temperature for knn search"})
    knn_index_file: str = field(default="/", metadata={"help": "path to the knn index file"})

@register_task("translation_knn", dataclass=KnnTranslationConfig)
class KnnTranslationTask(TranslationTask):

    def build_generator(self, models, args, seq_gen_cls=None,
                        extra_gen_cls_kwargs=None,
                        prefix_allowed_tokens_fn=None):

        from cdgm_textgen.knn_sequence_generator import KnnMTSequenceGenerator
        seq_gen_cls = KnnMTSequenceGenerator


        return seq_gen_cls(
            models,
            self.target_dictionary,
            beam_size=getattr(args, "beam", 1),
            min_len=getattr(args, "min_len", 1),
            search_strategy=getattr(args, "search_strategy", "beam"),
            metric=getattr(args, "search_strategy", "cosine"),
            knn_index_file = getattr(args, "knn_index_file", None),
            knn_k=getattr(args, "knn_k", 32),
            knn_lambda=getattr(args, "knn_lambda", 0.5),
            knn_temperature=getattr(args, "knn_temperature", 1.0),
            **extra_gen_cls_kwargs
        )