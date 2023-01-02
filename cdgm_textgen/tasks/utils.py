import torch
from fairseq.dataclass import FairseqDataclass


def build_model(self, cfg: FairseqDataclass, from_checkpoint=False):
    """
    Build the :class:`~fairseq.models.BaseFairseqModel` instance for this
    task.

    Args:
        cfg (FairseqDataclass): configuration object

    Returns:
        a :class:`~fairseq.models.BaseFairseqModel` instance
    """
    from cdgm_textgen import models
    from fairseq import quantization_utils

    model = models.build_model(cfg, self, from_checkpoint)
    model = quantization_utils.quantize_model_scalar(model, cfg)
    return model


def build_model_levenshtein(self, cfg, from_checkpoint=False):
    model = super().build_model(cfg, from_checkpoint=from_checkpoint)
    if getattr(cfg, "approximate_knn", False) and hasattr(model.decoder, "faiss_index"):
        model.decoder.build_faiss_index()
    elif hasattr(model.decoder, "nn_search_matrix"):
        model.decoder.setup_search_matrix()
    return model


def valid_step(self, sample, model, criterion):
    EVAL_BLEU_ORDER = 4
    model.eval()
    with torch.no_grad():
        sample["prev_target"] = self.inject_noise(sample["target"])
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
