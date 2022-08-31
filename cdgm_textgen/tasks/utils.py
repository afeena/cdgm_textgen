import torch

from fairseq.dataclass import FairseqDataclass
from fairseq import search


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
        model.decoder.setup_faiss_index()
    elif hasattr(model.decoder, "nn_search_matrix"):
        model.decoder.setup_search_matrix()
    return model


def build_generator(self, models, args, seq_gen_cls=None,
                    extra_gen_cls_kwargs=None,
                    prefix_allowed_tokens_fn=None):

    gen_type = getattr(args, "gen_type", None)

    if gen_type == "nat":
        from cdgm_textgen.iterative_refinement_generator import IterativeRefinementGenerator

        return IterativeRefinementGenerator(
            self.target_dictionary,
            eos_penalty=getattr(args, "iter_decode_eos_penalty", 0.0),
            max_iter=getattr(args, "iter_decode_max_iter", 2),
            beam_size=getattr(args, "iter_decode_with_beam", 1),
            reranking=getattr(args, "iter_decode_with_external_reranker", False),
            decoding_format=getattr(args, "decoding_format", None),
            adaptive=not getattr(args, "iter_decode_force_max_iter", False),
            retain_history=getattr(args, "retain_iter_history", False),
            measure=getattr(args, "decoding_measure", "cosine"),
            decoding_masking=getattr(args, "decoding_masking", "distance"),
            approximate_knn=getattr(args, "approximate_knn", False),

        )
    if gen_type == "seq2seq":
        if getattr(args, "score_reference", False):
            from fairseq.sequence_scorer import SequenceScorer

            return SequenceScorer(
                self.target_dictionary,
                compute_alignment=getattr(args, "print_alignment", False),
            )

        from fairseq.sequence_generator import (
            SequenceGenerator,
            SequenceGeneratorWithAlignment,
        )

        # Choose search strategy. Defaults to Beam Search.
        sampling = getattr(args, "sampling", False)
        sampling_topk = getattr(args, "sampling_topk", -1)
        sampling_topp = getattr(args, "sampling_topp", -1.0)
        diverse_beam_groups = getattr(args, "diverse_beam_groups", -1)
        diverse_beam_strength = getattr(args, "diverse_beam_strength", 0.5)
        match_source_len = getattr(args, "match_source_len", False)
        diversity_rate = getattr(args, "diversity_rate", -1)
        constrained = getattr(args, "constraints", False)
        if prefix_allowed_tokens_fn is None:
            prefix_allowed_tokens_fn = getattr(args, "prefix_allowed_tokens_fn", None)
        if (
                sum(
                    int(cond)
                    for cond in [
                        sampling,
                        diverse_beam_groups > 0,
                        match_source_len,
                        diversity_rate > 0,
                    ]
                )
                > 1
        ):
            raise ValueError("Provided Search parameters are mutually exclusive.")
        assert sampling_topk < 0 or sampling, "--sampling-topk requires --sampling"
        assert sampling_topp < 0 or sampling, "--sampling-topp requires --sampling"

        if sampling:
            search_strategy = search.Sampling(
                self.target_dictionary, sampling_topk, sampling_topp
            )
        elif diverse_beam_groups > 0:
            search_strategy = search.DiverseBeamSearch(
                self.target_dictionary, diverse_beam_groups, diverse_beam_strength
            )
        elif match_source_len:
            # this is useful for tagging applications where the output
            # length should match the input length, so we hardcode the
            # length constraints for simplicity
            search_strategy = search.LengthConstrainedBeamSearch(
                self.target_dictionary,
                min_len_a=1,
                min_len_b=0,
                max_len_a=1,
                max_len_b=0,
            )
        elif diversity_rate > -1:
            search_strategy = search.DiverseSiblingsSearch(
                self.target_dictionary, diversity_rate
            )
        elif constrained:
            search_strategy = search.LexicallyConstrainedBeamSearch(
                self.target_dictionary, args.constraints
            )
        elif prefix_allowed_tokens_fn:
            search_strategy = search.PrefixConstrainedBeamSearch(
                self.target_dictionary, prefix_allowed_tokens_fn
            )
        else:
            search_strategy = search.BeamSearch(self.target_dictionary)

        extra_gen_cls_kwargs = extra_gen_cls_kwargs or {}
        if seq_gen_cls is None:
            if getattr(args, "print_alignment", False):
                seq_gen_cls = SequenceGeneratorWithAlignment
                extra_gen_cls_kwargs["print_alignment"] = args.print_alignment
            else:
                seq_gen_cls = SequenceGenerator

        return seq_gen_cls(
            models,
            self.target_dictionary,
            beam_size=getattr(args, "beam", 5),
            max_len_a=getattr(args, "max_len_a", 0),
            max_len_b=getattr(args, "max_len_b", 200),
            min_len=getattr(args, "min_len", 1),
            normalize_scores=(not getattr(args, "unnormalized", False)),
            len_penalty=getattr(args, "lenpen", 1),
            unk_penalty=getattr(args, "unkpen", 0),
            temperature=getattr(args, "temperature", 1.0),
            match_source_len=getattr(args, "match_source_len", False),
            no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
            search_strategy=search_strategy,
            **extra_gen_cls_kwargs,
        )

    if gen_type == "custom":
        from cdgm_textgen.custom_generator import NaiveBeamGenerator
        seq_gen_cls = NaiveBeamGenerator
        seq_gen_cls(
            models,
            self.target_dictionary,
            beam_size=getattr(args, "beam", 5),
            max_len_a=getattr(args, "max_len_a", 0),
            max_len_b=getattr(args, "max_len_b", 200),
            min_len=getattr(args, "min_len", 1),
            normalize_scores=(not getattr(args, "unnormalized", False)),
            len_penalty=getattr(args, "lenpen", 1),
            unk_penalty=getattr(args, "unkpen", 0),
            temperature=getattr(args, "temperature", 1.0),
            match_source_len=getattr(args, "match_source_len", False),
            no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
            search_strategy=search.BeamSearch(self.target_dictionary)
        )


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
