# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
from argparse import Namespace
from typing import Dict, List, Optional

from fairseq.data import encoders
import torch
import torch.nn as nn
from torch import Tensor

from cdgm_textgen.models.transformer import TransformerModelContinuous


class ContinuousSequenceGenerator(nn.Module):
    """
        Beam sequence generator for continuous output model. Do not support ensembling and/or constrains.
        Supports:
            - beam search with cosine similarity/ps likelihood
    """

    def __init__(
            self,
            model,
            tgt_dict,
            beam_size=1,
            min_len=1,
            eos=None,
            search_strategy="beam",
            metric="cosine",
    ):
        """Generates translations of a given source sentence.

        Args:
            model (List[~fairseq.models.FairseqModel]): ensemble of models,
                currently support fairseq.models.TransformerModel for scripting
            beam_size (int, optional): beam width (default: 1)
        """
        super().__init__()
        self.model = model[0]

        assert isinstance(self.model, TransformerModelContinuous), \
            f"model._name has to be an instance of TransformerModelContinuous, got {type(self.model)}"

        self.tgt_dict = tgt_dict
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos() if eos is None else eos
        self.vocab_size = len(tgt_dict)

        self.beam_size = beam_size
        # the max beam size is the dictionary size - 1, since we never select pad
        self.beam_size = min(beam_size, self.vocab_size - 1)
        self.model.decoder.set_beam_size(self.beam_size)

        self.min_len = min_len
        self.max_len = self.model.max_decoder_positions()

        if search_strategy == "beam":
            self.search = self.beam_search_step

        self.metric = metric

        self.model.eval()

    def cuda(self):
        self.model.cuda()
        return self

    @torch.no_grad()
    def forward(
            self,
            sample: Dict[str, Dict[str, Tensor]],
            prefix_tokens: Optional[Tensor] = None,
            bos_token: Optional[int] = None,
    ):
        """Generate a batch of translations.

        Args:
            sample (dict): batch
            prefix_tokens (torch.LongTensor, optional): force decoder to begin
                with these tokens
            bos_token (int, optional): beginning of sentence token
                (default: self.eos)
        """
        return self._generate(sample, prefix_tokens, bos_token=bos_token)

    @torch.no_grad()
    def generate(
            self, models, sample: Dict[str, Dict[str, Tensor]], **kwargs
    ) -> List[List[Dict[str, Tensor]]]:
        """Generate translations. Match the api of other fairseq generators.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models
            sample (dict): batch
        """
        return self._generate(sample, **kwargs)

    def beam_search_step(self,
                         topk_token_score: Tensor,
                         topk_token_idx: Tensor,
                         prev_scores: Tensor,
                         bsz,
                         step: int,
                         finished: Tensor,
                         **kwargs):

        # for each of the hyp in the active beam choose self.beam possible extensions
        if len(topk_token_score.size()) == 1:
            topk_token_score = topk_token_score.unsqueeze(0)
            topk_token_idx = topk_token_idx.unsqueeze(0)
        new_candidate_scores = prev_scores.unsqueeze(-1).repeat(1, 1, self.beam_size) + topk_token_score
        if step == 1:
            # at the first step all hypotheses are equally likely, so use
            # only the first beam
            best_cand_scores = topk_token_score[::self.beam_size, :].contiguous()
            best_cand_tokens = topk_token_idx[::self.beam_size, :].contiguous()
            best_cand_idx = torch.zeros(bsz * self.beam_size, dtype=torch.long, device=topk_token_idx.device)
        else:
            # at the next steps we have to choose the best candidates 
            new_candidate_scores[finished.unsqueeze(0)] = float('-inf')
            new_candidate_scores = new_candidate_scores.view(bsz, -1)

            # remove finished hyps from consideration
            best_cand_scores, best_cand_idx = torch.topk(
                new_candidate_scores, self.beam_size, dim=-1
            )
            best_cand_scores = best_cand_scores  # (bsz * beam_size)
            best_cand_tokens = topk_token_idx.view(bsz, -1).gather(1, best_cand_idx)

        return best_cand_scores.view(-1), best_cand_tokens.view(-1), torch.div(best_cand_idx, self.beam_size,
                                                                               rounding_mode='trunc').view(-1)

    @torch.no_grad()
    def get_topk_model_scores(self, decoder_out):
        model_scores = self.model.forward_decoder_scores(decoder_out, self.metric)
        topk_token_score, topk_token_idx = torch.topk(model_scores.squeeze(), self.beam_size, dim=-1)
        return topk_token_score, topk_token_idx

    @torch.no_grad()
    def initialize_buffers(self, bsz, device, bos_token):
        scores = (
            torch.zeros(bsz * self.beam_size, self.max_len + 1, device=device, dtype=torch.float)
        )  # +1 for eos; pad is never chosen for scoring
        tokens = (
            torch.full((bsz * self.beam_size, self.max_len + 2), self.pad, device=device, dtype=torch.long)
        )  # +2 for eos and pad
        tokens[:, 0] = self.eos if bos_token is None else bos_token
        finished = torch.zeros(bsz * self.beam_size, device=device, dtype=torch.bool)
        offset = torch.arange(bsz, device=device).repeat(self.beam_size) * self.beam_size

        return scores, tokens, finished, offset

    @torch.no_grad()
    def _generate(
            self,
            sample: Dict[str, Dict[str, Tensor]],
            bos_token: Optional[int] = None,
            **kwargs
    ):
        incremental_states = torch.jit.annotate(
            Dict[str, Dict[str, Optional[Tensor]]], {})

        net_input = sample["net_input"]
        if "src_tokens" in net_input:
            src_tokens = net_input["src_tokens"]

        else:
            raise Exception(
                "expected src_tokens in net input. input keys: "
                + str(net_input.keys())
            )
        encoder_out = self.forward_encoder(net_input)

        # initialize buffers
        bsz = src_tokens.shape[0]
        scores, tokens, finished, offset = self.initialize_buffers(bsz, src_tokens.device, bos_token)

        new_order = torch.arange(bsz).view(-1, 1).repeat(1, self.beam_size).view(-1)
        new_order = new_order.to(src_tokens.device).long()
        encoder_out = self.model.encoder.reorder_encoder_out(encoder_out, new_order)

        for step in range(1, self.max_len + 1):
            prev_scores = scores[:, step - 1].detach().clone()
            prev_tokens = tokens[:, : step].detach().clone()

            decoder_out = self.forward_decoder(
                prev_tokens,
                encoder_out=encoder_out,
                incremental_state=incremental_states
            )

            topk_token_score, topk_token_idx = self.get_topk_model_scores(decoder_out)
            cand_scores, cand_tokens, candidate_idx = self.search(topk_token_score, topk_token_idx, prev_scores, bsz,
                                                                  step, finished)
            candidate_idx = candidate_idx + offset

            # update buffers
            active_hyp_idx = (~finished).nonzero().view(-1)
            finished_idx = finished.nonzero().long().view(-1)
            retain_hyps_idx = torch.empty(bsz * self.beam_size, device=cand_scores.device, dtype=torch.long)
            retain_hyps_idx[finished_idx] = finished_idx
            num_remain = self.beam_size - torch.sum(finished)
            retain_hyps_idx[active_hyp_idx] = candidate_idx[:num_remain]

            step_tensor = torch.full((1, active_hyp_idx.size(0)), step, device=src_tokens.device, dtype=torch.long)
            idx = (active_hyp_idx, step_tensor)
            tokens[idx] = cand_tokens[:num_remain]
            scores[idx] = cand_scores[:num_remain]

            new_scores = scores.index_select(0, retain_hyps_idx)
            new_tokens = tokens.index_select(0, retain_hyps_idx)

            tokens = new_tokens
            scores = new_scores

            if step > 0:
                # get finished hyps on current step
                finished = (finished | tokens[:, step].eq(self.eos)).bool()
                if torch.all(finished):
                    break
        return self.finalize_hyps(scores, tokens, bsz)

    def finalize_hyps(self, scores, tokens, bsz):
        final_hyps = []

        # get last scores for each hyp for each sent in batch
        lengths = torch.sum(tokens.ne(self.pad).int(), dim=-1) - 1
        last_scores = torch.gather(scores, -1, lengths.unsqueeze(-1))
        last_scores = last_scores.float()
        last_scores /= (lengths.float() + 1e-8).unsqueeze(-1)
        top_hyp_scores, top_hyp_idx = last_scores.view(bsz, -1).max(-1)

        for i in range(bsz):
            toks = tokens[top_hyp_idx[i] + i * self.beam_size]
            length = sum(toks.ne(self.pad).int())
            score = last_scores[top_hyp_idx[i] + i * self.beam_size].item()
            pos_scores = scores[top_hyp_idx[i] + i * self.beam_size]
            pos_scores[1:] = pos_scores[1:] - pos_scores[:-1]

            final_hyps.append([{"tokens": toks[1:length],
                                "score": score,
                                "attention": torch.empty(0),  # src_len x tgt_len
                                "alignment": torch.empty(0),
                                "positional_scores": pos_scores[1:length]}]
                              )
        return final_hyps

    def forward_encoder(self, net_input: Dict[str, Tensor]):
        return self.model.encoder.forward_torchscript(net_input)

    def forward_decoder(self,
                        tokens,
                        encoder_out: Dict[str, List[Tensor]],
                        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
                        **kwargs):
        with torch.no_grad():
            decoder_out = self.model.decoder.forward(
                tokens,
                encoder_out=encoder_out,
                incremental_state=incremental_state,
            )
        return decoder_out


class MixtureContinuousSequenceGenerator(ContinuousSequenceGenerator):
    def __init__(
            self,
            model,
            tgt_dict,
            beam_size=1,
            min_len=1,
            eos=None,
            search_strategy="beam",
            reduce_strategy=None,
            mixture_temperature=1.0,
            metric="cosine",
            **kwargs
    ):
        if beam_size > model[0].decoder.num_mixture_components:
            raise ValueError("Beam size cannot be larger than the number of mixtures")
        super().__init__(model,
                         tgt_dict,
                         beam_size=beam_size,
                         min_len=min_len,
                         eos=eos,
                         metric=metric)

        self.reduce_strategy = reduce_strategy
        if self.metric != "ps":
            logging.warning("PS decoding measure is recommended to use with a mixture model")

        if search_strategy == "mixture_beam":
            assert reduce_strategy is None or reduce_strategy == "None"
        self.mixture_temperature = mixture_temperature

    def get_topk_model_scores(self, decoder_out):
        model_scores = self.model.forward_decoder_scores(decoder_out, self.metric, self.reduce_strategy,
                                                         self.mixture_temperature)
        if self.reduce_strategy is None or self.reduce_strategy == "None":
            best_tokens_scores_by_mixtures, best_tokens_idx_by_mixtures = torch.max(model_scores, dim=-1)
            topk_token_score, topk_token_pos = best_tokens_scores_by_mixtures.t().topk(self.beam_size, dim=-1)
            topk_token_idx = best_tokens_idx_by_mixtures.gather(0, topk_token_pos)
        else:
            topk_token_score, topk_token_idx = torch.topk(model_scores.squeeze(), self.beam_size, dim=-1)

        return topk_token_score, topk_token_idx


class MBRContinuousSequenceGenerator(ContinuousSequenceGenerator):
    def __init__(self,
                 model,
                 tgt_dict,
                 beam_size=1,
                 min_len=1,
                 eos=None,
                 metric="ps",
                 n_samples=1,
                 **kwargs
                 ):
        super().__init__(model, tgt_dict,
                         beam_size=beam_size,
                         min_len=min_len,
                         eos=eos,
                         metric=metric,
                         search_strategy="beam", )
        assert self.beam_size == 1, "beam > 1 is not possible with MBR"
        self.n_samples = n_samples
        self.tokenizer = encoders.build_tokenizer(Namespace(
            tokenizer='moses'
        ))

    @torch.no_grad()
    def get_topk_model_scores(self, decoder_out):
        log_probs, topk_token_idx = self.model.forward_samples(decoder_out[0], decoder_out[1]["mixtures_out"],  self.n_samples)
        return log_probs, topk_token_idx

    @torch.no_grad()
    def initialize_buffers(self, bsz, device, bos_token):
        tokens = (
            torch.full((bsz * self.n_samples, self.max_len + 2), self.pad, device=device, dtype=torch.long)
        )  # +2 for eos and pad
        tokens[:, 0] = self.eos if bos_token is None else bos_token
        log_probs = torch.zeros(bsz * self.n_samples, device=device, dtype=torch.float)
        finished = torch.zeros(bsz * self.n_samples, device=device, dtype=torch.bool)

        return tokens, log_probs, finished

    def _generate(
            self,
            sample: Dict[str, Dict[str, Tensor]],
            bos_token: Optional[int] = None,
            **kwargs
    ):
        # generate n_samples hypotheses left-to-write
        # continue generation until all hypotheses are finished
        # return the best hypothesis
        incremental_states = torch.jit.annotate(
            Dict[str, Dict[str, Optional[Tensor]]], {})

        net_input = sample["net_input"]
        if "src_tokens" in net_input:
            src_tokens = net_input["src_tokens"]

        else:
            raise Exception(
                "expected src_tokens in net input. input keys: "
                + str(net_input.keys())
            )
        encoder_out = self.forward_encoder(net_input)

        # initialize buffers
        bsz = src_tokens.shape[0]
        tokens, log_probs, finished = self.initialize_buffers(bsz, src_tokens.device, bos_token)

        new_order = torch.arange(bsz).view(-1, 1).repeat(1, self.beam_size).view(-1)
        new_order = new_order.to(src_tokens.device).long()
        encoder_out = self.model.encoder.reorder_encoder_out(encoder_out, new_order)

        for step in range(1, self.max_len + 1):
            prev_tokens = tokens[:, : step].detach().clone()

            decoder_out = self.forward_decoder(
                prev_tokens,
                encoder_out=encoder_out,
                incremental_state=incremental_states
            )

            l_probs, sampled_mixture_tokens = self.get_topk_model_scores(decoder_out)

            # update buffers
            active_hyp_idx = (~finished).nonzero().view(-1)
            finished_idx = finished.nonzero().long().view(-1)
            retain_hyps_idx = torch.empty(bsz * self.n_samples, device=sampled_mixture_tokens.device,
                                          dtype=torch.long)
            retain_hyps_idx[finished_idx] = finished_idx
            num_remain = self.n_samples - torch.sum(finished)
            retain_hyps_idx[active_hyp_idx] = sampled_mixture_tokens[:num_remain]

            step_tensor = torch.full((1, active_hyp_idx.size(0)), step, device=src_tokens.device, dtype=torch.long)
            idx = (active_hyp_idx, step_tensor)
            tokens[idx] = sampled_mixture_tokens[:num_remain]
            log_probs[active_hyp_idx] += l_probs[active_hyp_idx]

            new_tokens = tokens.index_select(0, retain_hyps_idx)

            tokens = new_tokens

            if step > 0:
                # get finished hyps on current step
                finished = (finished | tokens[:, step].eq(self.eos)).bool()
                if torch.all(finished):
                    break

        return self.finalize_hyps(tokens, log_probs, bsz)

    def finalize_hyps(self, tokens, log_probs, bsz):
        import numpy as np
        import sacrebleu
        final_hyps = []

        # get last scores for each hyp for each sent in batch
        lengths = torch.sum(tokens.ne(self.pad).int(), dim=-1) - 1
        tokens = tokens.view(bsz, self.n_samples, -1)
        log_probs = log_probs.view(bsz, self.n_samples)

        # since we use log_probs, either take exp or negate uktility score and take min
        def _decode(toks, escape_unk=False):
            s = self.model.decoder.dictionary.string(
                toks.int().cpu(),
                "@@ ",
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=(
                    "UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"
                ),
            )
            s = self.tokenizer.decode(s)
            return s

        for i in range(bsz):

            hyp_strings = _decode(tokens[i])
            hyp_strings = hyp_strings.split("\n")

            hyp_strings = np.expand_dims(np.array(hyp_strings), axis=-1)
            x, y = np.broadcast_arrays(np.expand_dims(hyp_strings, axis=-1), np.expand_dims(hyp_strings, axis=0))
            res = np.concatenate((x, y))
            hyps = res[:, :, 0].tolist()
            tgts = res[:, :, 1].tolist()

            utility_scores = [
                [
                    sacrebleu.metrics.BLEU().sentence_score(hyps[k][j], [tgts[k][j]])
                    for j in range(len(hyp_strings))
                ]
                for k in range(len(hyp_strings))
            ]
            scores = (-1.0*torch.tensor(utility_scores) * log_probs[i]).sum(dim=-1)
            best_score, best_idx = scores.max(dim=-1)


            # for each hyp in samples, calculate metric given other samples as targets
            # calculate sum[log_probs*(-metric)]

            final_hyps.append([{"tokens":  tokens[i][best_idx][1:lengths[i]],
                                "score": best_score,
                                "attention": torch.empty(0),  # src_len x tgt_len
                                "alignment": torch.empty(0),
                                "positional_scores": None}]
                              )
        return final_hyps
