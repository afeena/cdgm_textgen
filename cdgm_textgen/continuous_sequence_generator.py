# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from cdgm_textgen.models.transformer import TransformerModelContinuous


class ContinuousSequenceGenerator(nn.Module):
    """
        Beam sequence generator for continuous output model. Do not support ensembling and/or constrains.
        Supports:
        Todo: Hierarchical beam search
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
            models (List[~fairseq.models.FairseqModel]): ensemble of models,
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
            prefix_tokens (torch.LongTensor, optional): force decoder to begin
                with these tokens
            constraints (torch.LongTensor, optional): force decoder to include
                the list of constraints
            bos_token (int, optional): beginning of sentence token
                (default: self.eos)
        """
        return self._generate(sample, **kwargs)

    def beam_search_step(self,
                         model_scores: Tensor,
                         prev_scores: Tensor,
                         bsz,
                         step: int,
                         finished: Tensor,
                         **kwargs):
        topk_token_score, topk_token_idx = torch.topk(model_scores.squeeze(), self.beam_size, dim=-1)
        # for each of the hyp in the acvtive beam choose self.beam possible extensions
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

    def get_model_scores(self, decoder_out):
        return self.model.forward_decoder_scores(decoder_out, self.metric)

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
            # length of the source text being the character length except EndOfSentence and pad
            src_lengths = (
                (src_tokens.ne(self.eos) & src_tokens.ne(self.pad)).long().sum(dim=1)
            )

        else:
            raise Exception(
                "expected src_tokens in net input. input keys: "
                + str(net_input.keys())
            )
        encoder_out = self.forward_encoder(net_input)

        # initialize buffers
        bsz = src_tokens.shape[0]
        scores = (
            torch.zeros(bsz * self.beam_size, self.max_len + 1).to(src_tokens).float()
        )  # +1 for eos; pad is never chosen for scoring
        tokens = (
            torch.zeros(bsz * self.beam_size, self.max_len + 2)
            .to(src_tokens)
            .long()
            .fill_(self.pad)
        )  # +2 for eos and pad
        tokens[:, 0] = self.eos if bos_token is None else bos_token

        new_order = torch.arange(bsz).view(-1, 1).repeat(1, self.beam_size).view(-1)
        new_order = new_order.to(src_tokens.device).long()
        encoder_out = self.model.encoder.reorder_encoder_out(encoder_out, new_order)
        finished = torch.zeros(bsz * self.beam_size).bool().to(src_tokens)
        offset = torch.arange(bsz).repeat(self.beam_size).to(src_tokens) * self.beam_size

        for step in range(1, self.max_len + 1):
            prev_scores = scores[:, step - 1].clone()
            prev_tokens = tokens[:, : step].clone()

            decoder_out = self.forward_decoder(
                prev_tokens,
                encoder_out=encoder_out,
                incremental_state=incremental_states
            )

            model_scores = self.get_model_scores(decoder_out)
            cand_scores, cand_tokens, candidate_idx = self.search(model_scores, prev_scores, bsz, step, finished)
            candidate_idx = candidate_idx + offset

            # update buffers
            active_hyp_idx = (~finished).nonzero().view(-1)
            finished_idx = finished.nonzero().long().view(-1)
            retain_hyps_idx = torch.empty(bsz * self.beam_size).to(cand_scores).long()
            retain_hyps_idx[finished_idx] = finished_idx
            num_remain = self.beam_size - torch.sum(finished)
            retain_hyps_idx[active_hyp_idx] = candidate_idx[:num_remain]

            step_tensor = torch.full((active_hyp_idx.size(0),), step).to(src_tokens).long()[..., None].t()
            idx = (active_hyp_idx, step_tensor)
            tokens[idx] = cand_tokens[:num_remain]
            scores[idx] = cand_scores[:num_remain]

            new_scores = scores.index_select(0, retain_hyps_idx.long())
            new_tokens = tokens.index_select(0, retain_hyps_idx.long())

            tokens = new_tokens
            scores = new_scores

            if step > 0:
                # get finished hyps on current step
                finished = (finished | tokens[:, step].eq(self.eos)).bool()
                if torch.all(finished):
                    break

        final_hyps = []

        # get last scores for each hyp for each sent in batch
        lengths = torch.sum(tokens.ne(self.pad).int(), dim=-1) - 1
        last_scores = torch.gather(scores, -1, lengths.unsqueeze(-1))
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

    @torch.jit.export
    def forward_encoder(self, net_input: Dict[str, Tensor]):
        return self.model.encoder.forward_torchscript(net_input)

    @torch.jit.export
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
            metric="ps",
            reduce_strategy=None
    ):
        super().__init__(model,
                         tgt_dict,
                         beam_size=beam_size,
                         min_len=min_len,
                         eos=eos,
                         metric=metric)

        self.reduce_strategy = reduce_strategy
        if search_strategy == "mixture_beam":
            assert reduce_strategy is None
            self.search = self.mixture_beam_search_step

    def mixture_beam_search_step(self, model_scores: Tensor,
                                      prev_scores: Tensor,
                                      bsz,
                                      step: int,
                                      finished: Tensor, ):
        """
        Model scores are of shape (bsz, 1, mixture, vocab_size)
        Reshape to (bsz, 1, mixture*vocab_size) and get tokens ids by t mod vocab_size
        """
        model_scores = model_scores.view(bsz, 1, -1)
        cand_scores, cand_tokens, candidate_idx = self.beam_search_step(model_scores, prev_scores, bsz, step, finished)

        cand_tokens = torch.div(cand_tokens, self.vocab_size, rounding_mode='trunc')
        return cand_scores, cand_tokens, candidate_idx

    def get_model_scores(self, decoder_out):
        return self.model.forward_decoder_scores(decoder_out, self.metric, self.reduce_strategy)


class MBRContinuousSequenceGenerator(ContinuousSequenceGenerator):
    def _generate(
            self,
            sample: Dict[str, Dict[str, Tensor]],
            bos_token: Optional[int] = None,
            **kwargs
    ):
        pass

    def _mbr_sampling_step(self,
                           model_scores: Tensor,
                           prev_scores: Tensor,
                           bsz: int,
                           num_samples: Optional[int] = 15,
                           decoder_out: Tensor = None,
                           **kwargs):
        assert self.metric == "ps", f"cannot use MBR sampling with metric {self.metric}"
        assert self.beam_size == 1, "beam > 1 is not possible with MBR"

        # sample from the model

        # if there are M mixtures, each mixture produce (1/M)*num_samples
        # so we have num_samples total for each token
        if hasattr(self.model, "num_mixture_components"):
            num_mxt = self.model.num_mixture_components
        else:
            num_mxt = 1

        model_scores = model_scores.view(-1, self.vocab_size)
        cand_samples = torch.multinomial(model_scores, num_samples=num_samples // num_mxt, replacement=True)
        # sample candidates BxTxnum_samples
        cand_samples = cand_samples.view(bsz, -1, num_samples)

        # now calculate pairwise cosine similarity for candidate hyp
        # use mbr sampling similar to https://arxiv.org/pdf/2108.04718.pdf
        # make it for each partial hyp
        # utility is cosine sim between hyps
        # p(y|x) is PS prob

        return None
