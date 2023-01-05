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
            self.search = self._beam_search_step

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

    def _beam_search_step(self, model_scores: Tensor,
                          prev_scores: Tensor,
                          bsz: int,
                          offset: Optional[Tensor] = None,
                          **kwargs):
        topk_token_score, topk_token_idx = torch.topk(model_scores.squeeze(), self.beam_size, dim=-1)
        # for each of the hyp in the acvtive beam choose self.beam possible extensions
        new_candidate_scores = prev_scores.unsqueeze(-1).repeat(1, self.beam_size) + topk_token_score
        _, best_cand_idx = new_candidate_scores.view(bsz, -1).topk(self.beam_size, dim=-1)
        best_cand_idx = best_cand_idx.view(-1) + offset
        # set new scores
        best_cand_scores = new_candidate_scores.view(-1)[best_cand_idx]
        best_cand_tokens = topk_token_idx.view(-1)[best_cand_idx]

        return best_cand_scores, best_cand_tokens

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
            torch.zeros(bsz * self.beam_size, 1).to(src_tokens).float()
        )  # +1 for eos; pad is never chosen for scoring
        tokens = (
            torch.zeros(bsz * self.beam_size, 1)
            .to(src_tokens)
            .long()
            .fill_(self.eos if bos_token is None else bos_token)
        )  # +2 for eos and pad

        new_order = torch.arange(bsz).view(-1, 1).repeat(1, self.beam_size).view(-1)
        new_order = new_order.to(src_tokens.device).long()
        encoder_out = self.model.encoder.reorder_encoder_out(encoder_out, new_order)
        offset = torch.arange(start=0, end=bsz * self.beam_size, step=self.beam_size)
        offset = offset.view(-1, 1).repeat(1, self.beam_size).view(-1).to(src_tokens.device).long()
        finished = torch.tensor([False for i in range(bsz * self.beam_size)]).to(src_tokens.device).long()

        beam = {"tokens": tokens,
                "score": scores[:, -1],
                "attention": torch.empty(0),  # src_len x tgt_len
                "alignment": torch.empty(0),
                "positional_scores": scores}

        for step in range(1, self.max_len + 1):
            prev_scores = beam["score"]
            prev_tokens = beam["tokens"]

            decoder_out = self.forward_decoder(
                prev_tokens,
                encoder_out=encoder_out,
                incremental_state=incremental_states
            )

            model_scores = self.get_model_scores(decoder_out)
            best_cand_scores, best_cand_tokens = self.search(model_scores, prev_scores, bsz, offset)

            curr_scores = torch.hstack((beam["positional_scores"], best_cand_scores.unsqueeze(-1)))
            curr_tokens = torch.hstack((prev_tokens, best_cand_tokens.unsqueeze(-1)))

            if step > 0:
                # pad previously finished hyps
                curr_scores[:, step][finished] = prev_scores[finished]
                curr_tokens[:, step][finished] = self.pad
                # get finished hyps on current step
                finished = (finished | curr_tokens[:, step].eq(self.eos)).bool()
                if torch.all(finished):
                    break

            beam = {"tokens": curr_tokens,
                    "score": curr_scores[:, -1],
                    "attention": torch.empty(0),  # src_len x tgt_len
                    "alignment": torch.empty(0),
                    "positional_scores": curr_scores}

        final_hyps = []
        top_hyp_scores, top_hyp_idx = beam["score"].view(bsz, -1).max(-1)

        for i in range(bsz):
            toks = beam["tokens"][top_hyp_idx[i] + offset[i + self.beam_size]]
            length = sum(toks.ne(self.pad).int())
            score = beam["score"][top_hyp_idx[i] + offset[i + self.beam_size]].item()
            pos_scores = beam["positional_scores"][top_hyp_idx[i] + offset[i + self.beam_size]]
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
            reduce_strategy="max"
    ):
        super().__init__(model,
                         tgt_dict,
                         beam_size=beam_size,
                         min_len=min_len,
                         eos=eos,
                         metric=metric)

        self.reduce_strategy = reduce_strategy
        if search_strategy == "hierarchical_beam":
            assert reduce_strategy is None
            self.search = self._hierarchical_beam_search_step

    def _hierarchical_beam_search_step(self, model_scores: Tensor,
                                       prev_scores: Tensor,
                                       bsz: int,
                                       offset: Optional[Tensor] = None,
                                       **kwargs):
        topk_token_score, topk_token_idx = torch.topk(model_scores.squeeze(), self.beam_size, dim=-1)
        # for each of the hyp in the acvtive beam choose self.beam possible extensions
        new_candidate_scores = prev_scores.unsqueeze(-1).repeat(1, self.beam_size) + topk_token_score
        _, best_cand_idx = new_candidate_scores.view(bsz, -1).topk(self.beam_size, dim=-1)
        best_cand_idx = best_cand_idx.view(-1) + offset
        # set new scores
        best_cand_scores = new_candidate_scores.view(-1)[best_cand_idx]
        best_cand_tokens = topk_token_idx.view(-1)[best_cand_idx]

        return best_cand_scores, best_cand_tokens

    def get_model_scores(self, decoder_out):
        return self.model.forwrd_decoder_scores(decoder_out, self.metric, self.reduce_strategy)


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
