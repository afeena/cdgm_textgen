# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from fairseq import search
from torch import Tensor

from cdgm_textgen.models.transformer import TransformerModelContinuous


class ContinuousSequenceGenerator(nn.Module):
    """
        Beam sequence generator for continuous output model. Do not support ensembling and/or constrains.
        Supports:
        Todo: Beam search.
        Todo: Hierarchical beam search
        Todo: MBR decoding.
    """

    def __init__(
            self,
            model,
            tgt_dict,
            beam_size=1,
            min_len=1,
            normalize_scores=True,
            search_strategy=None,
            eos=None,
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

        self.min_len = min_len
        self.max_len = self.model.max_decoder_positions()

        self.normalize_scores = normalize_scores

        self.search = (
            search.BeamSearch(tgt_dict) if search_strategy is None else search_strategy
        )
        # We only need to set src_lengths in LengthConstrainedBeamSearch.
        # As a module attribute, setting it would break in multithread
        # settings when the model is shared.
        self.should_set_src_lengths = (
                hasattr(self.search, "needs_src_lengths") and self.search.needs_src_lengths
        )
        self.metric = metric
        if getattr(self.model, "build_index", False):
            self.model.build_index(metric=self.metric)

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

    def _append_tok(self, tensor, step, ix):
        return tensor.index_fill(0, torch.tensor(step + 1), ix)

    def _generate(
            self,
            sample: Dict[str, Dict[str, Tensor]],
            prefix_tokens: Optional[Tensor] = None,
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
        encoder_out = self.model.encoder.forward(net_input["src_tokens"], src_lengths)

        # initialize buffers
        bsz, src_len = src_tokens.size()[:2]

        scores = (
            torch.zeros(bsz * self.beam_size, self.max_len + 2).to(src_tokens).float()
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
        offset = torch.arange(start=0, end=bsz, step=self.beam_size).to(src_tokens.device).long()
        finished = torch.tensor([False for i in range(bsz*self.beam_size)]).to(src_tokens.device).long()
        last_step = -1

        for step in range(self.max_len):
            curr_scores = scores
            curr_tokens = tokens
            if step > 0:
                # here get mask of finished setneces
                finished = finished.logical_or(curr_tokens[:, step].eq(self.eos))
                print(finished)
                if torch.all(finished):
                    last_step=step
                    break

            decoder_out = self.model.decoder.forward(
                curr_tokens[:, :step + 1],
                encoder_out=encoder_out,
                incremental_state=incremental_states
            )
            scores = self.model.get_similarity_scores(decoder_out, metric=self.metric)

            topk = torch.topk(scores.squeeze(), self.beam_size, dim=-1)
            new_candidate_scores = curr_scores[:, step].unsqueeze(-1).repeat(1, self.beam_size) + topk.values
            new_candidate_tokens = topk.indices
            best_cand = new_candidate_scores.view(bsz, -1).topk(self.beam_size, dim=-1)
            #set new scores
            curr_scores[:, step + 1] = best_cand.values.view(-1)
            curr_tokens[:, step + 1] = new_candidate_tokens.view(-1)[best_cand.indices.view(-1) + offset]
            #copy scores from prev step for finished hyps
            curr_scores[:, step+1][finished] = curr_scores[:, step][finished]

            scores = curr_scores
            tokens = curr_tokens

        return [[{"tokens": tokens[i:i+self.beam_size, 1:last_step],
                  "score": scores[i:i+self.beam_size, last_step - 1],
                  "attention": torch.empty(0),  # src_len x tgt_len
                  "alignment": torch.empty(0),
                  "positional_scores": torch.empty(0)}
                 ] for i in range(0,bsz*self.beam_size+1,self.beam_size)]
