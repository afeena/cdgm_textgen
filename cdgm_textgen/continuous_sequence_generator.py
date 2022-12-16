# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from fairseq import search, utils
from torch import Tensor
import numpy as np

from cdgm_textgen.models.transformer import TransformerModelContinuous
from cdgm_textgen.sequence_generator import EnsembleModel


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
        self.model = model

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
        self.model.set_decoder_beam_size(self.beam_size)

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
        if any(getattr(m, "build_index", False) for m in self.model.models):
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
    ):
        incremental_states = torch.jit.annotate(
            List[Dict[str, Dict[str, Optional[Tensor]]]],
            [
                torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {})
                for i in range(self.model.models_size)
            ],
        )
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
        encoder_out = self.model.encoder.forward(net_input, src_lengths)

        # initialize buffers
        bsz, src_len = src_tokens.size()[:2]

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
        beam = [(scores, tokens)]

        # search
        for step in range(self.max_len):

            new_beam = []

            expanded_any = False
            for curr_score, curr_tokens in beam:

                # check if sequence any sentence is finished
                if step > 0 and curr_tokens[:,step] in (self.eos, self.pad):
                    new_beam.append((curr_score, curr_seq))
                    continue

                expanded_any = True

                # not finished; expand

                ## this part is model-specific
                lprobs, _ = self.model.decoder.forward(
                    curr_seq[:step + 1].unsqueeze(0),
                    encoder_out=encoder_out,
                    incremental_state=incremental_states
                )

                lprobs = lprobs.squeeze(0)[-1]
                lprobs = torch.nn.functional.log_softmax(lprobs, dim=0)
                ## end model-specific

                topk = lprobs.topk(self.beam_size)

                new_candidates = [
                    (curr_score + topk.values[j].item(),
                     self._append_tok(curr_seq, step, topk.indices[j]))
                    for j in range(self.beam_size)
                ]

                new_beam.extend(new_candidates)

            if not expanded_any:
                print("All hypotheses in beam are completed, stopping early.")
                break

            new_beam.sort(key=lambda pair: pair[0], reverse=True)
            new_beam = new_beam[:self.beam_size]

            beam = new_beam

            print(f"Beam at step {step}")
            for curr_score, curr_seq in beam:
                curr_seq = curr_seq.tolist()
                # find eos from right
                curr_seq = curr_seq[1:]
                try:
                    last = curr_seq.index(self.eos)
                except ValueError:
                    last = step
                trim_seq = curr_seq[:last + 1]
                print(f"{curr_score:>8.2f} | {m.decode(trim_seq):<70} | {trim_seq}")

