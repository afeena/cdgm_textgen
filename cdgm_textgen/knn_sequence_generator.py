import math
from typing import Dict, List, Optional

import torch
from torch import Tensor
import faiss.contrib.torch_utils
from cdgm_textgen.sequence_generator import SequenceGenerator
import numpy as np

class KnnMTSequenceGenerator(SequenceGenerator):
    def __init__(self,
                 models,
                 tgt_dict,
                 beam_size=1,
                 max_len_a=0,
                 max_len_b=200,
                 max_len=0,
                 min_len=1,
                 normalize_scores=True,
                 len_penalty=1.0,
                 unk_penalty=0.0,
                 temperature=1.0,
                 match_source_len=False,
                 no_repeat_ngram_size=0,
                 search_strategy=None,
                 eos=None,
                 symbols_to_strip_from_output=None,
                 lm_model=None,
                 lm_weight=1.0,
                 tokens_to_suppress=(),
                 metric="cosine",
                 knn_index_file=None,
                 knn_k=5,
                 knn_lambda=0.5,
                 knn_temperature=1.0,
                 knn_metric="cosine",
                 knn_nprobe=32,
                 knn_pruning_weights = None,
                 ):
        super().__init__(models,
                 tgt_dict,
                 beam_size=beam_size,
                 max_len_a=max_len_a,
                 max_len_b=max_len_b,
                 max_len=max_len,
                 min_len=min_len,
                 normalize_scores=normalize_scores,
                 len_penalty=len_penalty,
                 unk_penalty=unk_penalty,
                 temperature=temperature,
                 match_source_len=match_source_len,
                 no_repeat_ngram_size=no_repeat_ngram_size,
                 search_strategy=search_strategy,
                 eos=eos,
                 symbols_to_strip_from_output=symbols_to_strip_from_output,
                 lm_model=lm_model,
                 lm_weight=lm_weight,
                 tokens_to_suppress=tokens_to_suppress,
                 metric=metric,
                )

        if knn_index_file is not None:
            res = faiss.StandardGpuResources()
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            self.knn_index = faiss.read_index(knn_index_file)
            self.knn_index = faiss.index_cpu_to_gpu(res,0,self.knn_index,co)
            self.knn_index.nprobe = knn_nprobe
            self.knn_k = knn_k
            self.knn_lambda = knn_lambda
            self.knn_temperature = knn_temperature
            self.knn_metric = knn_metric


            print("load vals to retrieve vocab indices")
            vals_memmap = np.memmap(f"{knn_index_file}_vals.npy", dtype=np.int32, mode='r', shape=(self.knn_index.ntotal,))
            self.vals = torch.zeros(self.knn_index.ntotal, dtype=torch.int64)
            self.vals[:] = torch.from_numpy(vals_memmap[:])

            if knn_pruning_weights is not None:
                print("load knn pruning weights")
                pruning_weights_memmap = np.memmap(knn_pruning_weights, dtype=np.float32, mode='r',
                                                   shape=(self.knn_index.ntotal,))
                self.knn_pruning_weights = torch.zeros(self.knn_index.ntotal, dtype=torch.float32)
                self.knn_pruning_weights[:] = torch.from_numpy(pruning_weights_memmap[:])
    def get_knn_score(self, lprobs, decoder_out):
        #rare_prediction_mapping = None
        # if self.knn_rare_only > 0:
        #     _, top_one_idx = lprobs.topk(1, dim=-1)
        #     rare_prediction_mapping = (
        #                 torch.norm(decoder_out, dim=-1).pow(2).ge(self.knn_rare_only)).squeeze().nonzero()
        #     if rare_prediction_mapping.size(0) > 0:
        #         print(f"run knn for {len(rare_prediction_mapping)} rare tokens out of {len(top_one_idx)}")
        #         decoder_out = decoder_out[rare_prediction_mapping, :]
        #         assert len(decoder_out.shape)==2
        #     else:
        #         return None

        if self.knn_metric == 'cosine':
            decoder_out = decoder_out / decoder_out.norm(dim=-1, keepdim=True)
        if len(decoder_out.shape) == 3:
            decoder_out = decoder_out.squeeze()
        knn_out_dists, knn_out_idx = self.knn_index.search(decoder_out.detach(),
                                                           self.knn_k)

        knn_target_idx = self.vals[knn_out_idx.cpu().to(torch.int32)]
        knn_target_idx = knn_target_idx.to(decoder_out.device)
        knn_p = torch.log_softmax(-knn_out_dists / self.knn_temperature, dim=-1)
        if self.knn_pruning_weights is not None:
            knn_p = knn_p * self.knn_pruning_weights[knn_target_idx]

        # here sum the log probs for each token
        # logsumexp for each token
        # here is the tricky part
        # adapted from https://github.com/urvashik/knnmt/blob/master/fairseq/knnlm.py#L160

        unique_indices, mapping = torch.unique(knn_target_idx, return_inverse=True)
        # (Bxbeam)xKxn where n = num unique vals in indices
        knn_scores_by_index = torch.ones([knn_target_idx.shape[0], knn_target_idx.shape[1], len(unique_indices)],
                                         dtype=torch.float32).to(device=decoder_out.device)
        knn_scores_by_index[:] = -10000  # -math.inf
        knn_vals_by_index = torch.ones([knn_target_idx.shape[0], knn_target_idx.shape[1], len(unique_indices)],
                                       dtype=knn_target_idx.dtype).to(device=decoder_out.device)
        knn_vals_by_index[:] = self.pad

        # (Bxbeam)x1xK
        knn_target_idx = knn_target_idx.unsqueeze(2)
        knn_p = knn_p.unsqueeze(2)
        mapping = mapping.unsqueeze(2)
        knn_scores_by_index.scatter_(dim=2, index=mapping, src=knn_p)
        knn_vals_by_index.scatter_(dim=2, index=mapping, src=knn_target_idx)
        # (Bxbeam)xn
        knn_scores_by_index = knn_scores_by_index.logsumexp(dim=1)
        knn_vals_by_index = knn_vals_by_index.max(dim=1)[0]

        full_knn_scores = torch.ones_like(lprobs)
        full_knn_scores[:] = -10000
        # if rare_prediction_mapping is not None:
        #     mapped_knn_vals_by_index = torch.zeros(lprobs.size(0), knn_vals_by_index.size(1)).long().to(
        #         device=decoder_out.device)
        #     mapped_knn_vals_by_index[rare_prediction_mapping] = knn_vals_by_index
        #     mapped_knn_scores_by_index = torch.zeros(lprobs.size(0), knn_vals_by_index.size(1), device=decoder_out.device)
        #     mapped_knn_scores_by_index[rare_prediction_mapping] = knn_scores_by_index
        #     full_knn_scores.scatter_(dim=1, index=mapped_knn_vals_by_index, src=mapped_knn_scores_by_index)
        # else:
        full_knn_scores.scatter_(dim=1, index=knn_vals_by_index, src=knn_scores_by_index)
        return full_knn_scores

    def _generate(
            self,
            sample: Dict[str, Dict[str, Tensor]],
            prefix_tokens: Optional[Tensor] = None,
            constraints: Optional[Tensor] = None,
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
        elif "source" in net_input:
            src_tokens = net_input["source"]
            src_lengths = (
                net_input["padding_mask"].size(-1) - net_input["padding_mask"].sum(-1)
                if net_input["padding_mask"] is not None
                else torch.tensor(src_tokens.size(-1)).to(src_tokens)
            )
        elif "features" in net_input:
            src_tokens = net_input["features"]
            src_lengths = (
                net_input["padding_mask"].size(-1) - net_input["padding_mask"].sum(-1)
                if net_input["padding_mask"] is not None
                else torch.tensor(src_tokens.size(-1)).to(src_tokens)
            )
        else:
            raise Exception(
                "expected src_tokens or source in net input. input keys: "
                + str(net_input.keys())
            )

        # bsz: total number of sentences in beam
        # Note that src_tokens may have more than 2 dimensions (i.e. audio features)
        bsz, src_len = src_tokens.size()[:2]
        beam_size = self.beam_size

        if constraints is not None and not self.search.supports_constraints:
            raise NotImplementedError(
                "Target-side constraints were provided, but search method doesn't support them"
            )

        # Initialize constraints, when active
        self.search.init_constraints(constraints, beam_size)

        max_len: int = -1
        if self.match_source_len:
            max_len = src_lengths.max().item()
        else:
            max_len = min(
                int(self.max_len_a * src_len + self.max_len_b),
                self.max_len - 1,
            )
        assert (
                self.min_len <= max_len
        ), "min_len cannot be larger than max_len, please adjust these!"
        # compute the encoder output for each beam
        with torch.autograd.profiler.record_function("EnsembleModel: forward_encoder"):
            encoder_outs = self.model.forward_encoder(net_input)

        # placeholder of indices for bsz * beam_size to hold tokens and accumulative scores
        new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
        new_order = new_order.to(src_tokens.device).long()
        encoder_outs = self.model.reorder_encoder_out(encoder_outs, new_order)
        # ensure encoder_outs is a List.
        assert encoder_outs is not None

        # initialize buffers
        scores = (
            torch.zeros(bsz * beam_size, max_len + 1).to(src_tokens).float()
        )  # +1 for eos; pad is never chosen for scoring
        tokens = (
            torch.zeros(bsz * beam_size, max_len + 2)
            .to(src_tokens)
            .long()
            .fill_(self.pad)
        )  # +2 for eos and pad
        tokens[:, 0] = self.eos if bos_token is None else bos_token
        attn: Optional[Tensor] = None

        # A list that indicates candidates that should be ignored.
        # For example, suppose we're sampling and have already finalized 2/5
        # samples. Then cands_to_ignore would mark 2 positions as being ignored,
        # so that we only finalize the remaining 3 samples.
        cands_to_ignore = (
            torch.zeros(bsz, beam_size).to(src_tokens).eq(-1)
        )  # forward and backward-compatible False mask

        # list of completed sentences
        finalized = torch.jit.annotate(
            List[List[Dict[str, Tensor]]],
            [torch.jit.annotate(List[Dict[str, Tensor]], []) for i in range(bsz)],
        )  # contains lists of dictionaries of infomation about the hypothesis being finalized at each step

        # a boolean array indicating if the sentence at the index is finished or not
        finished = [False for i in range(bsz)]
        num_remaining_sent = bsz  # number of sentences remaining

        # number of candidate hypos per step
        cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

        # offset arrays for converting between different indexing schemes
        bbsz_offsets = (
            (torch.arange(0, bsz) * beam_size)
            .unsqueeze(1)
            .type_as(tokens)
            .to(src_tokens.device)
        )
        cand_offsets = torch.arange(0, cand_size).type_as(tokens).to(src_tokens.device)

        reorder_state: Optional[Tensor] = None
        batch_idxs: Optional[Tensor] = None

        original_batch_idxs: Optional[Tensor] = None
        if "id" in sample and isinstance(sample["id"], Tensor):
            original_batch_idxs = sample["id"]
        else:
            original_batch_idxs = torch.arange(0, bsz).type_as(tokens)

        for step in range(max_len + 1):  # one extra step for EOS marker
            # reorder decoder internal states based on the prev choice of beams
            if reorder_state is not None:
                if batch_idxs is not None:
                    # update beam indices to take into account removed sentences
                    corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(
                        batch_idxs
                    )
                    reorder_state.view(-1, beam_size).add_(
                        corr.unsqueeze(-1) * beam_size
                    )
                    original_batch_idxs = original_batch_idxs[batch_idxs]
                self.model.reorder_incremental_state(incremental_states, reorder_state)
                encoder_outs = self.model.reorder_encoder_out(
                    encoder_outs, reorder_state
                )
            with torch.autograd.profiler.record_function(
                    "EnsembleModel: forward_decoder"
            ):
                lprobs, avg_attn_scores, decoder_out = self.model.forward_decoder(
                    tokens[:, : step + 1],
                    encoder_outs,
                    incremental_states,
                    self.temperature,
                    self.metric,
                    return_decoder_output=True,
                )
            if self.knn_index is not None:
                full_knn_scores  = self.get_knn_score(lprobs,decoder_out)
                if full_knn_scores is not None:
                    lprobs = torch.stack([lprobs, full_knn_scores.to(lprobs)], dim=0)
                    coeffs = torch.ones_like(lprobs)
                    coeffs[0] = np.log(1 - self.knn_lambda)
                    coeffs[1] = np.log(self.knn_lambda)
                    lprobs = torch.logsumexp(lprobs + coeffs, dim=0)


            if self.lm_model is not None:
                lm_out = self.lm_model(tokens[:, : step + 1])
                probs = self.lm_model.get_normalized_probs(
                    lm_out, log_probs=True, sample=None
                )
                probs = probs[:, -1, :] * self.lm_weight
                lprobs += probs

            lprobs[lprobs != lprobs] = torch.tensor(-math.inf).to(lprobs)

            lprobs[:, self.pad] = -math.inf  # never select pad
            lprobs[:, self.unk] -= self.unk_penalty  # apply unk penalty

            # handle max length constraint
            if step >= max_len:
                lprobs[:, : self.eos] = -math.inf
                lprobs[:, self.eos + 1:] = -math.inf

            # handle prefix tokens (possibly with different lengths)
            if (
                    prefix_tokens is not None
                    and step < prefix_tokens.size(1)
                    and step < max_len
            ):
                lprobs, tokens, scores = self._prefix_tokens(
                    step, lprobs, scores, tokens, prefix_tokens, beam_size
                )
            else:
                if step < self.min_len:
                    # minimum length constraint (does not apply if using prefix_tokens)
                    lprobs[:, self.eos] = -math.inf

                if self.token_indices_to_suppress is not None:
                    lprobs[:, self.token_indices_to_suppress] = -math.inf

            # Record attention scores, only support avg_attn_scores is a Tensor
            if avg_attn_scores is not None:
                if attn is None:
                    attn = torch.empty(
                        bsz * beam_size, avg_attn_scores.size(1), max_len + 2
                    ).to(scores)
                attn[:, :, step + 1].copy_(avg_attn_scores)

            scores = scores.type_as(lprobs)
            eos_bbsz_idx = torch.empty(0).to(
                tokens
            )  # indices of hypothesis ending with eos (finished sentences)
            eos_scores = torch.empty(0).to(
                scores
            )  # scores of hypothesis ending with eos (finished sentences)

            if self.should_set_src_lengths:
                self.search.set_src_lengths(src_lengths)

            if self.repeat_ngram_blocker is not None:
                lprobs = self.repeat_ngram_blocker(tokens, lprobs, bsz, beam_size, step)

            # Shape: (batch, cand_size)
            cand_scores, cand_indices, cand_beams = self.search.step(
                step,
                lprobs.view(bsz, -1, self.vocab_size),
                scores.view(bsz, beam_size, -1)[:, :, :step],
                tokens[:, : step + 1],
                original_batch_idxs,
            )

            # cand_bbsz_idx contains beam indices for the top candidate
            # hypotheses, with a range of values: [0, bsz*beam_size),
            # and dimensions: [bsz, cand_size]
            cand_bbsz_idx = cand_beams.add(bbsz_offsets)

            # finalize hypotheses that end in eos
            # Shape of eos_mask: (batch size, beam size)
            eos_mask = cand_indices.eq(self.eos) & cand_scores.ne(-math.inf)
            eos_mask[:, :beam_size][cands_to_ignore] = torch.tensor(0).to(eos_mask)

            # only consider eos when it's among the top beam_size indices
            # Now we know what beam item(s) to finish
            # Shape: 1d list of absolute-numbered
            eos_bbsz_idx = torch.masked_select(
                cand_bbsz_idx[:, :beam_size], mask=eos_mask[:, :beam_size]
            )

            finalized_sents: List[int] = []
            if eos_bbsz_idx.numel() > 0:
                eos_scores = torch.masked_select(
                    cand_scores[:, :beam_size], mask=eos_mask[:, :beam_size]
                )

                finalized_sents = self.finalize_hypos(
                    step,
                    eos_bbsz_idx,
                    eos_scores,
                    tokens,
                    scores,
                    finalized,
                    finished,
                    beam_size,
                    attn,
                    src_lengths,
                    max_len,
                )
                num_remaining_sent -= len(finalized_sents)

            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                break
            if self.search.stop_on_max_len and step >= max_len:
                break
            assert step < max_len, f"{step} < {max_len}"

            # Remove finalized sentences (ones for which {beam_size}
            # finished hypotheses have been generated) from the batch.
            if len(finalized_sents) > 0:
                new_bsz = bsz - len(finalized_sents)

                # construct batch_idxs which holds indices of batches to keep for the next pass
                batch_mask = torch.ones(
                    bsz, dtype=torch.bool, device=cand_indices.device
                )
                batch_mask[finalized_sents] = False
                # TODO replace `nonzero(as_tuple=False)` after TorchScript supports it
                batch_idxs = torch.arange(
                    bsz, device=cand_indices.device
                ).masked_select(batch_mask)

                # Choose the subset of the hypothesized constraints that will continue
                self.search.prune_sentences(batch_idxs)

                eos_mask = eos_mask[batch_idxs]
                cand_beams = cand_beams[batch_idxs]
                bbsz_offsets.resize_(new_bsz, 1)
                cand_bbsz_idx = cand_beams.add(bbsz_offsets)
                cand_scores = cand_scores[batch_idxs]
                cand_indices = cand_indices[batch_idxs]

                if prefix_tokens is not None:
                    prefix_tokens = prefix_tokens[batch_idxs]
                src_lengths = src_lengths[batch_idxs]
                cands_to_ignore = cands_to_ignore[batch_idxs]

                scores = scores.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                tokens = tokens.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                if attn is not None:
                    attn = attn.view(bsz, -1)[batch_idxs].view(
                        new_bsz * beam_size, attn.size(1), -1
                    )
                bsz = new_bsz
            else:
                batch_idxs = None

            # Set active_mask so that values > cand_size indicate eos hypos
            # and values < cand_size indicate candidate active hypos.
            # After, the min values per row are the top candidate active hypos

            # Rewrite the operator since the element wise or is not supported in torchscript.

            eos_mask[:, :beam_size] = ~((~cands_to_ignore) & (~eos_mask[:, :beam_size]))
            active_mask = torch.add(
                eos_mask.type_as(cand_offsets) * cand_size,
                cand_offsets[: eos_mask.size(1)],
            )

            # get the top beam_size active hypotheses, which are just
            # the hypos with the smallest values in active_mask.
            # {active_hypos} indicates which {beam_size} hypotheses
            # from the list of {2 * beam_size} candidates were
            # selected. Shapes: (batch size, beam size)
            new_cands_to_ignore, active_hypos = torch.topk(
                active_mask, k=beam_size, dim=1, largest=False
            )

            # update cands_to_ignore to ignore any finalized hypos.
            cands_to_ignore = new_cands_to_ignore.ge(cand_size)[:, :beam_size]
            # Make sure there is at least one active item for each sentence in the batch.
            assert (~cands_to_ignore).any(dim=1).all()

            # update cands_to_ignore to ignore any finalized hypos

            # {active_bbsz_idx} denotes which beam number is continued for each new hypothesis (a beam
            # can be selected more than once).
            active_bbsz_idx = torch.gather(cand_bbsz_idx, dim=1, index=active_hypos)
            active_scores = torch.gather(cand_scores, dim=1, index=active_hypos)

            active_bbsz_idx = active_bbsz_idx.view(-1)
            active_scores = active_scores.view(-1)

            # copy tokens and scores for active hypotheses

            # Set the tokens for each beam (can select the same row more than once)
            tokens[:, : step + 1] = torch.index_select(
                tokens[:, : step + 1], dim=0, index=active_bbsz_idx
            )
            # Select the next token for each of them
            tokens.view(bsz, beam_size, -1)[:, :, step + 1] = torch.gather(
                cand_indices, dim=1, index=active_hypos
            )
            if step > 0:
                scores[:, :step] = torch.index_select(
                    scores[:, :step], dim=0, index=active_bbsz_idx
                )
            scores.view(bsz, beam_size, -1)[:, :, step] = torch.gather(
                cand_scores, dim=1, index=active_hypos
            )

            # Update constraints based on which candidates were selected for the next beam
            self.search.update_constraints(active_hypos)

            # copy attention for active hypotheses
            if attn is not None:
                attn[:, :, : step + 2] = torch.index_select(
                    attn[:, :, : step + 2], dim=0, index=active_bbsz_idx
                )

            # reorder incremental state in decoder
            reorder_state = active_bbsz_idx

        # sort by score descending
        for sent in range(len(finalized)):
            scores = torch.tensor(
                [float(elem["score"].item()) for elem in finalized[sent]]
            )
            _, sorted_scores_indices = torch.sort(scores, descending=True)
            finalized[sent] = [finalized[sent][ssi] for ssi in sorted_scores_indices]
            finalized[sent] = torch.jit.annotate(
                List[Dict[str, Tensor]], finalized[sent]
            )
        return finalized

