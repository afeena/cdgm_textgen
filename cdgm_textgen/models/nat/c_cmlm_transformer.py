# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This file implements:
Ghazvininejad, Marjan, et al.
"Constant-time machine translation with conditional masked language models."
arXiv preprint arXiv:1904.09324 (2019).
"""
import faiss
import faiss.contrib.torch_utils
import omegaconf
import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.models.nat import ensemble_decoder
from fairseq.models.nat.cmlm_transformer import CMLMNATransformerModel
from fairseq.models.nat.nonautoregressive_transformer import NATransformerDecoder
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.utils import new_arange

from cdgm_textgen.dataclass.utils import convert_omegaconf_to_namesapce
from cdgm_textgen.iterative_refinement_generator import DecoderOut
from cdgm_textgen.models import register_model
from cdgm_textgen.models.nat.c_cmlm_transformer_config import CCMLMTransformerConfig


def _skeptical_unmasking_reverse(output_scores, output_masks, p):
    sorted_index = output_scores.sort(dim=-1, descending=True)[1]
    boundary_len = (
            (output_masks.sum(1, keepdim=True).type_as(output_scores) - 2) * p
    ).long()
    skeptical_mask = new_arange(output_masks) < boundary_len
    return skeptical_mask.scatter(1, sorted_index, skeptical_mask)


@register_model("cont_cmlm_transformer", dataclass=CCMLMTransformerConfig)
class CCMLMNATransformerModel(CMLMNATransformerModel):
    @classmethod
    def build_model(cls, args, task):
        if isinstance(args, omegaconf.DictConfig):
            args = convert_omegaconf_to_namesapce(args)
        model = super().build_model(args, task)
        return model

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        # TODO: no last layernorm
        decoder = ContNATransformerDecoder(args, tgt_dict, embed_tokens, )
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)

        decoder.target_embed = cls.build_embedding(args, tgt_dict, args.decoder_output_dim, args.target_embed_path)
        decoder.target_embed.weight.requires_grad = False
        decoder.target_embed.bias = False

        return decoder

    def forward(
            self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, **kwargs
    ):
        assert not self.decoder.src_embedding_copy, "do not support embedding copy."

        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)

        # length prediction
        length_out = self.decoder.forward_length(
            normalize=False, encoder_out=encoder_out
        )
        length_tgt = self.decoder.forward_length_prediction(
            length_out, encoder_out, tgt_tokens
        )

        # decoding
        feat_ins_mask = prev_output_tokens.eq(self.unk)
        feat_ins_out, dist_params = self.decoder(
            normalize=False,
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
        )

        return {
            "word_ins": {
                "out": feat_ins_out,
                "tgt": tgt_tokens,
                "mask": feat_ins_mask,
                "dist_params": dist_params
            },
            "length": {
                "out": length_out,
                "tgt": length_tgt,
                "factor": self.decoder.length_loss_factor
            },
        }

    def get_targets(self, sample, net_output):
        """Get targets from either the sample or the net's output."""
        return self.decoder.target_embed(sample["target"])

    def initialize_output_tokens(self, encoder_out, src_tokens, tgt_tokens):
        # length prediction
        length_tgt = self.decoder.forward_length_prediction(
            self.decoder.forward_length(normalize=True, encoder_out=encoder_out),
            encoder_out=encoder_out,
            tgt_tokens=tgt_tokens
        )

        max_length = length_tgt.clamp_(min=2).max()
        idx_length = utils.new_arange(src_tokens, max_length)

        initial_output_tokens = src_tokens.new_zeros(
            src_tokens.size(0), max_length
        ).fill_(self.pad)
        initial_output_tokens.masked_fill_(
            idx_length[None, :] < length_tgt[:, None], self.unk
        )
        initial_output_tokens[:, 0] = self.bos
        initial_output_tokens.scatter_(1, length_tgt[:, None] - 1, self.eos)

        initial_output_scores = torch.full(
            initial_output_tokens.size(), float('-inf'), device=initial_output_tokens.device
        ).type_as(encoder_out["encoder_out"][0])

        return DecoderOut(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            attn=None,
            step=0,
            max_step=0,
            history=None,
            output_decoder_states=None
        )

    def forward_decoder(self, decoder_out, encoder_out, decoding_format=None,
                        measure="cosine", decoding_masking="distance", approximate_knn=False, **kwargs):
        step = decoder_out.step
        max_step = decoder_out.max_step

        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        history = decoder_out.history

        # execute the decoder
        # print("input step {}".format(decoder_out.step))
        # for tok, sc in zip (output_tokens[0], output_scores[0]):
        #    print("token\t", tok.item(), "\tscore\t", sc.item())
        output_masks = output_tokens.eq(self.unk)
        distances, dist_param = self.decoder(
            normalize=True,
            prev_output_tokens=output_tokens,
            encoder_out=encoder_out,
            measure=measure,
            step=step,
            approximate_knn=approximate_knn
        )

        _scores, _tokens = distances[0], distances[1]

        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        if dist_param is not None and decoding_masking == "kappa":
            kappa = -dist_param["kappa"]
            output_scores.masked_scatter_(output_masks, kappa[output_masks])
        elif decoding_masking == "distance":
            output_scores.masked_scatter_(output_masks, _scores[output_masks])
        else:
            raise NotImplementedError

        if history is not None:
            history.append(output_tokens.clone())

        # skeptical decoding (depend on the maximum decoding steps.)
        if (step + 1) < max_step:
            skeptical_mask = _skeptical_unmasking_reverse(
                output_scores, output_tokens.ne(self.pad), 1 - (step + 1) / max_step
            )

            output_tokens.masked_fill_(skeptical_mask, self.unk)
            output_scores.masked_fill_(skeptical_mask, float('-inf'))

            if history is not None:
                history.append(output_tokens.clone())

        # else:
        #     if self.decoder.kappa == "tau":
        #         kappa = 511 * torch.exp(kappa)
        #     print("kappa: ", " ".join([str(x) for x in kappa[0][0].data.cpu().numpy()[1:-1]]))
        #     print("cosine_scores: ", " ".join([str(x) for x in _scores[0].data.cpu().numpy()[1:-1]]))
        #     print("hyp_bpe_tokens: ": self.decoder.dictionary.string(_tokens[:,1:-1]))

        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attn=None,
            history=history,
        )


class ContNATransformerDecoder(NATransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args,
                         dictionary, embed_tokens,
                         no_encoder_attn=False,
                         )
        self.target_embed = None
        self.nn_search_matrix = None
        self.kappa = args.kappa
        self.faiss_index = None
        if self.kappa == "tau":
            self.tau_fc = torch.nn.Linear(self.embed_dim, 1)

    def setup_search_matrix(self):
        assert self.target_embed is not None
        self.nn_search_matrix = F.normalize(self.target_embed.weight.detach().clone(),
                                            dim=-1
                                            )

    def setup_faiss_index(self):
        res = faiss.StandardGpuResources()
        self.faiss_index = faiss.GpuIndexFlat(res, self.target_embed.shae[-1], faiss.METRIC_INNER_PRODUCT)
        self.faiss_index.add(F.normalize(self.target_embed, dim=-1))

    def forward_kappa(self, outputs):
        if self.kappa == 'norm':
            return {"kappa": torch.norm(outputs, dim=-1)}
        if self.kappa == 'norm2':
            return {"kappa": torch.pow(torch.norm(outputs, dim=-1), 2.0)}
        elif self.kappa == 'tau':
            tau = self.tau_fc(outputs)
            return {"kappa": (outputs.shape[-1] - 1) * torch.exp(tau),
                    "tau": tau}
        else:
            return None

    @ensemble_decoder
    def forward(self, normalize,
                encoder_out,
                prev_output_tokens,
                step=0,
                measure="cosine",
                return_features=False,
                states=None,
                approximate_knn=False,
                **unused
                ):

        features, _ = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            embedding_copy=(step == 0) & self.src_embedding_copy,
            step=step,
            states=states

        )
        dist_param = self.forward_kappa(features)
        if normalize:
            scores = self.__nearest_neighbor_search(features, measure, approximate_knn)
            if return_features:
                return scores, dist_param, features
            else:
                return scores, dist_param
        return features, dist_param

    def __nearest_neighbor_search(self, features, measure, approximate_knn=False):
        # if measure == "l2":
        #     distances = torch.cdist(features, self.target_embed.weight.unsqueeze(0)) ** 2
        #     return distances.min(-1)
        with torch.no_grad():
            if measure == "cosine":
                features = F.normalize(features, p=2.0, dim=-1)
                if not approximate_knn:
                    self.nn_search_matrix = self.nn_search_matrix.to(features.device)
                    distances = 1.0 - features.matmul(self.nn_search_matrix.t())
                    return distances.min(-1)
                else:
                    # faiss works with 2d arrays only, so we need to transform 3d->2d and back
                    sim, _tokens = self.faiss_index.search(torch.flatten(features, start_dim=0, end_dim=1), 1)
                    return (1 - sim).view(features.shape[0], -1), _tokens.view(features.shape[0], -1)
            else:
                raise NotImplementedError
