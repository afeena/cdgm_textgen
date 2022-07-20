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
import numpy as np
import torch

from fairseq.models import register_model, register_model_architecture
from fairseq.utils import new_arange

from fairseq.models.nat import CCMLMNATransformerModel

@register_model("iterative_cont_cmlm_transformer")
class IterativeCCMLMNATransformerModel(CCMLMNATransformerModel):

    @staticmethod
    def add_args(parser):
        CCMLMNATransformerModel.add_args(parser)
        parser.add_argument(
            "--train-step",
            type=int,
            help="number of refinement iterations during training",
        )


    @classmethod
    def build_model(cls, args, task):
        model = super().build_model(args, task)
        model.train_step = getattr(args, "train_step", 1)

        return model

    def get_targets(self, sample, net_output):
        return net_output["word_ins"]["tgt"]

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
        feat_ins_out, kappa = self.decoder(
                normalize=False,
                prev_output_tokens=prev_output_tokens,
                encoder_out=encoder_out,
                step=0
            )

        train_step = np.random.randint(low=0, high=self.train_step)
        word_ins_outs = [feat_ins_out]
        tgt_emb = self.decoder.target_embed(tgt_tokens)
        word_ins_tgts = [tgt_emb]
        word_ins_masks = [feat_ins_mask]
        kappas=[kappa[0]]

        for t in range(1, train_step+1):
            word_ins_tgts.append(tgt_emb)
            feat_ins_out, kappa = self.decoder(
                normalize=False,
                prev_output_tokens=prev_output_tokens,
                encoder_out=encoder_out,
                step=t,
                states=feat_ins_out
            )
            kappas.append(kappa[0])

            mask_rate = 1 - (t / train_step)
            num_replaced = torch.round(feat_ins_mask.sum(1).float() * mask_rate)
            scores = feat_ins_mask.float().uniform_()
            scores[~feat_ins_mask] = 0
            _, target_rank = scores.sort(1, descending=True)
            unk_emb = self.decoder.target_embed(torch.tensor(self.decoder.dictionary.unk(),
                                                             device=prev_output_tokens.device))
            target_cutoff = new_arange(target_rank) < num_replaced[:, None].long()

            iter_mask = target_cutoff.scatter(1, target_rank, target_cutoff)
            feat_ins_out[iter_mask,:] = unk_emb
            word_ins_masks.append(iter_mask)
            word_ins_outs.append(feat_ins_out)


        if kappas[0] is not None:
            kappas = torch.cat(kappas, 0)
        else:
            kappas = None

        return {
            "word_ins": {
                "out": torch.cat(word_ins_outs,0),
                "tgt": torch.cat(word_ins_tgts,0),
                "mask": torch.cat(word_ins_masks,0),
                "nll_loss": False,
                "kappa": kappas
            },
            "length": {
                "out": length_out,
                "tgt": length_tgt,
                "factor": self.decoder.length_loss_factor
            },
        }
    def forward_decoder(self, decoder_out, encoder_out, decoding_format=None, measure="cosine", **kwargs):
        step = decoder_out.step
        max_step = decoder_out.max_step

        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        output_decoder_states = decoder_out.output_decoder_states
        history = decoder_out.history

        # execute the decoder
        # print("input step {}".format(decoder_out.step))
        # for tok, sc in zip (output_tokens[0], output_scores[0]):
        #    print("token\t", tok.item(), "\tscore\t", sc.item())
        output_masks = output_tokens.eq(self.unk)
        dec_out = self.decoder(
            normalize=True,
            prev_output_tokens=output_tokens,
            encoder_out=encoder_out,
            measure=measure,
            step=step,
            states=output_decoder_states,
            return_features=True
        )
        _scores, _tokens = dec_out[0].min(-1)
        output_decoder_states = dec_out[-1]

        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])

        if history is not None:
            history.append(output_tokens.clone())

        # skeptical decoding (depend on the maximum decoding steps.)
        if (step + 1) < max_step:
            _, target_rank = _scores.sort(1, descending=True)
            unk_emb = self.decoder.target_embed(torch.tensor(self.decoder.dictionary.unk(),
                                                             device=output_tokens.device))
            boundary_len = (
                    (output_masks.sum(1, keepdim=True).type_as(output_scores) - 2) * (1 - (step + 1) / max_step)
            ).long()
            skeptical_mask = new_arange(output_masks) < boundary_len

            output_decoder_states[skeptical_mask, :] = unk_emb


        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attn=None,
            history=history,
            output_decoder_states=output_decoder_states
        )

@register_model_architecture("iterative_cont_cmlm_transformer", "iterative_cont_cmlm_transformer")
def iterative_cont_cmlm_base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.apply_bert_init = getattr(args, "apply_bert_init", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    # --- special arguments ---
    args.sg_length_pred = getattr(args, "sg_length_pred", False)
    args.pred_length_offset = getattr(args, "pred_length_offset", False)
    args.length_loss_factor = getattr(args, "length_loss_factor", 0.1)
    args.ngram_predictor = getattr(args, "ngram_predictor", 1)
    args.src_embedding_copy = getattr(args, "src_embedding_copy", False)
    args.train_step = getattr(args, "train_step", 1)
    args.kappa = getattr(args, "kappa", None)
    args.no_last_dec_layer_norm = getattr(args, "no_last_dec_layer_norm", True)
