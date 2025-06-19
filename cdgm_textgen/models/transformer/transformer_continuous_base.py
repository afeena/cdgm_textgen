import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List

import torch
from fairseq.models.transformer import TransformerConfig
from torch import Tensor
import torch.nn.functional as F

from cdgm_textgen.models import register_model
from cdgm_textgen.models.transformer.decoder_continuous_base import \
    ContinuousTransformerDecoderBase
from cdgm_textgen.models.transformer.transformer_base import TransformerModelBase

logger = logging.getLogger("cdgm_textgen.models.transformer.transformer_continuous_base")


@dataclass
class ContinuousTransformerConfig(TransformerConfig):
    target_embed_path: Optional[str] = field(
        default=None,
        metadata={"help": "path to target embeddings"},
    )
    no_last_dec_layer_norm: bool = field(
        default=True,
        metadata={"help": "don't add an layernorm after the last decoder block"},
    )
    freeze_decoder_embed: bool = field(default=True)


@register_model("continuous_transformer", dataclass=ContinuousTransformerConfig)
class TransformerModelContinuous(TransformerModelBase):
    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):
        if cfg.target_embed_path is not None:
            logger.info(f"loading target embeddings from {cfg.target_embed_path}...")
            target_embed = cls.build_embedding(
                cfg, tgt_dict, cfg.decoder.output_dim, cfg.target_embed_path
            )
        else:
            target_embed = cls.build_embedding(
                cfg, tgt_dict, cfg.decoder.output_dim,
            )
            target_embed.weight.data = F.normalize(torch.randn(len(tgt_dict),cfg.decoder.output_dim), dim=-1)

        assert target_embed is not None
        decoder = ContinuousTransformerDecoderBase(
            cfg,
            tgt_dict,
            embed_tokens,
            target_embed
        )
        if cfg.freeze_decoder_embed:
            logger.info("freezing decoder target embeddings")
            decoder.target_embed.weight.requires_grad = False
        else:
            logger.info("not freezing decoder target embeddings")
            decoder.target_embed.weight.requires_grad = True
        return decoder

    def forward(
            self,
            src_tokens,
            src_lengths,
            prev_output_tokens,
            return_all_hiddens: bool = True,
            features_only: bool = True,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=True,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        return decoder_out

    def set_normalized_tgt_embeddings(self):
        self.decoder.set_normalized_tgt_embeddings()

    def load_state_dict(
            self,
            state_dict,
            strict=False,
            model_cfg=None,
    ):
        """Copies parameters and buffers from *state_dict* into this module and
        its descendants.

        Overrides the method in :class:`nn.Module`. Compared with that method
        this additionally "upgrades" *state_dicts* from old checkpoints.
        """
        model_dict = self.state_dict()
        # filter out keys
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        self.upgrade_state_dict(state_dict)
        # here allow partial loading from other models
        return super().load_state_dict(state_dict, strict=False)

    def get_targets(self, sample, net_output, **kwargs):
        return self.decoder.target_embed(sample["target"]).detach()

    def forward_decoder_scores(self, decoder_out: Tuple[Tensor, Dict], metric: str) -> Tensor:
        assert metric is not None
        scores = None
        model_out = decoder_out[0]

        if metric == "cosine":
            scores = self.decoder.get_cosine_similarity_scores(model_out)

        return scores

    # legacy
    @torch.no_grad()
    def get_normalized_probs(
            self,
            net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
            log_probs: bool,
            sample: Optional[Dict[str, Tensor]] = None,
            metric: Optional[str] = "cosine",
    ):
        return self.decoder.get_normalized_probs(net_output, log_probs, sample, metric)
