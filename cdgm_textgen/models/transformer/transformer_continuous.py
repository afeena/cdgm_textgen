import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import torch
from torch import Tensor

from fairseq.models.transformer import TransformerConfig

from cdgm_textgen.models import register_model
from cdgm_textgen.models.transformer.transformer_decoder_continuous import (
    ContinuousTransformerDecoder,
    MixtureContinuousTransformerDecoder,
    ClusteredTransformerDecoder
)
from cdgm_textgen.models.transformer.transformer_base import (
    TransformerModelBase
)

logger = logging.getLogger("cdgm_textgen.models.transformer.transfore_continuous")


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


@register_model("continuous_transformer", dataclass=ContinuousTransformerConfig)
class TransformerModelContinuous(TransformerModelBase):

    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):
        decoder = ContinuousTransformerDecoder(
            cfg,
            tgt_dict,
            embed_tokens,
        )
        return decoder

    def build_index(self, metric=None):
        self.decoder.build_index(metric)

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

    def get_targets(self, sample, net_output):
        return self.decoder.target_embed(sample["target"])

    def get_normalized_probs(
            self,
            net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
            log_probs: bool,
            sample: Optional[Dict[str, Tensor]] = None,
            metric: Optional[str] = None,
    ):
        return self.decoder.get_normalized_probs(net_output, log_probs, sample, metric)


@dataclass
class MixtureContinuousTransformerConfig(ContinuousTransformerConfig):
    num_mixture_components: int = field(
        default=2,
        metadata={"help": "number of mixture components for the distribution"},
    )
    mixture_dropout: float = field(
        default=0,
        metadata={"help": "dropout for the mixture head"},
    )


@register_model("continuous_transformer_mixture", dataclass=MixtureContinuousTransformerConfig)
class TransformerMixtureModelContinuous(TransformerModelContinuous):

    def __init__(self, cfg, encoder, decoder):
        super().__init__(cfg, encoder, decoder)
        if cfg.num_mixture_components == 1:
            logger.error("Running continuous_transformer_mixture with 1 mixture component is not recommended! "
                         "Use continuous_transformer instead.")
            raise ValueError("num_mixture_components must be >1 for continuous_transformer_mixture")

    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):
        decoder = MixtureContinuousTransformerDecoder(
            cfg,
            tgt_dict,
            embed_tokens
        )
        return decoder



@dataclass
class TransformerClusteredConfig(MixtureContinuousTransformerConfig):
    # TODO consider refactoring
    init_components_file: Optional[str] = field(
        default=None,
        metadata={"help": "path to .pt file containing cluster center tensor"}
    )


@register_model("transformer_clustered",
                dataclass=TransformerClusteredConfig)
class TransformerClustered(TransformerModelContinuous):
    """Continuous mixture model with word embedding k-means centroid init."""

    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):

        decoder = ClusteredTransformerDecoder(
            cfg,
            tgt_dict,
            embed_tokens
        )

        # load here from file
        if cfg.init_components_file:
            decoder.centroids[:] = torch.load(cfg.init_components_file)

        return decoder
