from fairseq.models.transformer import TransformerConfig
from fairseq.modules import BaseLayer
from hypll import nn as hnn
from hypll.manifolds.poincare_ball import Curvature, PoincareBall
from hypll.tensors import TangentTensor
from typing import Dict, List, Optional

from cdgm_textgen.models import register_model
from transformer_decoder import TransformerDecoder
from transformer_base import TransformerModelBase

@register_model("transformer_hypll", dataclass=TransformerConfig)
class HypTransformerModel(TransformerModelBase):
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):
        return HypTransformerDecoder(cfg, tgt_dict, embed_tokens)


class HypTransformerDecoder(TransformerDecoder):
    def __init__(self, cfg,
                 dictionary,
                 embed_tokens,
                 no_encoder_attn=False,
                 output_projection=None):

        super().__init__(cfg,
                         dictionary,
                         embed_tokens,
                         no_encoder_attn=no_encoder_attn,
                         output_projection=output_projection)

        self.manifold = PoincareBall(c=Curvature(requires_grad=True))

    def build_output_projection(self, cfg, dictionary, embed_tokens):
        if self.share_input_output_embed:
            self.output_projection = hnn.HLinear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=False,
                manifold=self.manifold
            )
            self.output_projection.weight = self.embed_tokens.weight
        else:
            self.output_projection = hnn.HLinear(
                self.output_embed_dim, len(dictionary), bias=False, manifold=self.manifold,
            )

        num_base_layers = cfg.base_layers
        for i in range(num_base_layers):
            self.layers.insert(
                ((i + 1) * cfg.decoder.layers) // (num_base_layers + 1),
                BaseLayer(cfg),
            )
    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
    ):
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )

        tangents = TangentTensor(data=x, man_dim=1, manifold=self.manifold)
        manifold_inputs = self.manifold.expmap(tangents)
        x = self.output_layer(x)

        return x, extra