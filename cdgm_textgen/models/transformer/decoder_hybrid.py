import logging
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

from cdgm_textgen.models.transformer.decoder_continuous import ContinuousTransformerDecoder
from cdgm_textgen.models.transformer.transformer_base import Embedding

logger = logging.getLogger("cdgm_textgen.models.transformer.transformer_decoder_hybrid")

from geoopt import SphereExact, ManifoldParameter

class HybridTransformerDecoder(ContinuousTransformerDecoder):
    def __init__(self, cfg,
                 dictionary,
                 embed_tokens,
                 target_embed: Optional[Embedding] = None,
                 no_encoder_attn=False,
                 output_projection=torch.nn.Identity(),
                 ):
        super().__init__(cfg,
                         dictionary, embed_tokens,
                         no_encoder_attn=no_encoder_attn,
                         output_projection=output_projection,
                         )



        # make sure there is always a projection from hidden state after layer_norm to output_embeddings_dim
        if self.project_out_dim is None:
            self.project_out_dim = torch.nn.Linear(self.embed_dim, self.output_embed_dim, bias=False)
        self.target_embed = target_embed

        sphere = SphereExact()

        self.target_embed.weight = ManifoldParameter(self.target_embed.weight, manifold=sphere)
        self.target_embed.weight.data /= self.target_embed.weight.norm(dim=-1, keepdim=True)

        for pn, p in self.named_parameters():
            if pn.endswith("target_embed") or "target_embed" in pn:
                p.param_group = "target_embed"

    def forward(
            self,
            prev_output_tokens,
            encoder_out: Optional[Dict[str, List[Tensor]]] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            features_only: bool = True,
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

        return x, extra
