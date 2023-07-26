import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from cdgm_textgen.models.transformer.decoder_continuous import ContinuousTransformerDecoder
from cdgm_textgen.models.transformer.transformer_base import Embedding

logger = logging.getLogger("cdgm_textgen.models.transformer.transformer_decoder_hybrid")


class HybridTransformerDecoder(ContinuousTransformerDecoder):
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
        w_norm = torch.norm(self.target_embed.weight, p=2, dim=-1).detach()
        x_norm = torch.norm(x, p=2, dim=-1).detach()
        x_projected = x.div(x_norm[...,None].expand_as(x)) @ ((self.target_embed.weight).div(w_norm[...,None].expand_as(self.target_embed.weight))).T
        extra["projected"] = x_projected
        return x, extra
