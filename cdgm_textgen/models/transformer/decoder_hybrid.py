import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from cdgm_textgen.models.transformer.decoder_continuous import ContinuousTransformerDecoder
from cdgm_textgen.models.transformer.transformer_base import Embedding

logger = logging.getLogger("cdgm_textgen.models.transformer.transformer_decoder_hybrid")


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
                         target_embed=target_embed,
                         no_encoder_attn=no_encoder_attn,
                         output_projection=output_projection,
                         )
        with torch.no_grad():
            self.target_embed.weight.data = F.normalize(self.target_embed.weight.data, dim=-1)
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
