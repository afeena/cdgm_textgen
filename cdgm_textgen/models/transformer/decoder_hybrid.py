import logging
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

from cdgm_textgen.models.transformer.decoder_continuous_base import ContinuousTransformerDecoderBase
from cdgm_textgen.models.transformer.transformer_base import Embedding

logger = logging.getLogger("cdgm_textgen.models.transformer.transformer_decoder_hybrid")

from geoopt import SphereExact, ManifoldParameter

class HybridTransformerDecoder(ContinuousTransformerDecoderBase):
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

        #put the target embedding and output projection on the sphere

    def set_params_on_manifold(self, manifold):
        self.target_embed.weight = ManifoldParameter(self.target_embed.weight, manifold=manifold)
        self.target_embed.weight.data /= self.target_embed.weight.norm(dim=-1, keepdim=True)

        self.project_out_dim.weight = ManifoldParameter(self.project_out_dim.weight, manifold=manifold)
        self.project_out_dim.weight.data /= self.project_out_dim.weight.norm(dim=-1, keepdim=True)