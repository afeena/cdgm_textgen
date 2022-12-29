import logging
from typing import Any, Dict, List, Optional, Tuple

import power_spherical
import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.models.transformer import TransformerDecoderBase
from torch import Tensor
from torch.nn import Identity

from cdgm_textgen.models.transformer.transformer_base import Embedding
from cdgm_textgen.modules.cmk_approx import _neglogcmk

logger = logging.getLogger("cdgm_textgen.models.transformer.transformer_decoder_continuous")


class ContinuousTransformerDecoder(TransformerDecoderBase):
    def __init__(self, cfg,
                 dictionary,
                 embed_tokens,
                 no_encoder_attn=False,
                 output_projection=None,
                 ):
        super().__init__(cfg,
                         dictionary, embed_tokens,
                         no_encoder_attn=no_encoder_attn,
                         output_projection=output_projection,
                         )
        self.output_projection = None
        self.dict_len = len(self.dictionary)
        if cfg.no_last_dec_layer_norm:
            self.layers[-1].final_layer_norm = Identity(self.embed_dim, export=cfg.export)

        # build target (reference) emb
        padding_idx = dictionary.pad()

        with torch.no_grad():
            self.target_embed = Embedding(self.dict_len, self.output_embed_dim, padding_idx)
            if cfg.target_embed_path:
                logger.info(f"loading target embeddings from {cfg.target_embed_path}...")
                embed_dict = utils.parse_embedding(cfg.target_embed_path)
                utils.load_embedding(embed_dict, dictionary, self.target_embed)
            self.target_embed.weight.requires_grad = False
            self.target_embed_normalized = F.normalize(self.target_embed.weight.detach(),
                                                       dim=-1).to(self.target_embed.device)

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


class MixtureContinuousTransformerDecoder(ContinuousTransformerDecoder):
    def __init__(self,
                 cfg,
                 dictionary,
                 embed_tokens
                 ):
        super().__init__(cfg, dictionary, embed_tokens)
        self.num_mixture_components = cfg.num_mixture_components

        # project output to have emb_dim*num_mixture_components
        self.ouput_ff_with_mix_components = torch.nn.Linear(self.output_embed_dim,
                                                            self.output_embed_dim * self.num_mixture_components,
                                                            bias=False)
        # define mixture head to predict the cluster
        self.mixture_head_ff_1 = torch.nn.Linear(self.output_embed_dim,
                                                 self.num_mixture_components, bias=False)

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

        extra["mixtures_out"] = self.forward_mixture(x)

        x = self.ouput_ff_with_mix_components(x)

        return x.view(x.shape[0], -1, self.num_mixture_components, self.output_embed_dim), extra

    def forward_mixture(self, decoder_output):
        return self.mixture_head_ff_1(decoder_output)