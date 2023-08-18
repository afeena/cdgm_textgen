import logging
from typing import Any, Dict, List, Optional, Tuple

import power_spherical
import torch
import torch.nn.functional as F

from fairseq.models.transformer import TransformerDecoderBase
from torch import Tensor

from cdgm_textgen.models.transformer.transformer_base import Embedding
from cdgm_textgen.modules.cmk_approx import _neglogcmk

logger = logging.getLogger("cdgm_textgen.models.transformer.transformer_decoder_continuous")

from geoopt import ManifoldParameter, ManifoldTensor

class ContinuousTransformerDecoder(TransformerDecoderBase):
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

    def set_target_to_manifold(self, manifold):
        self.target_embed.weight = ManifoldParameter(self.target_embed.weight, manifold=manifold)
        self.target_embed.weight.data /= self.target_embed.weight.norm(dim=-1, keepdim=True)

    def set_normalized_tgt_embeddings(self):
        self.target_embed_normalized = F.normalize(self.target_embed.weight.detach(),
                                                   dim=-1).cuda()

    def get_ps_likelihood_scores(self, features: Tensor) -> Tensor:
        """get likelihood of target tokens based on output PS distribution"""
        with torch.no_grad():
            bsz = features.shape[0]
            features = features.view(-1, self.output_embed_dim)
            kappa = torch.norm(features, dim=-1).unsqueeze(1)
            features = F.normalize(features, p=2.0, dim=-1).unsqueeze(1)

            q_dist = power_spherical.PowerSpherical(features, kappa)
            log_probs = q_dist.log_prob(self.target_embed_normalized)

            output_proj = log_probs.view(bsz, -1, self.dict_len)
            return output_proj

    def get_cosine_similarity_scores(self, features: Tensor) -> Tensor:
        with torch.no_grad():
            kappa = torch.tensor([1.], device=features.device)
            features = F.normalize(features, p=2.0, dim=-1)
            m = torch.tensor([self.target_embed_normalized.shape[-1]], device=features.device)
            norm_const = _neglogcmk(m, kappa)

            output_proj = features @ self.target_embed_normalized.T
            if norm_const is not None:
                output_proj += norm_const.squeeze()

            return output_proj

    def get_arc_distance(self, features: Tensor) -> Tensor:
        with torch.no_grad():
            kappa = torch.tensor([1.], device=features.device)
            features = F.normalize(features, p=2.0, dim=-1)
            m = torch.tensor([self.target_embed_normalized.shape[-1]], device=features.device)
            norm_const = _neglogcmk(m, kappa)

            arc_distance_mat = (torch.arccos(features @ self.target_embed_normalized.T) / torch.pi)
            arc_distance_mat = torch.nan_to_num(arc_distance_mat)
            if norm_const is not None:
                arc_distance_mat += norm_const.squeeze()

            return arc_distance_mat

    # legacy. compatible with fairseq translation task
    def get_normalized_probs(
            self,
            net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
            log_probs: bool,
            sample: Optional[Dict[str, Tensor]] = None,
            metric: Optional[str] = "cosine",
    ):
        with torch.no_grad():
            scores = None
            if metric == "cosine":
                scores = self.get_cosine_similarity_scores(net_output[0])
            if metric == "ps":
                scores = self.get_ps_likelihood_scores(net_output[0])
            if metric == "arc":
                scores = self.get_arc_distance(net_output[0])

        return scores
