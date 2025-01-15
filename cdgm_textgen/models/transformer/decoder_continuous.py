import logging
from typing import Any, Dict, List, Optional, Tuple

import geoopt.manifolds.stereographic.math as pmath
import power_spherical
import torch
import torch.nn.functional as F
from torch import Tensor

from cdgm_textgen.models.transformer.decoder_continuous_base import ContinuousTransformerDecoderBase
from cdgm_textgen.models.transformer.transformer_base import Embedding
from cdgm_textgen.modules.utils import stereographic_poincare_dist

logger = logging.getLogger("cdgm_textgen.models.transformer.transformer_decoder_continuous")

from geoopt import ManifoldParameter, SphereExact, StereographicExact, Euclidean, Lorentz, ProductManifold

class ManifoldDecoderContinuous(ContinuousTransformerDecoderBase):
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
                         target_embed=target_embed,
                         )
        self.manifold = None
        if cfg.manifold == "sphere":
            self.manifold = SphereExact()
            self.set_params_to_manifold()
        if cfg.manifold == "poincare":
            self.manifold = StereographicExact(k=-1.0)
            self.set_params_to_manifold()
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
        # project to a manifold
        if isinstance(self.manifold, StereographicExact) or isinstance(self.manifold, Lorentz):
            x = pmath.project(self.manifold.expmap0(x), k=self.manifold.k)
        elif isinstance(self.manifold, SphereExact):
            x = F.normalize(x, dim=-1)
        return x, extra

    def set_params_to_manifold(self):
        if not self.manifold.check_point_on_manifold(self.target_embed.weight):
            self.target_embed.weight = ManifoldParameter(self.target_embed.weight, manifold=self.manifold, requires_grad=False)
            if isinstance(self.manifold, SphereExact) or isinstance(self.manifold, ProductManifold):
                self.target_embed.weight.data /= torch.norm(self.target_embed.weight.data, dim=-1, keepdim=True)
            elif isinstance(self.manifold, StereographicExact) or isinstance(self.manifold, Lorentz):
                self.target_embed.weight.data = pmath.project(self.manifold.expmap0(self.target_embed.weight.data),
                                                              k=self.manifold.k)

    def _check_and_project_on_manifold(self, x):
        if not self.manifold.check_point_on_manifold(x):
            if isinstance(self.manifold, StereographicExact):
                x = pmath.project(self.manifold.expmap0(x), k=self.manifold.k)
            elif isinstance(self.manifold, SphereExact):
                x = x / torch.norm(x, dim=-1, keepdim=True)
        return x

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

        return scores
class ProbaTransformerDecoder(ContinuousTransformerDecoderBase):
    def __init__(self, cfg,
                 dictionary,
                 embed_tokens,
                 target_embed: Optional[Embedding] = None,
                 no_encoder_attn=False,
                 output_projection=torch.nn.Identity()):

        super().__init__(cfg,
                         dictionary,
                         embed_tokens,
                         target_embed,
                         no_encoder_attn=no_encoder_attn,
                         output_projection=output_projection)

        if cfg.kappa == "ff":
            self.kappa_ff = torch.nn.Sequential(
                torch.nn.Linear(self.output_embed_dim, 1),
                torch.nn.Softplus()
            )
        else:
            self.kappa_ff = None

    def forward_kappa(self, features: Tensor) -> Tensor:
        """get kappa for the output PS distribution"""
        if self.kappa_ff is None:
            return torch.norm(features, dim=-1)
        else:
            return self.kappa_ff(features).squeeze()

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
            elif metric == "ps":
                scores = self.get_ps_likelihood_scores(net_output[0])

        return scores


class PseudoContinuousTransformerDecoder(ContinuousTransformerDecoderBase):
    def __init__(self, dictionary,
                 embed_tokens,
                 target_embed: Optional[Embedding] = None,
                 no_encoder_attn=False,
                 output_projection=torch.nn.Identity()):
        super().__init__(dictionary, embed_tokens,
                         target_embed,
                         no_encoder_attn,
                         output_projection)
        self.manifold = None

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
        x, extra = super().forward(prev_output_tokens, encoder_out, incremental_state, features_only,
                                   full_context_alignment,
                                   alignment_layer, alignment_heads, src_lengths, return_all_hiddens)

        # do a projection to a vocab before a softmax
        x = x @ self.target_embed.weight.T

        return x, extra

    def get_normalized_probs(
            self,
            net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
            log_probs: bool,
            sample: Optional[Dict[str, Tensor]] = None,
            metric: Optional[str] = "cosine",
    ):
        # TODO: return log_softmax
        pass