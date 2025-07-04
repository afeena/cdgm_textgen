import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List

import power_spherical
from torch import Tensor
import torch
import torch.nn.functional as F

from cdgm_textgen.models import register_model
from cdgm_textgen.models.transformer.decoder_continuous_base import ContinuousTransformerDecoderBase
from cdgm_textgen.models.transformer.transformer_base import Embedding
from cdgm_textgen.models.transformer.transformer_continuous_base import TransformerModelContinuous, ContinuousTransformerConfig

logger = logging.getLogger("cdgm_textgen.models.transformer.proba_trafo")

@dataclass
class ProbaTransformerConfig(ContinuousTransformerConfig):
    kappa: str = field(default="norm", metadata={"help": "kappa to use"})

@register_model("proba_transformer", dataclass=ProbaTransformerConfig)
class TransformerModelProba(TransformerModelContinuous):

    def get_kappa(self, out, target, kappa_type) -> Tensor:
        if kappa_type == "out":
            return self.decoder.get_kappa(out, "norm")
        elif kappa_type == "target":
            return self.decoder.get_kappa(target, "inv_freq")

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
            target_embed.weight.data = F.normalize(torch.randn(len(tgt_dict), cfg.decoder.output_dim), dim=-1)

        assert target_embed is not None
        decoder = ProbaTransformerDecoder(
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

    def forward_decoder_scores(self, decoder_out: Tuple[Tensor, Dict], metric: str) -> Tensor:
        assert metric is not None
        scores = None
        model_out = decoder_out[0]

        scores = self.decoder.get_ps_likelihood_scores(model_out)
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

    def get_kappa(self, features: Tensor, kappa_type: str = "norm") -> Tensor:
        """get kappa value for the features"""
        if kappa_type not in ["norm", "freq"]:
            raise ValueError(f"Unknown kappa type: {kappa_type}. Use 'norm' or 'freq'.")

        if kappa_type == "norm":
            return torch.norm(features, dim=-1)
        else:
            return 1.0 / self.self.decoder.dictionary.get_counts(features).float().clamp(min=1.0)

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
