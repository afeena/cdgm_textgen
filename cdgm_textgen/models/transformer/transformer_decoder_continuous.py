import gc
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
        self.index = None
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
                                                       dim=-1).cuda()

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

    def get_normalized_probs(
            self,
            net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
            log_probs: bool,
            sample: Optional[Dict[str, Tensor]] = None,
            metric: Optional[str] = None,
    ):
        with torch.no_grad():
            scores = self.nearest_neighbor_search(net_output[0], metric=metric)

        return scores


    def nearest_neighbor_search(self, features, metric=None):
        with torch.no_grad():
            if metric in ["cosine", "l2"]:
                norm_const = None
                if metric == "cosine":
                    kappa = torch.tensor([1.], device=features.device)
                    features = F.normalize(features, p=2.0, dim=-1)
                    m = torch.tensor([self.target_embed.weight.shape[-1]], device=features.device)
                    norm_const = _neglogcmk(m, kappa)

                output_proj = features @ self.target_embed_normalized.T
                if norm_const is not None:
                    output_proj += norm_const.squeeze()

            if metric == "ps":
                bsz = features.shape[0]
                features = features.view(-1, self.output_embed_dim)
                kappa = torch.norm(features, dim=-1).unsqueeze(1)
                features = F.normalize(features, p=2.0, dim=-1).unsqueeze(1)

                q_dist = power_spherical.PowerSpherical(features, kappa)
                log_probs = q_dist.log_prob(self.target_embed_normalized)

                output_proj = log_probs.view(bsz, -1, self.dict_len)

        return output_proj


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

    @torch.jit.export
    def get_normalized_probs(
            self,
            net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
            log_probs: bool,
            sample: Optional[Dict[str, Tensor]] = None,
            metric: Optional[str] = None,
    ):
        with torch.no_grad():
            out = net_output[0]
            mixtures_out = net_output[1]["mixtures_out"]

            if self.num_mixture_components > 1:
                dist = F.softmax(mixtures_out, -1)
                _, idx = dist.max(dim=-1)

                # for debug purposes
                # from datetime import datetime
                # torch.save(dist, f"mixture_dist_{datetime.now().timestamp()}")

                idx = idx[..., None, None].expand(-1, -1, 1, out.size(3))
                out = torch.gather(out, dim=2, index=idx).squeeze(dim=2)

                del idx

            scores = self.nearest_neighbor_search(out, metric=metric)

            return scores
