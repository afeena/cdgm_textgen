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

    def get_ps_likelihood_scores(self, features: Tensor) -> Tensor:
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

    def get_normalized_probs(
            self,
            net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
            log_probs: bool,
            sample: Optional[Dict[str, Tensor]] = None,
            metric: Optional[str] = None,
    ):
        with torch.no_grad():
            scores = torch.zeros(0)
            if metric == "cosine":
                scores = self.get_cosine_similarity_scores(net_output[0])
            if metric == "ps":
                scores = self.get_ps_likelihood_scores(net_output[0])

        return scores


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

    def get_mixture_output(self, decoder_out, mixtures_out, reduce=None):
        model_scores = self.get_ps_likelihood_scores(decoder_out)
        mixture_probs = F.softmax(mixtures_out, -1)
        #for debug purposes
        #from datetime import datetime
        #torch.save(mixture_probs, f"mixture_dist_{datetime.now().timestamp()}")
        if reduce is None:
            return model_scores

        if reduce == "max":
            _, max_idx = mixture_probs.max(dim=-1)
            idx = max_idx[..., None].expand(-1, -1, model_scores.shape[-1])
            out = torch.gather(model_scores, dim=1, index=idx).squeeze(-1)
            # for debug purposes
            # from datetime import datetime
            # torch.save(dist, f"mixture_dist_{datetime.now().timestamp()}")
            del idx
            return out
        
        if reduce == "sum":
            mixture_probs = mixture_probs.squeeze(dim=1)[..., None].expand(-1, -1, model_scores.shape[-1])
            out = mixture_probs * model_scores
            out = torch.sum(out, dim=1)

            return out

        if reduce == "sample":
            mixture_probs = mixture_probs.squeeze(dim=1)
            idx = torch.multinomial(mixture_probs, 1)
            idx = idx[..., None].expand(-1, -1, model_scores.shape[-1])
            out = torch.gather(model_scores, dim=1, index=idx).squeeze(-1)
            del idx
            return out
    def sample_mixture_and_output(self, decoder_out, mixtures_out, n_samples=1):
        """
        sample mixture index first, then sample from respective PS distribution
        """
        with torch.no_grad():
            bsz = decoder_out.shape[0]
            features = decoder_out.view(-1, self.output_embed_dim)
            mixture_probs = F.softmax(mixtures_out, -1)
            idx = torch.multinomial(mixture_probs, 1)
            features_idx = torch.gather(features, dim=1, index=idx).squeeze(-1)

            kappa = torch.norm(features_idx, dim=-1).unsqueeze(1)
            features_idx = F.normalize(features_idx, p=2.0, dim=-1).unsqueeze(1)

            q_dist = power_spherical.PowerSpherical(features_idx, kappa)
            sample = q_dist.sample()

            return sample, idx


    #legacy. used only with fairseq generator
    def get_normalized_probs(
            self,
            net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
            log_probs: bool,
            sample: Optional[Dict[str, Tensor]] = None,
            metric: Optional[str] = None,
    ):
        with torch.no_grad():
            scores = torch.zeros(0)
            if metric == "cosine":
                scores = self.get_cosine_similarity_scores(net_output[0])
            if metric == "ps":
                scores = self.get_ps_likelihood_scores(net_output[0])
            mixture_probs = F.softmax(net_output[1]["mixtures_out"], dim=-1)
            _, max_idx = mixture_probs.max(dim=-1)
            idx = max_idx[..., None, None].expand(-1, -1, 1, scores.shape[-1])
            out = torch.gather(scores, dim=2, index=idx).squeeze(dim=2)
            # for debug purposes
            # from datetime import datetime
            # torch.save(dist, f"mixture_dist_{datetime.now().timestamp()}")

            del idx
            return out
