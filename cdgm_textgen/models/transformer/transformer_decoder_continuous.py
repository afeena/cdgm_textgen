import logging
from typing import Any, Dict, List, Optional, Tuple

import power_spherical
import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.models.transformer import TransformerDecoderBase
from functorch import vmap, combine_state_for_ensemble
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

    def sample_output(self, decoder_out, n_samples=1):
        kappa = torch.norm(decoder_out, dim=-1).unsqueeze(1)
        features_idx = F.normalize(decoder_out, p=2.0, dim=-1).unsqueeze(1)

        q_dist = power_spherical.PowerSpherical(features_idx, kappa)
        sample = q_dist.sample(n_samples)
        log_probs_samples = q_dist.log_prob(sample)

        return sample, log_probs_samples

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
        self.project_out_dim = None

        # project output to have emb_dim*num_mixture_components
        self.output_ff_with_mix_components = torch.nn.ModuleList([
            torch.nn.Linear(self.embed_dim, self.output_embed_dim, bias=False)
            for _ in range(self.num_mixture_components)
        ])

        # define mixture head to predict the cluster
        self.mixture_head_ff_1 = torch.nn.Linear(self.embed_dim,
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

        # map the output to the mixture components

        fmodel, mixture_params, mixture_buffers = combine_state_for_ensemble(self.output_ff_with_mix_components)

        x = vmap(fmodel, (0, 0, None))(mixture_params, mixture_buffers, x)

        return x, extra

    def forward_mixture(self, decoder_output):
        mixtures_out = self.mixture_head_ff_1(decoder_output)
        return mixtures_out

    def get_mixture_output(self, decoder_out, mixtures_out, reduce=None):
        #Get decoder_our in MxBxTxD where T=1
        model_scores = self.get_ps_likelihood_scores(decoder_out).squeeze()
        mixture_probs = F.softmax(mixtures_out, -1)
        # for debug purposes
        # from datetime import datetime
        # torch.save(mixture_probs, f"mixture_dist_{datetime.now().timestamp()}")
        mixture_probs = mixture_probs.squeeze(dim=1).unsqueeze(-1)
        model_scores = model_scores * mixture_probs

        if reduce == "max":
            _, max_idx = mixture_probs.max(dim=1)
            out = model_scores[max_idx,:]
            # for debug purposes
            # from datetime import datetime
            # torch.save(dist, f"mixture_dist_{datetime.now().timestamp()}")
            del max_idx
            return out

        elif reduce == "sum":
            out = torch.sum(model_scores, dim=0)
            return out
        else:
            return model_scores

    def sample_mixture_and_output(self, decoder_out, mixtures_out, n_samples=1):
        """
        sample mixture index first, then sample from respective PS distribution
        """
        with torch.no_grad():
            bsz = decoder_out.shape[0]
            features = decoder_out.view(-1, self.output_embed_dim)
            mixture_probs = F.softmax(mixtures_out, -1)
            idx = torch.multinomial(mixture_probs, n_samples, replacement=True)
            features_idx = torch.gather(features, dim=1, index=idx).squeeze(-1)

            kappa = torch.norm(features_idx, dim=-1).unsqueeze(1)
            features_idx = F.normalize(features_idx, p=2.0, dim=-1).unsqueeze(1)

            q_dist = power_spherical.PowerSpherical(features_idx, kappa)
            sample = q_dist.sample(n_samples)
            log_probs_samples = q_dist.log_prob(sample)

            return sample, log_probs_samples

    # legacy. used only with fairseq generator
    def get_normalized_probs(
            self,
            net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
            log_probs: bool,
            sample: Optional[Dict[str, Tensor]] = None,
            metric: Optional[str] = None,
    ):
        with torch.no_grad():
            scores = self.get_ps_likelihood_scores(net_output[0]).squeeze()
            mixture_probs = F.softmax(net_output[1]["mixtures_out"], dim=-1)
            _, max_idx = mixture_probs.max(dim=-1)
            out = scores[max_idx,:]
            # for debug purposes
            # from datetime import datetime
            # torch.save(dist, f"mixture_dist_{datetime.now().timestamp()}")

            return out
