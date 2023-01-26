import logging
from typing import Any, Dict, List, Optional

import power_spherical
import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.models.transformer import TransformerDecoderBase
from functorch import vmap, combine_state_for_ensemble
from torch import Tensor

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
        # remove projection to vocab
        self.output_projection = None

        # make sure there is always a projection from hidden state after layer_norm to output_embeddings_dim
        self.project_out_dim = None

        self.ff_output_emb = torch.nn.Linear(self.embed_dim, self.output_embed_dim, bias=False)

        self.dict_len = len(self.dictionary)

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
        # do customize output projection after LayerNorm
        x = self.ff_output_emb(x)

        return x, extra

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

    def sample_output(self, decoder_out, n_samples=1):
        kappa = torch.norm(decoder_out, dim=-1).unsqueeze(1)
        features_idx = F.normalize(decoder_out, p=2.0, dim=-1).unsqueeze(1)

        q_dist = power_spherical.PowerSpherical(features_idx, kappa)
        sample = q_dist.sample(n_samples)
        log_probs_samples = q_dist.log_prob(sample)

        return sample, log_probs_samples


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
        self.ff_output_emb = torch.nn.ModuleList([
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

        fmodel, mixture_params, mixture_buffers = combine_state_for_ensemble(self.ff_output_emb)
        x = vmap(fmodel, (0, 0, None))(mixture_params, mixture_buffers, x)

        return x, extra

    def forward_mixture(self, decoder_output):
        mixtures_out = self.mixture_head_ff_1(decoder_output)
        return mixtures_out

    def get_mixture_output(self, decoder_out, mixtures_out, reduce=None):
        # Get decoder_our in MxBxTxD where T=1
        model_scores = self.get_ps_likelihood_scores(decoder_out).squeeze(dim=2)
        mixture_probs = F.softmax(mixtures_out, -1)
        # for debug purposes
        # from datetime import datetime
        # torch.save(mixture_probs, f"mixture_dist_{datetime.now().timestamp()}")
        _, max_idx = mixture_probs.max(dim=-1)
        mixture_probs = mixture_probs.squeeze(dim=1).unsqueeze(-1)
        model_scores = model_scores * mixture_probs.transpose(0, 1)

        if reduce == "max":
            max_idx = max_idx.t()
            max_idx = max_idx[..., None].expand(-1, -1, model_scores.size(-1))
            out = model_scores.gather(0, max_idx)
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

    def get_ps_likelihood_scores(self, features: Tensor) -> Tensor:
        """get likelihood of target tokens based on output PS distribution"""
        with torch.no_grad():
            mixt = features.shape[0]
            bsz = features.shape[1]
            kappa = torch.norm(features, dim=-1).unsqueeze(1)
            features = F.normalize(features, p=2.0, dim=-1).unsqueeze(1)

            q_dist = power_spherical.PowerSpherical(features, kappa)
            log_probs = q_dist.log_prob(self.target_embed_normalized)

            output_proj = log_probs.view(mixt, bsz, -1, self.dict_len)
            return output_proj

    def sample_mixture_and_output(self, decoder_out, mixtures_out, n_samples=1):
        """
        sample mixture index first, then sample from respective PS distribution
        decoder_out: MxBxTxD
        mixtures_out: BxTxM
        """

        raise NotImplementedError
