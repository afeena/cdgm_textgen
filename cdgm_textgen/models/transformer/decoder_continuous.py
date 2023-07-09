import logging
from typing import Any, Dict, List, Optional, Tuple

import power_spherical
import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.models.transformer import TransformerDecoderBase
from torch import Tensor, nn
from torch.nn.parameter import Parameter

from cdgm_textgen.models.transformer.transformer_base import Embedding
from cdgm_textgen.modules import log_prob_scriptable
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
                         output_projection=None,
                         )

        # make sure there is always a projection from hidden state after layer_norm to output_embeddings_dim
        self.project_out_dim = torch.nn.Linear(self.embed_dim, self.output_embed_dim, bias=False)

        self.dict_len = len(self.dictionary)

        # build target (reference) emb
        padding_idx = dictionary.pad()
        if cfg.target_embd == 'hypercube':
            self.target_embed = HyperCubeEmbed(self.dict_len, self.output_embed_dim, pad_idx=padding_idx,
                                               permute=cfg.target_embd_random_cube)
            self.target_embed.requires_grad_(False)

        else:
            assert cfg.target_embed_path is not None
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

        return scores

    def sample_mbr_string(self, decoder_out: Tensor):
        kappa = torch.norm(decoder_out, dim=-1)
        features_idx = F.normalize(decoder_out, p=2.0, dim=-1)

        q_dist = power_spherical.PowerSpherical(features_idx, kappa)
        samples = q_dist.sample()
        log_probs_samples = log_prob_scriptable(q_dist.loc, q_dist.scale, samples, "jk, ljk -> lj")

        # rounding to a target vocab token
        token_nn_scores: Tensor = samples.squeeze(0) @ self.target_embed_normalized.T
        token_nn_scores[:, self.dictionary.pad()] = float("-inf")

        token_idxs = token_nn_scores.max(-1).indices

        return log_probs_samples, token_idxs
    def sample_mbr_token(self, decoder_out: Tensor, n_samples: int = 10) -> Tuple[Tensor, Tensor]:
        """
        main idea: sample TOKENS and calculate utility functions based on TOKENS. So the function return
        the best token according utility. Then treat it as a greedy search.
        """
        # remove token dim but keep batch dim
        bsz = decoder_out.shape[0]
        kappa = torch.norm(decoder_out, dim=-1).squeeze(1)
        features_idx = F.normalize(decoder_out, p=2.0, dim=-1).squeeze(1)

        q_dist = power_spherical.PowerSpherical(features_idx, kappa)
        samples = q_dist.sample((n_samples,))
        log_probs_samples = log_prob_scriptable(q_dist.loc, q_dist.scale, samples, "jk, sjk -> sj")

        dot_prods = samples.movedim(0,1) @ samples.movedim(2,1)

        dot_prods = dot_prods.view(bsz, n_samples, n_samples)
        rbf_utility = (((dot_prods - 1) / (.1 ** 2)) + log_probs_samples.T.unsqueeze(-1)).sum(dim=-1)

        #sum log probabilities
        #choose the best among dim=-1
        best_utility_hyp_id = rbf_utility.max(dim=-1).indices
        # rounding to a target vocab token
        token_nn_scores: Tensor = (samples[best_utility_hyp_id] @ self.target_embed_normalized.T).squeeze(0)
        token_nn_scores[:, self.dictionary.pad()] = float("-inf")

        token_idxs = token_nn_scores.max(dim=-1).indices

        return rbf_utility[:,best_utility_hyp_id], token_idxs

    def sample_output(self, decoder_out: Tensor, n_samples: int = 1, token_mbr=False) -> Tuple[Tensor, Tensor]:
        if token_mbr is False:
            return self.sample_mbr_string(decoder_out)
        else:
            return self.sample_mbr_token(decoder_out, n_samples=n_samples)

class HyperCubeEmbed(nn.Module):
    def __init__(self, num_emb, embed_dim, pad_idx=None, permute=False, dtype=None, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.permute = permute
        self.num_emb = num_emb
        self.fac_kwargs = {
            'dtype': dtype,
            'requires_grad': False,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        self.pad_idx = pad_idx
        self.weight = self.build_cube_embed()


    def build_cube_embed(self):
        #stretch embeddings in the cube
        if self.permute:
            d = torch.randint(0,2**(self.embed_dim - 1),(self.num_emb*2,), **self.fac_kwargs).unique()[:self.num_emb]
        else:
            step = (2 ** (self.embed_dim - 1)) // self.num_emb
            d = torch.arange(0,self.num_emb, **self.fac_kwargs)*step
        mask = 2 ** torch.arange(self.embed_dim - 1, -1, -1).to(d.device, d.dtype)
        d = d.unsqueeze(-1).bitwise_and(mask).ne(0).float()
        d[d == 0] = -1

        return d

    def forward(self, x):
        with torch.no_grad():
            return self.weight[x]
