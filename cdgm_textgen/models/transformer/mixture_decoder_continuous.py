import logging
from typing import Any, Dict, List, Optional, Tuple

import power_spherical
import torch
import torch.nn.functional as F
from functorch import vmap, combine_state_for_ensemble
from torch import Tensor

from cdgm_textgen.models.transformer.decoder_continuous import ContinuousTransformerDecoder
from cdgm_textgen.modules import parse_cluster_means

logger = logging.getLogger("cdgm_textgen.models.transformer.transformer_decoder_continuous")


class MixtureContinuousTransformerDecoder(ContinuousTransformerDecoder):
    def __init__(self,
                 cfg,
                 dictionary,
                 embed_tokens,
                 ):
        super().__init__(cfg, dictionary, embed_tokens)
        self.num_mixture_components = cfg.num_mixture_components
        self.project_out_dim = None

        # project output to have emb_dim*num_mixture_components (replaces project_out_dim)
        self.parallel_output = cfg.output_parallel
        if self.parallel_output:
            self.ff_output_head = torch.nn.ModuleList([
                torch.nn.Linear(self.embed_dim, self.output_embed_dim, bias=False)
                for _ in range(self.num_mixture_components)
            ])
        else:
            self.ff_output_head = torch.nn.Linear(self.embed_dim + self.output_embed_dim,
                                                  self.output_embed_dim, bias=False)
            if cfg.mixture_clusters_init_path is not None:
                self.cluster_means = parse_cluster_means(cfg.mixture_clusters_init_path, torch.Size(
                    [self.num_mixture_components, self.output_embed_dim]))
                self.cluster_means = torch.nn.Parameter(self.cluster_means)
                self.cluster_means.requires_grad = not cfg.freeze_mixture_clusters

        # define mixture head to predict the cluster
        self.ff_mixture_head = torch.nn.Linear(self.embed_dim,
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

        extra["mixtures_out"] = self.forward_mixture_head(x)

        if self.parallel_output:
            x = self.forward_output_head_parallel(x)
        else:
            x = self.forward_clustered_output_concat(x)

        return x, extra

    def forward_output_head_parallel(self, x):
        # map the output to the mixture components
        fmodel, mixture_params, mixture_buffers = combine_state_for_ensemble(self.ff_output_head)
        x = vmap(fmodel, (0, 0, None))(mixture_params, mixture_buffers, x)
        return x

    def forward_clustered_output_concat(self, x):
        # map the output to the mixture components
        bsz = x.shape[0]
        seq_len = x.shape[1]
        x = x[None, ...].expand(self.num_mixture_components, -1, -1, -1)
        latent_z = self.cluster_means[:, None, None, ...].expand(-1, bsz, seq_len, -1)
        x = torch.cat([x, latent_z], -1)
        x = self.ff_output_head(x)
        return x

    def forward_mixture_head(self, decoder_output):
        mixtures_out = self.ff_mixture_head(decoder_output)
        return mixtures_out

    def get_mixture_output(self, decoder_out, mixtures_out, metric, reduce=None):
        with torch.no_grad():
            score_fn = self.get_ps_likelihood_scores
            if metric == "cosine":
                score_fn = self.get_cosine_similarity_scores

            mixture_probs = F.softmax(mixtures_out, -1)
            # for debug purposes
            # from datetime import datetime
            # torch.save(mixture_probs, f"mixture_dist_{datetime.now().timestamp()}")

            if reduce == "max":
                # choose max before getting the model scores
                _, max_idx = mixture_probs.max(dim=-1)
                max_idx = max_idx.t()
                max_idx = max_idx[..., None, None].expand(-1, -1, 1, decoder_out.size(-1))
                decoder_out = decoder_out.gather(0, max_idx)

                model_scores = score_fn(decoder_out)
                del max_idx

            else:
                # Get model_scores in MxBxTxD where T=1
                model_scores = score_fn(decoder_out).squeeze(dim=2)

                mixture_probs = mixture_probs.squeeze(dim=1).unsqueeze(-1)
                model_scores = model_scores * mixture_probs.transpose(0, 1)

                if reduce == "sum":
                    model_scores = torch.sum(model_scores, dim=0)

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

    def get_cosine_similarity_scores(self, features: Tensor) -> Tensor:
        "use for valid only"
        with torch.no_grad():
            mixt = features.shape[0]
            bsz = features.shape[1]
            features = F.normalize(features, p=2.0, dim=-1).unsqueeze(1)
            output_proj = features @ self.target_embed_normalized.T

            return output_proj.view(mixt, bsz, -1, self.dict_len)

    def sample_mixture_and_output(self, decoder_out, mixtures_out, n_samples=1):
        """
        sample mixture index first, then sample from respective PS distribution
        decoder_out: MxBxTxD
        mixtures_out: BxTxM
        """
        mixtures_out_probs = F.softmax(mixtures_out, dim=-1).view(-1, self.num_mixture_components)
        mixture_idx = torch.multinomial(mixtures_out_probs, 1, replacement=True).squeeze(dim=-1)
        mixture_idx = mixture_idx.t()
        decoder_out = decoder_out[mixture_idx]
        normalized_out = F.normalize(decoder_out, p=2.0, dim=-1)
        kappa_z = torch.norm(decoder_out, dim=-1)
        q = power_spherical.PowerSpherical(normalized_out, kappa_z)
        samples = q.sample(n_samples)
        return mixture_idx, samples, q.log_prob(samples)

    def get_normalized_probs(
            self,
            net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
            log_probs: bool,
            sample: Optional[Dict[str, Tensor]] = None,
            metric: Optional[str] = "pas",
            reduce_strategy: Optional[str] = "max",
    ):
        with torch.no_grad():
            scores = self.get_mixture_output(net_output[0], net_output[1]["mixtures_out"], metric, reduce=reduce_strategy)

        return scores
