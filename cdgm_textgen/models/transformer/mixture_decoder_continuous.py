import logging
from typing import Any, Dict, List, Optional, Tuple

import power_spherical
import torch
import torch.nn.functional as F
from functorch import vmap, combine_state_for_ensemble
from torch import Tensor

from cdgm_textgen.models.transformer.decoder_continuous import ContinuousTransformerDecoder
from cdgm_textgen.modules import parse_cluster_means, triple_sum_scriptable, log_prob_scriptable
from cdgm_textgen.modules.riemannian_clustering import cluster_embeddings_riemannian

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
            self.ff_output_head = torch.nn.Linear(self.embed_dim,
                                                  self.output_embed_dim, bias=False)
            self.ff_cluster_head = torch.nn.Linear(self.output_embed_dim,
                                                   self.output_embed_dim, bias=False)
            if cfg.mixture_clusters_init_path is not None:
                cluster_means = parse_cluster_means(cfg.mixture_clusters_init_path, torch.Size(
                    [self.num_mixture_components, self.output_embed_dim]))
            else:
                cluster_means = cluster_embeddings_riemannian(self.target_embed_normalized,
                                                              self.num_mixture_components)
            self.cluster_means = torch.nn.Parameter(cluster_means, requires_grad=not cfg.freeze_mixture_clusters)

        self.uniform_prior = cfg.uniform_prior
        if not self.uniform_prior:
            # define mixture head to predict the cluster
            self.ff_mixture_head = torch.nn.Linear(self.embed_dim,
                                                   self.num_mixture_components, bias=False)

    def forward(
            self,
            prev_output_tokens: Tensor,
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
            f, x = self.forward_clustered_output(x)
            extra["norm_f"] = torch.norm(f, dim=-1)

        return x, extra

    def forward_output_head_parallel(self, x: Tensor) -> Tensor:
        # map the output to the mixture components
        fmodel, mixture_params, mixture_buffers = combine_state_for_ensemble(self.ff_output_head)
        x = vmap(fmodel, (0, 0, None))(mixture_params, mixture_buffers, x)
        return x

    def forward_clustered_output(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        # map the output to the mixture components
        c_means = self.cluster_means.unsqueeze(1).unsqueeze(1)
        x_out = self.ff_output_head(x)
        cluster_out = self.ff_cluster_head(c_means)

        return triple_sum_scriptable(x_out, cluster_out, c_means)

    def forward_mixture_head(self, decoder_output: Tensor) -> Tensor:
        if not self.uniform_prior:
            mixtures_out = self.ff_mixture_head(decoder_output)
        else:

            mixtures_out = torch.ones(decoder_output.size(0),
                                      decoder_output.size(1),
                                      self.num_mixture_components,
                                      device=decoder_output.device)
        return mixtures_out

    def get_mixture_output(self, decoder_out: Tensor,
                           mixtures_out: Tensor,
                           metric: str,
                           reduce: str = None) -> Tensor:
        with torch.no_grad():
            score_fn = self.get_ps_likelihood_scores
            if metric == "cosine":
                score_fn = self.get_cosine_similarity_scores

            mixture_probs = F.softmax(mixtures_out, -1)
            # Get model_scores in MxBxTxD where T=1
            model_scores = score_fn(decoder_out)

            if reduce == "max":
                # choose max before getting the model scores
                model_scores = model_scores * mixture_probs.movedim(2, 0)[..., None]
                best_mixt_idx = torch.max(model_scores, dim=0).indices
                model_scores = model_scores.gather(0, best_mixt_idx.unsqueeze(0))
                # for debug purposes
                # from datetime import datetime
                # model_scores[:,:,:,self.dictionary.pad()] = -1e10
                # vocab_tokens_idx = torch.topk(model_scores, 5, dim=-1).indices
                # mixt_of_best_token = best_mixt_idx.gather(-1, vocab_tokens_idx.squeeze(2))
                # torch.save(mixt_of_best_token, f"mixt_of_best5_token_{datetime.now().timestamp()}")

            if reduce == "sum":
                model_scores = torch.sum(model_scores, dim=0)

        return model_scores

    def get_ps_likelihood_scores(self, features: Tensor) -> Tensor:
        """get likelihood of target tokens based on output PS distribution"""
        with torch.no_grad():
            mixt = features.shape[0]
            bsz = features.shape[1]
            features = F.normalize(features, p=2.0, dim=-1)
            kappa = torch.norm(features, dim=-1)

            q_dist = power_spherical.PowerSpherical(features, kappa)
            log_probs = log_prob_scriptable(q_dist.loc, q_dist.scale,
                                            self.target_embed_normalized,
                                            "mijk,lk->mil")

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

    def sample_output_tokens(self,
                            decoder_out: Tensor,
                            mixtures_out: Tensor,
                            n_samples: int = 1) -> Tuple[Tensor, Tensor]:
        """
        sample mixture index first, then sample from respective PS distribution
        decoder_out: MxBxTxD
        mixtures_out: BxTxM
        n_samples: how much mixtures to sample
        Returns: token idx BxTxN_samples
        """
        bsz, seq_len, num_mixt = mixtures_out.shape
        # BxTxM
        mixture_log_probs = F.log_softmax(mixtures_out, -1)

        #from datetime import datetime
        #postfix = f"_{datetime.now().timestamp()}"
        #torch.save(mixture_probs, f"mixture_probs{postfix}")
        # BxTxN_samples
        mixture_samples = torch.distributions.Categorical(logits=mixture_log_probs).sample()
        mixture_log_probs = mixture_log_probs.gather(-1, mixture_samples)
        #torch.save(mixture_samples, f"mixture_samples{postfix}")
        # N_samplesxBxT
        mixture_samples = mixture_samples.view(1, bsz, seq_len)

        # define output distribution
        kappa: Tensor = torch.norm(decoder_out, dim=-1)
        features: Tensor = F.normalize(decoder_out, p=2.0, dim=-1)

        q_dist: power_spherical.PowerSpherical = power_spherical.PowerSpherical(features, kappa)

        # sample from output dist
        samples: Tensor = q_dist.sample()
        log_probs: Tensor = log_prob_scriptable(q_dist.loc, q_dist.scale,samples, "ijk, ljk -> lij")
        sampled_mixtures_out: Tensor = torch.gather(samples, 0, mixture_samples[...,None].expand(-1,-1,-1,features.shape[-1]))
        log_probs = torch.gather(log_probs, 0, mixture_samples)

        log_probs = log_probs.view(bsz, -1)+mixture_log_probs.view(bsz,-1)

        # rounding to a target vocab token
        token_nn_scores: Tensor = sampled_mixtures_out.squeeze(0) @ self.target_embed_normalized.T
        token_nn_scores[:,:,self.dictionary.pad()] = float("-inf")
    
        token_idxs = token_nn_scores.max(-1).indices
        #torch.save(token_idxs, f"token_idxs{postfix}")
        
        return log_probs, token_idxs.view(bsz, -1)

    def get_normalized_probs(
            self,
            net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
            log_probs: bool,
            sample: Optional[Dict[str, Tensor]] = None,
            metric: Optional[str] = "ps",
            reduce_strategy: Optional[str] = "max",
    ):
        with torch.no_grad():
            scores = self.get_mixture_output(net_output[0], net_output[1]["mixtures_out"], metric,
                                             reduce=reduce_strategy)
            scores = scores.squeeze(0)

        return scores
