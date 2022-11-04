from torch.nn import Identity

from cdgm_textgen.models.transformer.transformer_base import Embedding
from fairseq import utils
import faiss
import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple

from fairseq.models.transformer import TransformerDecoderBase


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
        self.metric = cfg.decoder_nn_metric
        if cfg.no_last_dec_layer_norm:
            self.layers[-1].final_layer_norm = Identity(self.embed_dim, export=cfg.export)

        #build target (reference) emb
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        self.target_embed  = Embedding(num_embeddings, self.output_embed_dim, padding_idx)
        if cfg.target_embed_path:
            embed_dict = utils.parse_embedding(cfg.target_embed_path)
            utils.load_embedding(embed_dict, dictionary, self.target_embed)


    def build_faiss_index(self):
        res = faiss.StandardGpuResources()
        if self.metric == "cosine":
            self.faiss_index = faiss.GpuIndexFlat(res, self.target_embed.weight.shape[-1], faiss.METRIC_INNER_PRODUCT)
            self.faiss_index.add(F.normalize(self.target_embed.weight, dim=-1))
        elif self.metric == "l2":
            self.faiss_index = faiss.GpuIndexFlat(res, self.target_embed.weight.shape[-1], faiss.METRIC_L2)
            self.faiss_index.add(self.target_embed.weight.data)

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
    ):
        with torch.no_grad():
            out = net_output[0]

            #  kappa = torch.norm(out, dim=-1)

            scores = self.nearest_neighbor_search(out)
            # m = torch.tensor([self.target_embed.weight.shape[-1]], device=net_output[0].device)
            # norm_const = _neglogcmk(m, kappa)
            # scores += norm_const.squeeze()

            return scores

    def nearest_neighbor_search(self, features, k_nearest=5):
        with torch.no_grad():
            if self.metric == "cosine":
                features = F.normalize(features, p=2.0, dim=-1)
            # faiss works with 2d arrays only, so we need to transform 3d->2d and back
            ff = torch.flatten(features, start_dim=0, end_dim=1)
            sim, _tokens = self.faiss_index.search(ff, k_nearest)
            _tokens = _tokens.view(features.shape[0], features.shape[1], -1)
            sim = sim.view(features.shape[0], features.shape[1], -1)

            # sequence generator expects the matrix emb x vocab, so...
            output_proj = torch.full(
                (features.shape[0], features.shape[1], self.dict_len), 0.0
            ).to(features.device)

            output_proj = output_proj.scatter_(-1, _tokens, sim)

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
                                                            self.output_embed_dim * self.num_mixture_components)
        # output of the mixture FF network
        self.mixture_output_ff = torch.nn.Linear(self.output_embed_dim,
                                                 self.num_mixture_components)

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

        mixtures_out = self.mixture_output_ff(x)

        extra["mixtures_out"] = mixtures_out
        x = self.ouput_ff_with_mix_components(x)

        return x.view(x.shape[0], -1, self.num_mixture_components, self.output_embed_dim), extra

    def get_normalized_probs(
            self,
            net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
            log_probs: bool,
            sample: Optional[Dict[str, Tensor]] = None,
    ):
        with torch.no_grad():
            out = net_output[0]

            if self.num_mixture_components > 1:
                out = self.get_max_mixture_component(out,
                                                     net_output[1]["mixtures_out"])

            #  kappa = torch.norm(out, dim=-1)

            scores = self.nearest_neighbor_search(out)
            # m = torch.tensor([self.target_embed.weight.shape[-1]], device=net_output[0].device)
            # norm_const = _neglogcmk(m, kappa)
            # scores += norm_const.squeeze()

            return scores

    def get_max_mixture_component(self,
                                  out: Tensor,
                                  mixtures_out: Tensor):
        bsz = out.shape[0]
        _, idx = F.softmax(mixtures_out, -1).max(dim=-1)
        idx_bool_mask = F.one_hot(idx, num_classes=self.num_mixture_components).bool()
        out = out[idx_bool_mask].view(bsz, -1, self.output_embed_dim)
        return out
