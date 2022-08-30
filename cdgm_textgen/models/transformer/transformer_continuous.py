from typing import Any, Dict, List, Optional, Tuple
import faiss
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Identity

from dataclasses import dataclass, field

from fairseq.models.transformer import TransformerConfig, TransformerDecoderBase

from cdgm_textgen.models import register_model
from cdgm_textgen.models.transformer.transformer_base import TransformerModelBase
from cdgm_textgen.modules.cmk_approx import _neglogcmk


@dataclass
class ContinuousTransformerConfig(TransformerConfig):
    target_embed_path: Optional[str] = field(
        default=None,
        metadata={"help": "path to target embeddings"},
    )
    no_last_dec_layer_norm: bool = field(
        default=True,
        metadata={"help": "don't add an layernorm after the last decoder block"},
    )
    decoder_nn_metric: str = field(
        default="cosine",
        metadata={"help": "metric for NN decoding"},
    )


@register_model("continuous_transformer", dataclass=ContinuousTransformerConfig)
class TransformerModelContinuous(TransformerModelBase):
    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):
        decoder = ContinuousTransformerDecoder(
            cfg,
            tgt_dict,
            embed_tokens,
            metric=cfg.decoder_nn_metric
        )
        if cfg.no_last_dec_layer_norm:
            decoder.layers[-1].final_layer_norm = Identity(decoder.embed_dim, export=cfg.export)

        decoder.target_embed = cls.build_embedding(cfg, tgt_dict, cfg.decoder.output_dim, cfg.target_embed_path)

        decoder.target_embed.weight.requires_grad = False
        decoder.setup_faiss_index()

        return decoder

    def forward(
            self,
            src_tokens,
            src_lengths,
            prev_output_tokens,
            return_all_hiddens: bool = True,
            features_only: bool = True,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=True,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        return decoder_out

    def get_targets(self, sample, net_output):
        return self.decoder.target_embed(sample["target"])


class ContinuousTransformerDecoder(TransformerDecoderBase):
    def __init__(self, cfg,
                 dictionary,
                 embed_tokens,
                 no_encoder_attn=False,
                 output_projection=None,
                 metric="cosine"
                 ):
        super().__init__(cfg,
                         dictionary, embed_tokens,
                         no_encoder_attn=no_encoder_attn,
                         output_projection=output_projection,
                         )
        self.target_embed = None
        self.faiss_index = None
        self.output_projection = None
        self.dict_len = len(self.dictionary)
        self.metric = metric

    def setup_faiss_index(self):
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
        scores = self.__nearest_neighbor_search(net_output[0])
        # TODO this is only for case k=1!
        with torch.no_grad():
            m = torch.tensor([self.target_embed.weight.shape[-1]], device=net_output[0].device)
            k = torch.tensor([1.], device=net_output[0].device)
            scores += _neglogcmk(m, k)

        return scores

    def __nearest_neighbor_search(self, features, beam_size=5):
        with torch.no_grad():
            if self.metric == "cosine":
                features = F.normalize(features, p=2.0, dim=-1)
            # faiss works with 2d arrays only, so we need to transform 3d->2d and back
            ff = torch.flatten(features, start_dim=0, end_dim=1)
            sim, _tokens = self.faiss_index.search(ff, beam_size)
            _tokens = _tokens.view(features.shape[0], features.shape[1], -1)
            sim = sim.view(features.shape[0], features.shape[1], -1)

            # sequence generator expects the matrix emb x vocab, so...
            output_proj = torch.full(
                (features.shape[0], features.shape[1], self.dict_len), 0.0
            ).to(features.device)

            output_proj = output_proj.scatter_(-1, _tokens, sim)

            return output_proj
