import logging
from dataclasses import field, dataclass
from typing import Optional, Dict, List, Any

import geoopt.manifolds.stereographic.math as pmath
import torch
import torch.nn as nn
from fairseq import utils
from fairseq.models.transformer import TransformerConfig, TransformerDecoderBase
from geoopt import SphereExact, ManifoldParameter, StereographicExact, Euclidean, Lorentz
from cdgm_textgen.modules.utils import stereographic_poincare_dist
from torch import Tensor

from cdgm_textgen.models import register_model
from cdgm_textgen.models.transformer import TransformerModelBase
from cdgm_textgen.models.transformer.transformer_base import Embedding

logger = logging.getLogger("cdgm_textgen.models.transformer.riemann_transformer")


_MANIFOLDS = {"euclidean": Euclidean(),
              "sphere": SphereExact(),
              "poincare": StereographicExact(k=-1.0),
              "hyperbolic": StereographicExact(k=-1.0),
              'lorentz': Lorentz(),
              }

@dataclass
class RiemannianTransformerConfig(TransformerConfig):
    riemannian_enc_inp: Optional[bool] = field(
        default=True,
        metadata={"help": "use RiemannianAdam for encoder input embeddings"},
    )
    riemannian_dec_inp: Optional[bool] = field(
        default=True,
        metadata={"help": "use RiemannianAdam for decoder input embeddings"},
    )
    riemannian_dec_out: Optional[bool] = field(
        default=True,
        metadata={"help": "use RiemannianAdam for decoder output embeddings"},
    )
    manifold: Optional[str] = field(
        default='sphere',
        metadata={"help": "use RiemannianAdam for decoder output embeddings"},
    )


@register_model("riemannian_emb_transformer", dataclass=RiemannianTransformerConfig)
class RiemannianEmbeddingsTransformerModel(TransformerModelBase):
    @classmethod
    def build_model(cls, cfg, task):
        """Build a new model instance."""
        # --  TODO T96535332
        #  bug caused by interaction between OmegaConf II and argparsing
        cfg.decoder.input_dim = int(cfg.decoder.input_dim)
        cfg.decoder.output_dim = int(cfg.decoder.output_dim)
        # --

        if cfg.encoder.layers_to_keep:
            cfg.encoder.layers = len(cfg.encoder.layers_to_keep.split(","))
        if cfg.decoder.layers_to_keep:
            cfg.decoder.layers = len(cfg.decoder.layers_to_keep.split(","))

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if cfg.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if cfg.encoder.embed_dim != cfg.decoder.embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if cfg.decoder.embed_path and (
                    cfg.decoder.embed_path != cfg.encoder.embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            assert cfg.riemannian_enc_inp == cfg.riemannian_dec_inp == cfg.riemannian_dec_out, ValueError(
                "--share-all-embeddings requires all embedding layers to be Riemannian"
            )
            encoder_embed_tokens = cls.build_embedding(
                cfg, src_dict, cfg.encoder.embed_dim, cfg.encoder.embed_path, cfg.manifold
            )
            decoder_embed_tokens = encoder_embed_tokens
            cfg.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(
                cfg, src_dict, cfg.encoder.embed_dim, cfg.encoder.embed_path, cfg.manifold
            )
            decoder_embed_tokens = cls.build_embedding(
                cfg, tgt_dict, cfg.decoder.embed_dim, cfg.decoder.embed_path, cfg.manifold
            )
        if cfg.offload_activations:
            cfg.checkpoint_activations = True  # offloading implies checkpointing
        encoder = cls.build_encoder(cfg, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(cfg, tgt_dict, decoder_embed_tokens)
        model = cls(cfg, encoder, decoder)

        for pn, p in model.named_parameters():
            if p.requires_grad:
                if cfg.riemannian_enc_inp and (pn.endswith("encoder.embed_tokens") or "encoder.embed_tokens" in pn):
                    p.param_group = "riemannian"
                    print(f"{pn} optimizer is set to RiemannianAdam, manifold: {p.manifold}")
                elif cfg.riemannian_dec_inp and (pn.endswith("decoder.embed_tokens") or "decoder.embed_tokens" in pn):
                    p.param_group = "riemannian"
                    print(f"{pn} optimizer is set to RiemannianAdam, manifold: {p.manifold}")
                elif cfg.riemannian_dec_out and (
                        pn.endswith("output_projection") or "output_projection" in pn):
                    p.param_group = "riemannian"
                    print(f"{pn} optimizer is set to RiemannianAdam, manifold: {p.manifold}")
                else:
                    p.param_group = "default"

        return model

    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):
        decoder = ManifoldDecoder(cfg, tgt_dict, embed_tokens)
        return decoder

    @classmethod
    def build_embedding(cls, cfg, dictionary, embed_dim, path=None, manifold=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = ManifoldEmbedding(num_embeddings, embed_dim, padding_idx, manifold_name=manifold)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        if manifold == 'sphere':
            # make sure after loading embeddings are on the sphere
            emb.weight.data /= emb.weight.data.norm(dim=-1, keepdim=True)
        elif manifold == 'poincare':
            # make sure after loading embeddings are on the sphere
            emb.weight.data = pmath.project(_MANIFOLDS[manifold].expmap0(emb.weight.data), k=_MANIFOLDS[manifold].k)
        return emb


def ManifoldEmbedding(num_embeddings, embedding_dim, padding_idx, manifold_name=None):
    if manifold_name is None:
        return Embedding(num_embeddings, embedding_dim, padding_idx)
    else:
        manifold = _MANIFOLDS[manifold_name]
        w_ = torch.randn(num_embeddings, embedding_dim)
        w_[padding_idx, :] = 1
        if manifold_name == 'sphere':
            w_ /= w_.norm(dim=-1, keepdim=True)
        elif manifold_name == 'poincare' or manifold_name == 'lorentz':
            w_ = pmath.project(manifold.expmap0(w_), k=manifold.k)
        w = ManifoldParameter(data=w_, manifold=manifold)
        m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)

        del m.weight
        m.weight = w
        return m

class ManifoldDecoder(TransformerDecoderBase):
    def __init__(
            self,
            cfg,
            dictionary,
            embed_tokens,
            no_encoder_attn=False,
            output_projection=None,
    ):
        super().__init__(cfg, dictionary, embed_tokens, no_encoder_attn, output_projection)
        self.manifold = _MANIFOLDS[cfg.manifold]
        if (cfg.manifold == 'sphere' or  cfg.manifold == 'sphereproduct') and not cfg.share_decoder_input_output_embed:
            self.output_projection.weight = ManifoldParameter(self.output_projection.weight,
                                                              manifold=_MANIFOLDS[cfg.manifold])
            self.output_projection.weight.data /= self.output_projection.weight.data.norm(
                dim=-1, keepdim=True)
        if (cfg.manifold == 'poincare' or cfg.manifold == 'lorentz') and not cfg.share_decoder_input_output_embed:
            self.output_projection.weight = ManifoldParameter(self.output_projection.weight,
                                                              manifold=_MANIFOLDS[cfg.manifold])
            self.output_projection.weight.data = pmath.project(_MANIFOLDS[cfg.manifold].expmap0(
                self.output_projection.weight.data), k=_MANIFOLDS[cfg.manifold].k)

    def forward(
            self,
            prev_output_tokens,
            encoder_out: Optional[Dict[str, List[Tensor]]] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            features_only: bool = False,
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

        if not isinstance(self.manifold, StereographicExact):
            x = self.output_projection(x)
        else:
            x = x @ self.manifold.logmap0(self.output_projection.weight).T

        return x, extra
