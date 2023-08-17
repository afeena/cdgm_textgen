import logging
from dataclasses import field, dataclass
from typing import Optional

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.models.transformer import TransformerConfig
from geoopt import SphereExact, ManifoldParameter

from cdgm_textgen.models import register_model
from cdgm_textgen.models.transformer import TransformerModelBase
from cdgm_textgen.models.transformer.transformer_base import Embedding

logger = logging.getLogger("cdgm_textgen.models.transformer.riemann_transformer")


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

        # here set the embed to the optimizer group of riemannian adam
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
        decoder = super().build_decoder(cfg, tgt_dict, embed_tokens)
        if cfg.manifold == 'sphere' and not cfg.share_decoder_input_output_embed:
            decoder.output_projection.weight = ManifoldParameter(decoder.output_projection.weight,
                                                                   manifold=SphereExact())
            decoder.output_projection.weight.data /= decoder.output_projection.weight.data.norm(
            dim=-1, keepdim=True)
        return decoder
    @classmethod
    def build_embedding(cls, cfg, dictionary, embed_dim, path=None, manifold=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        if not manifold:
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
        elif manifold == 'sphere':
            emb = SphericalEmbedding(num_embeddings, embed_dim, padding_idx)
        else:
            raise NotImplementedError(f"{manifold} embeddings are not implemented")

        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        if manifold == 'sphere':
            # make sure after loading embeddings are on the sphere
            emb.weight.data /= emb.weight.data.norm(dim=-1, keepdim=True)
        return emb


def SphericalEmbedding(num_embeddings, embedding_dim, padding_idx):
    w_ = torch.randn(num_embeddings, embedding_dim)
    w_[padding_idx, :] = 1
    w_ /= w_.norm(dim=-1, keepdim=True)
    w = ManifoldParameter(data=w_, manifold=SphereExact())
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)

    # have to force the type here
    del m.weight
    m.weight = w
    return m
