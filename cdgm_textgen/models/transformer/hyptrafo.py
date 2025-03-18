from fairseq.models.transformer import TransformerConfig
from fairseq.modules import BaseLayer
from hypll import nn as hnn
from hypll.manifolds.poincare_ball import Curvature, PoincareBall
from hypll.tensors import TangentTensor
from typing import Dict, List, Optional

from cdgm_textgen.models import register_model
from transformer_decoder import TransformerDecoder
from transformer_base import TransformerModelBase

@register_model("transformer_hypll", dataclass=TransformerConfig)
class HypTransformerModel(TransformerModelBase):
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):
        return HypTransformerDecoder(cfg, tgt_dict, embed_tokens)

    @classmethod
    def build_model(cls, cfg, task):
        """Build a new model instance."""
        # --  TODO T96535332
        #  bug caused by interaction between OmegaConf II and argparsing
        cfg.decoder.input_dim = int(cfg.decoder.input_dim)
        cfg.decoder.output_dim = int(cfg.decoder.output_dim)
        # --
        manifold = cfg.manifold
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
                cfg, src_dict, cfg.encoder.embed_dim, cfg.encoder.embed_path,
                manifold if cfg.riemannian_enc_inp else None
            )
            decoder_embed_tokens = encoder_embed_tokens
            cfg.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(
                cfg, src_dict, cfg.encoder.embed_dim, cfg.encoder.embed_path,
                manifold if cfg.riemannian_enc_inp else None
            )
            decoder_embed_tokens = cls.build_embedding(
                cfg, tgt_dict, cfg.decoder.embed_dim, cfg.decoder.embed_path,
                manifold if cfg.riemannian_dec_inp else None
            )
        if cfg.offload_activations:
            cfg.checkpoint_activations = True  # offloading implies checkpointing
        encoder = cls.build_encoder(cfg, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(cfg, tgt_dict, decoder_embed_tokens, manifold)

        model = cls(cfg, encoder, decoder)

        for pn, p in model.named_parameters():
            if p.requires_grad:
                if hasattr(p, "manifold"):
                    p.param_group = "riemannian"
                    print(f"{pn} optimizer is set to RiemannianAdam, manifold: {p.manifold}")
                else:
                    p.param_group = "default"
        return model


class HypTransformerDecoder(TransformerDecoder):
    def __init__(self, cfg,
                 dictionary,
                 embed_tokens,
                 no_encoder_attn=False,
                 output_projection=None):

        super().__init__(cfg,
                         dictionary,
                         embed_tokens,
                         no_encoder_attn=no_encoder_attn,
                         output_projection=output_projection)

        self.manifold = PoincareBall(c=Curvature(requires_grad=True))

    def build_output_projection(self, cfg, dictionary, embed_tokens):
        if self.share_input_output_embed:
            self.output_projection = hnn.HLinear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=False,
                manifold=self.manifold
            )
            self.output_projection.weight = self.embed_tokens.weight
        else:
            self.output_projection = hnn.HLinear(
                self.output_embed_dim, len(dictionary), bias=False, manifold=self.manifold,
            )

        num_base_layers = cfg.base_layers
        for i in range(num_base_layers):
            self.layers.insert(
                ((i + 1) * cfg.decoder.layers) // (num_base_layers + 1),
                BaseLayer(cfg),
            )
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

        tangents = TangentTensor(data=x, man_dim=1, manifold=self.manifold)
        manifold_inputs = self.manifold.expmap(tangents)
        x = self.output_layer(x)

        return x, extra