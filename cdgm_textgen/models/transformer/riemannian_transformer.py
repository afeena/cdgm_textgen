import logging
from dataclasses import field, dataclass
from typing import Optional, Dict, List, Any

from fairseq.models.transformer import TransformerConfig, TransformerDecoderBase
from torch import Tensor

from cdgm_textgen.models import register_model
from cdgm_textgen.models.transformer import TransformerModelBase
from cdgm_textgen.models.transformer.transformer_base import Embedding
from cdgm_textgen.modules.riemannian_embeddings import build_manifold_embeddings
from cdgm_textgen.modules.riemannian_linear import build_manifold_linear


logger = logging.getLogger("cdgm_textgen.models.transformer.riemann_transformer")


@dataclass
class RiemannianTransformerConfig(TransformerConfig):
    riemannian_enc_inp: Optional[bool] = field(
        default=False,
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
                cfg, src_dict, cfg.encoder.embed_dim, cfg.encoder.embed_path, manifold if cfg.riemannian_enc_inp else None
            )
            decoder_embed_tokens = encoder_embed_tokens
            cfg.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(
                cfg, src_dict, cfg.encoder.embed_dim, cfg.encoder.embed_path, manifold if cfg.riemannian_enc_inp else None
            )
            decoder_embed_tokens = cls.build_embedding(
                cfg, tgt_dict, cfg.decoder.embed_dim, cfg.decoder.embed_path, manifold if cfg.riemannian_dec_inp else None
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

    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens, manifold=None):
        decoder = ManifoldDecoder(cfg, tgt_dict, embed_tokens, manifold=manifold)
        return decoder

    @classmethod
    def build_embedding(cls, cfg, dictionary, embed_dim, path=None, manifold=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        if manifold is not None:
            return build_manifold_embeddings(num_embeddings, embed_dim, manifold, padding_idx=padding_idx)
        else:
            embeddings = Embedding(num_embeddings, embed_dim, padding_idx)
            if path:
                embeddings.load_embeddings(path, dictionary)
        return embeddings


class ManifoldDecoder(TransformerDecoderBase):
    def __init__(
        self,
        cfg,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
        manifold=None,
    ):
        super(ManifoldDecoder, self).__init__(cfg, dictionary, embed_tokens, no_encoder_attn, output_projection)
        self.manifold = manifold
        if cfg.riemannian_dec_out is True and self.manifold is not None:
            self.build_manifold_output_projection(cfg, dictionary, embed_tokens)

    def build_manifold_output_projection(self, cfg, dictionary, embed_tokens):
        if self.share_input_output_embed:
            self.output_projection = build_manifold_linear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                self.manifold,
                padding_idx=self.padding_idx
            )
            self.output_projection.weight = self.embed_tokens.weight
        else:
            self.output_projection = build_manifold_linear(
                self.output_embed_dim, len(dictionary), self.manifold, padding_idx=self.padding_idx
            )

    def extract_features(
            self,
            prev_output_tokens,
            encoder_out: Optional[Dict[str, List[Tensor]]],
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
    ):
        return self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )

    """
    A scriptable subclass of this class has an extract_features method and calls
    super().extract_features, but super() is not supported in torchscript. A copy of
    this function is made to be used in the subclass instead.
    """

    def extract_features_scriptable(
            self,
            prev_output_tokens,
            encoder_out: Optional[Dict[str, List[Tensor]]],
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        bs, slen = prev_output_tokens.size()
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        enc: Optional[Tensor] = None
        padding_mask: Optional[Tensor] = None
        if encoder_out is not None and len(encoder_out["encoder_out"]) > 0:
            enc = encoder_out["encoder_out"][0]
        if encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0:
            padding_mask = encoder_out["encoder_padding_mask"][0]

        # embed positions
        positions = None
        if self.embed_positions is not None:
            positions = self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # Prevent torchscript exporting issue for dynamic quant embedding
        prev_output_tokens = prev_output_tokens.contiguous()
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        # here if manifold is not None, we project the input from manifold
        if hasattr(self.embed_tokens, "manifold"):
            self.embed_tokens.to_tangent_space(x)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, layer_attn, _ = layer(
                x,
                enc,
                padding_mask,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}
