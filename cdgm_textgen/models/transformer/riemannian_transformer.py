import logging
from dataclasses import field, dataclass
from typing import Optional, Dict, List, Any

import geoopt.manifolds.stereographic.math as pmath
import torch
import torch.nn as nn
from fairseq import utils
from fairseq.models.transformer import TransformerConfig, TransformerDecoderBase
from geoopt import SphereExact, ManifoldParameter, StereographicExact, Lorentz
from torch import Tensor

from cdgm_textgen.models import register_model
from cdgm_textgen.models.transformer import TransformerModelBase
from cdgm_textgen.models.transformer.transformer_base import Embedding

logger = logging.getLogger("cdgm_textgen.models.transformer.riemann_transformer")


_MANIFOLDS = {
              "sphere": SphereExact(),
              "poincare": StereographicExact(k=-1.0),
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

        if self.manifold is None:
            x = self.output_projection(x)
            return x, extra
        else:
            # here: map output x to manifold and perform linear projection on manifold
            # better to work with the float64 precision
            x = x.double()

            if isinstance(self.manifold, Lorentz):
                #https://github.com/chenweize1998/fully-hyperbolic-nn/blob/main/mt/onmt/modules/hyper_nets.py#L41

                x = self.manifold.expmap0(x)
                x = self.output_projection(x)

                x_narrow = x.narrow(1, 1, x.shape[0] - 1)

                time = x.narrow(1, 0, 1).sigmoid() + 1.1
                scale = (time * time - self.manifold.k) / \
                (x_narrow * x_narrow).sum(dim=1, keepdim=True)
                x = torch.cat([time, x_narrow * scale.sqrt()], dim=1)

            if isinstance(self.manifold, StereographicExact):
                x = pmath.mobius_matvec(self.output_projection.weight,
                                        self.manifold.expmap0(x),
                                        k=self.manifold.k)

            if isinstance(self.manifold, SphereExact):
                x = self.output_projection(x)
                x = torch.nn.functional.normalize(x, dim=1)
            x = x.float()
            return x, extra

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

        #here if manifold is not None, we project the input from manifold
        if self.manifold is not None and isinstance(self.manifold, StereographicExact):
            x = pmath.project(pmath.logmap0(x, k=self.manifold.k), k=self.manifold.k)

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