from argparse import Namespace
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any

from fairseq.models.transformer import TransformerConfig
from omegaconf import DictConfig
from torch import Tensor
from torch.nn import Linear

from cdgm_textgen.models import register_model
from cdgm_textgen.models.transformer import TransformerModelBase
from cdgm_textgen.models.transformer.transformer_decoder import TransformerDecoderBase


@dataclass
class TransformerWithDispersionAdapterConfig(TransformerConfig):
    adapter_dim: int = field(
        default=128,
        metadata={"help": "Whether to use the dispersion adapter."},
    )


@register_model("transformer_with_dispersion_adapter", dataclass=TransformerWithDispersionAdapterConfig)
class TransformerWithDispersionAdapter(TransformerModelBase):
    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):
        return TransformerDecoderWithDispersionAdapter(cfg, tgt_dict, embed_tokens, adapter_dim=cfg.adapter_dim)

    def build_model(cls, cfg, task):
        super().build_model(cfg, task)

        for name, param in cls.named_parameters():
            if "dispersion_adapter" in name or "adapter_to_dim_proj" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def load_state_dict(
            self,
            state_dict,
            strict=False,
            model_cfg: Optional[DictConfig] = None,
            args: Optional[Namespace] = None,
    ):
        # Load the model state dict
        super().load_state_dict(state_dict, strict, model_cfg, args)

class TransformerDecoderWithDispersionAdapter(TransformerDecoderBase):
    def __init__(
            self,
            cfg,
            dictionary,
            embed_tokens,
            no_encoder_attn=False,
            output_projection=None,
            adapter_dim=128,
    ):

        super().__init__(
            cfg = cfg,
            dictionary = dictionary,
            embed_tokens = embed_tokens,
            no_encoder_attn = no_encoder_attn,
            output_projection = output_projection
        )
        self.adapter_dim = adapter_dim

        self.linear_adapter = self.build_linear_adapter(cfg)
        if self.adapter_dim!=self.output_embed_dim:
            self.adapter_to_dim_proj = Linear(self.adapter_dim, self.embed_dim)
        else:
            self.adapter_to_dim_proj = None

    def build_linear_adapter(self, cfg):
        return Linear(cfg.decoder.embed_dim, cfg.decoder.adapter_dim)

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
        extra["last_ff_out"] = x.clone()

        if not features_only:
            x = self.output_layer(x)

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

        if self.linear_adapter:
            x = self.forward_linear_adapter(x)
            embed_after_adapter = x.clone()
            if self.adapter_to_dim_proj is not None:
                x = self.adapter_to_dim_proj(x)


        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states, "embed_after_adapter":embed_after_adapter}

    def forward_linear_adapter(self, x):
        """
            Args: x (torch.Tensor): input tensor with shape `(batch_size, T, C)`, hidden state
            Returns: x (torch.Tensor): output tensor with shape `(batch_size, T, C)`
        """
        residual = x
        x = self.linear_adapter(x)
        x = self.torch.relu(x + residual)

        return x