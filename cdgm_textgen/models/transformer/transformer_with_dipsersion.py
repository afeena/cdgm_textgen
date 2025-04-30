import logging
import os
import sys
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any

from fairseq.models.transformer import TransformerConfig
from fairseq.utils import reset_logging
from omegaconf import DictConfig
from torch import Tensor, nn
import torch.nn.functional as F

from cdgm_textgen.models import register_model
from cdgm_textgen.models.transformer import TransformerModelBase
from cdgm_textgen.models.transformer.transformer_decoder import TransformerDecoderBase

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("model.transformer_dispersion_ft")

@dataclass
class TransformerOutDispersionFTConfig(TransformerConfig):
    dispersion_dim: int = field(default=512, metadata={"help": "dimension of the dispersion layer"})
    train_layers: set = field(default=("decoder.layers.5.fc1",
                                        "decoder.layers.5.fc2",
                                        "decoder.layers.5.final_layer_norm",
                                       "decoder.project_out_dim"),
                               metadata={"help": "list of layers to train"}
                              )

    features_to_disperse: str = field(default="decoder_out")
@register_model("transformer_dispersion_ft", dataclass=TransformerOutDispersionFTConfig)
class TransformerOutDispersionFT(TransformerModelBase):
    '''
    This class loads pre-trained model and freeze all layers except project_out_dim and output_layer.
    project_out_dim: projects emb dimension of the pre-trained model to the desired hidden output dimension.
    output_layer: projects the hidden output dimension to the vocabulary size.
    Output of the project_out_dim layer is stored in extra['last_ff_out'] and used for calculating dispersion loss.
    Intended objective is ls_ce_output_dispersed objective.
    '''

    def __init__(self, cfg, encoder, decoder):
        super().__init__(cfg, encoder, decoder)
        self.layers_to_train = cfg.train_layers

    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):
        return TransformerDecoderOutDispersion(cfg, tgt_dict, embed_tokens,
                                               dispersion_dim=cfg.dispersion_dim,
                                               feature_to_disperse=cfg.features_to_disperse)

    @classmethod
    def build_model(cls, cfg, task):
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
            encoder_embed_tokens = cls.build_embedding(
                cfg, src_dict, cfg.encoder.embed_dim, cfg.encoder.embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            cfg.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(
                cfg, src_dict, cfg.encoder.embed_dim, cfg.encoder.embed_path
            )
            decoder_embed_tokens = cls.build_embedding(
                cfg, tgt_dict, cfg.decoder.embed_dim, cfg.decoder.embed_path
            )
        if cfg.offload_activations:
            cfg.checkpoint_activations = True  # offloading implies checkpointing
        encoder = cls.build_encoder(cfg, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(cfg, tgt_dict, decoder_embed_tokens)

        model = cls(cfg, encoder, decoder)

        # Freeze all layers except last fc layers of last decoder layer

        for name, param in model.named_parameters():
            if any(layer in name for layer in model.layers_to_train):
                param.requires_grad = True
                logger.info(f"Training {name} with shape {param.shape} and requires_grad {param.requires_grad}")
            else:
                param.requires_grad = False
                logger.info(f"Freezing {name} with shape {param.shape} and requires_grad {param.requires_grad}")
        return model

    def load_state_dict(
            self,
            state_dict,
            strict=False,
            model_cfg: Optional[DictConfig] = None,
            args: Optional[Namespace] = None,
    ):
        reset_logging()

        pretrained_dict = {}
        for k, v in state_dict.items():
            if k not in self.state_dict():
                logger.info(f"Key {k} not found in state_dict")
            elif k in self.state_dict():
                if v.shape != self.state_dict()[k].shape:
                    logger.info(f"Key {k} has shape {v.shape} but state_dict has shape {self.state_dict()[k].shape}")
                else:
                    pretrained_dict[k] = v
        strict = False
        super().load_state_dict(pretrained_dict, strict, model_cfg, args)


class TransformerDecoderOutDispersion(TransformerDecoderBase):
    def __init__(
            self,
            cfg,
            dictionary,
            embed_tokens,
            no_encoder_attn=False,
            output_projection=None,
            dispersion_dim=512,
            feature_to_disperse="decoder_out",
    ):
        super().__init__(cfg,
            dictionary,
            embed_tokens,
            no_encoder_attn=no_encoder_attn,
            output_projection=output_projection)

        self.dispersion_dim = dispersion_dim

        self.features_to_disperse = feature_to_disperse
        if self.features_to_disperse=='fc1_out' and self.dispersion_dim != cfg.decoder.ffn_embed_dim:
            self.layers[-1].fc1 = nn.Linear(self.embed_dim, self.dispersion_dim)
            self.layers[-1].fc2 = nn.Linear(self.dispersion_dim, self.embed_dim)

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
        extra["features"] = extra["features_to_disperse"]

        if not features_only:
            x = self.output_layer(x)

        return x, extra
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

        features_to_disperse = None
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

            x, layer_attn, extra = layer(
                x,
                enc,
                padding_mask,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
                return_fc1_out=bool(idx == self.num_layers - 1)&(self.features_to_disperse=="fc1_out"),
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)
            if extra is not None and type(extra) is dict and "fc1_out" in extra:
                features_to_disperse = extra["fc1_out"]

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

        if self.features_to_disperse=='decoder_out':
            features_to_disperse = x.clone()

        return x, {"attn": [attn], "inner_states": inner_states, "features_to_disperse": features_to_disperse}
