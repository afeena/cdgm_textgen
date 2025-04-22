from argparse import Namespace
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
import torch

from fairseq.models.transformer import TransformerConfig
from omegaconf import DictConfig
from torch import Tensor
from torch.nn import Linear

from cdgm_textgen.models import register_model
from cdgm_textgen.models.transformer import TransformerModelBase
from cdgm_textgen.models.transformer.transformer_decoder import TransformerDecoderBase






@register_model("transformer_dispersion_ft", dataclass=TransformerConfig)
class TransformerOutDispersionFT(TransformerModelBase):
    '''
    This class loads pre-trained model and freeze all layers except project_out_dim and output_layer.
    project_out_dim: projects emb dimension of the pre-trained model to the desired hidden output dimension.
    output_layer: projects the hidden output dimension to the vocabulary size.
    Output of the project_out_dim layer is stored in extra['last_ff_out'] and used for calculating dispersion loss.
    Intended objective is ls_ce_output_dispersed objective.
    '''
    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):
        return TransformerDecoderOutDispersion(cfg, tgt_dict, embed_tokens)
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


        for name, param in model.named_parameters():
            if "project_out_dim" in name or "output_layer" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        return model

    def load_state_dict(
            self,
            state_dict,
            strict=False,
            model_cfg: Optional[DictConfig] = None,
            args: Optional[Namespace] = None,
    ):
        # Load the model state dict
        strict = False
        super().load_state_dict(state_dict, strict, model_cfg, args)

class TransformerDecoderOutDispersion(TransformerDecoderBase):
    def __init__(
            self,
            cfg,
            dictionary,
            embed_tokens,
            no_encoder_attn=False,
            output_projection=None,
    ):

        super().__init__(
            cfg = cfg,
            dictionary = dictionary,
            embed_tokens = embed_tokens,
            no_encoder_attn = no_encoder_attn,
            output_projection = output_projection
        )
        if self.project_out_dim is None:
            self.project_out_dim = Linear(
                cfg.decoder.embed_dim, cfg.decoder.output_dim, bias=False
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
        extra["last_ff_out"] = x.clone()

        if not features_only:
            x = self.output_layer(x)

        return x, extra