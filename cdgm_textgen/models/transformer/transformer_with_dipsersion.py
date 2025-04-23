import logging
import os
import sys
from argparse import Namespace
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
            if "dispersion_block" in name or "output_projection" in name:
                param.requires_grad = True
                logger.info(f"Training {name} with shape {param.shape} and requires_grad {param.requires_grad}")
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
        reset_logging()

        pretrained_dict = {}
        for k, v in state_dict.items():
            if k not in self.state_dict():
                logger.info(f"Key {k} not found in state_dict")
            elif k.startswith("decoder.output_projection"):
                if v.shape != self.state_dict()[k].shape:
                    logger.info(f"Key {k} shape mismatch: {v.shape} vs {self.state_dict()[k].shape}")
                    logger.info("skipping decoder.output_projection")
                continue
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
    ):

        super().__init__(
            cfg=cfg,
            dictionary=dictionary,
            embed_tokens=embed_tokens,
            no_encoder_attn=no_encoder_attn,
            output_projection=output_projection
        )
        self.dispersion_block = self.build_dispersion_block(cfg, dictionary, embed_tokens)

    def build_dispersion_block(self, cfg, dictionary, embed_tokens):
        def weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.eye_(m.weight)

        disp_block = nn.Sequential(
            nn.Linear(cfg.decoder.output_dim, cfg.decoder.output_dim),
            nn.ReLU(),
            nn.Linear(cfg.decoder.output_dim, cfg.decoder.output_dim, bias=False)
        )
        disp_block.apply(weights_init)
        return disp_block

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
        features = self.dispersion_block(x)
        features = F.relu(features)
        extra["features"] = features.clone()

        if not features_only:
            x = self.output_layer(x)

        return x, extra
