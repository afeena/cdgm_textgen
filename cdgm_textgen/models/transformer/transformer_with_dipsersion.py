import logging
import os
import sys
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Optional

from fairseq.models.transformer import TransformerConfig
from fairseq.utils import reset_logging
from omegaconf import DictConfig


from cdgm_textgen.models import register_model
from cdgm_textgen.models.transformer import TransformerModelBase
from cdgm_textgen.models.transformer.transformer_decoder_dispersion import TransformerDecoderOutDispersion, TransformerDecoderWithDispersionAdapter


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

    features_to_disperse: str = field(default="decoder_out", metadata={"help": "features to disperse, choices are: decoder_out, out_before_ffn, fc1_out, fc2_out, out_after_ln, adapter"})
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
        if cfg.features_to_disperse == "adapter":
            return TransformerDecoderWithDispersionAdapter(cfg, tgt_dict, embed_tokens,
                                                           dispersion_dim=cfg.dispersion_dim)
        else:
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
        if model.training:
            freezed_count = 0
            for name, param in model.named_parameters():
                if any(layer in name for layer in model.layers_to_train):
                    param.requires_grad = True
                    logger.info(f"Training {name} with shape {param.shape} and requires_grad {param.requires_grad}")
                else:
                    param.requires_grad = False
                    freezed_count+=1
            logger.info(f"Freezing {freezed_count} parameters")
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