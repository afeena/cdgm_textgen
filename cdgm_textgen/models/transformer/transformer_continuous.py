from typing import Optional
from torch.nn import Identity
import logging

from dataclasses import dataclass, field

from fairseq.models.transformer import TransformerConfig
from cdgm_textgen.models.transformer.transformer_decoder_continuous import ContinuousTransformerDecoder, \
    MixtureContinuousTransformerDecoder

from cdgm_textgen.models import register_model
from cdgm_textgen.models.transformer.transformer_base import TransformerModelBase

logger = logging.getLogger("cdgm_textgen.models.transformer.transfore_continuous")


@dataclass
class ContinuousTransformerConfig(TransformerConfig):
    target_embed_path: Optional[str] = field(
        default=None,
        metadata={"help": "path to target embeddings"},
    )
    no_last_dec_layer_norm: bool = field(
        default=True,
        metadata={"help": "don't add an layernorm after the last decoder block"},
    )
    decoder_nn_metric: str = field(
        default="cosine",
        metadata={"help": "metric for NN decoding"},
    )


@register_model("continuous_transformer", dataclass=ContinuousTransformerConfig)
class TransformerModelContinuous(TransformerModelBase):
    def __init__(self, cfg, encoder, decoder):
        super().__init__(cfg, encoder, decoder)
        self.decoder_class = ContinuousTransformerDecoder

    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):
        decoder = cls.decoder_class(
            cfg,
            tgt_dict,
            embed_tokens
        )

        if cfg.no_last_dec_layer_norm:
            decoder.layers[-1].final_layer_norm = Identity(decoder.embed_dim, export=cfg.export)

        decoder.target_embed = cls.build_embedding(cfg, tgt_dict, cfg.decoder.output_dim, cfg.target_embed_path)
        decoder.target_embed.weight.requires_grad = False

        decoder.setup_faiss_index()

        return decoder

    def forward(
            self,
            src_tokens,
            src_lengths,
            prev_output_tokens,
            return_all_hiddens: bool = True,
            features_only: bool = True,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=True,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        return decoder_out

    def get_targets(self, sample, net_output):
        return self.decoder.target_embed(sample["target"])


@dataclass
class MixtureContinuousTransformerConfig(ContinuousTransformerConfig):
    num_mixture_components: int = field(
        default=1,
        metadata={"help": "number of mixture components for the distribution"},
    )


@register_model("continuous_transformer_mixture", dataclass=MixtureContinuousTransformerConfig)
class TransformerMixtureModelContinuous(TransformerModelContinuous):

    def __init__(self, cfg, encoder, decoder):
        super(TransformerModelContinuous, self).__init__(cfg, encoder, decoder)
        self.decoder_class = MixtureContinuousTransformerDecoder
        if cfg.num_mixture_components == 1:
            logger.error("Running continuous_transformer_mixture with 1 mixture component is not recommended! "
                         "Use continuous_transformer instead.")
            raise ValueError("num_mixture_components must be >1 for continuous_transformer_mixture")
