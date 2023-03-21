import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List

import torch
from fairseq.models.transformer import TransformerConfig
from torch import Tensor

from cdgm_textgen.models import register_model
from cdgm_textgen.models.transformer.transformer_base import TransformerModelBase
from cdgm_textgen.models.transformer.decoder_continuous import ContinuousTransformerDecoder
from cdgm_textgen.models.transformer.mixture_decoder_continuous import MixtureContinuousTransformerDecoder
from cdgm_textgen.modules.utils import output_with_temperature_scriptable
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


@register_model("continuous_transformer", dataclass=ContinuousTransformerConfig)
class TransformerModelContinuous(TransformerModelBase):

    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):
        decoder = ContinuousTransformerDecoder(
            cfg,
            tgt_dict,
            embed_tokens,
        )
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

    def load_state_dict(
            self,
            state_dict,
            strict=False,
            model_cfg=None,
    ):
        """Copies parameters and buffers from *state_dict* into this module and
        its descendants.

        Overrides the method in :class:`nn.Module`. Compared with that method
        this additionally "upgrades" *state_dicts* from old checkpoints.
        """
        model_dict = self.state_dict()
        # filter out keys
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        self.upgrade_state_dict(state_dict)
        # here allow partial loading from other models
        return super().load_state_dict(state_dict, strict=False)

    def get_targets(self, sample, net_output):
        return self.decoder.target_embed(sample["target"])

    def forward_decoder_scores(self, decoder_out: Tuple[Tensor, Dict], metric: str) -> Tensor:
        assert metric is not None
        scores = None
        model_out = decoder_out[0]

        if metric == "cosine":
            scores = self.decoder.get_cosine_similarity_scores(model_out)
        if metric == "ps":
            scores = self.decoder.get_ps_likelihood_scores(model_out)
        # if metric == "confidence":
        #     scores = self.decoder.get_confidence_scores(model_out)

        return scores

    def forward_samples(self, decoder_out: Tuple[Tensor, Dict], n_samples=1,
                        mixture_out: Optional[Tensor] = None) -> Tensor:
        return self.decoder.sample_output(decoder_out, n_samples)

    # legacy
    @torch.no_grad()
    def get_normalized_probs(
            self,
            net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
            log_probs: bool,
            sample: Optional[Dict[str, Tensor]] = None,
            metric: Optional[str] = "cosine",
    ):
        return self.decoder.get_normalized_probs(net_output, log_probs, sample, metric)


@dataclass
class MixtureContinuousTransformerConfig(ContinuousTransformerConfig):
    num_mixture_components: int = field(
        default=2,
        metadata={"help": "number of mixture components for the distribution"},
    )
    mixture_dropout: float = field(
        default=0,
        metadata={"help": "dropout for the mixture head"},
    )
    output_parallel: bool = field(default=False,
                                  metadata={"help": "output each mixture component with separate layer"}
                                  )
    mixture_clusters_init_path: Optional[str] = field(default=None,
                                                      metadata={"help": "path to mixture head weights"}
                                                      )
    freeze_mixture_clusters: Optional[bool] = field(default=True,
                                                    metadata={"help": "freeze mixture head weights during training"})


@register_model("continuous_transformer_mixture", dataclass=MixtureContinuousTransformerConfig)
class TransformerMixtureModelContinuous(TransformerModelContinuous):

    def __init__(self, cfg, encoder, decoder):
        super().__init__(cfg, encoder, decoder)
        if cfg.num_mixture_components == 1:
            logger.error("Running continuous_transformer_mixture with 1 mixture component is not recommended! "
                         "Use continuous_transformer instead.")
            raise ValueError("num_mixture_components must be >1 for continuous_transformer_mixture")

    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):
        decoder = MixtureContinuousTransformerDecoder(
            cfg,
            tgt_dict,
            embed_tokens
        )
        return decoder

    def forward_decoder_scores(self, decoder_out: Tuple[Tensor, Dict],
                               metric: str = "ps",
                               reduce: Optional[str] = "max",
                               mixture_temperature: Optional[float] = 1.0) -> Tensor:
        assert metric is not None

        model_out = decoder_out[0]
        mixtures_out = output_with_temperature_scriptable(decoder_out[1]["mixtures_out"], mixture_temperature)
        scores = self.decoder.get_mixture_output(model_out, mixtures_out, metric, reduce)
        return scores

    def forward_samples(self, decoder_out: Tuple[Tensor, Dict], n_samples=1,
                        mixture_out: Optional[Tensor] = None) -> Tensor:
        return self.decoder.sample_mixture_and_output(decoder_out, n_samples, mixture_out)

