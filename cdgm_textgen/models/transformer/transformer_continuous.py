import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List

import torch
from fairseq.models.transformer import TransformerConfig
from torch import Tensor

from cdgm_textgen.models import register_model
from cdgm_textgen.models.transformer.decoder_continuous import \
    ContinuousTransformerDecoder, \
    ProbaTransformerDecoder, \
    PseudoContinuousTransformerDecoder
from cdgm_textgen.models.transformer.mixture_decoder_continuous import MixtureContinuousTransformerDecoder
from cdgm_textgen.models.transformer.transformer_base import TransformerModelBase
from cdgm_textgen.modules.utils import output_with_temperature_scriptable

logger = logging.getLogger("cdgm_textgen.models.transformer.transformer_continuous")


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
    manifold: Optional[str] = field(default=None, metadata={"help": "manifold to use for output of the model"})


@register_model("continuous_transformer", dataclass=ContinuousTransformerConfig)
class TransformerModelContinuous(TransformerModelBase):
    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):
        assert cfg.target_embed_path is not None
        logger.info(f"loading target embeddings from {cfg.target_embed_path}...")
        target_embed = cls.build_embedding(
            cfg, tgt_dict, cfg.decoder.output_dim, cfg.target_embed_path
        )
        decoder = ContinuousTransformerDecoder(
            cfg,
            tgt_dict,
            embed_tokens,
            target_embed
        )

        decoder.target_embed.weight.requires_grad = False
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

    def set_normalized_tgt_embeddings(self):
        self.decoder.set_normalized_tgt_embeddings()

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

    def get_targets(self, sample, net_output, **kwargs):
        return self.decoder.target_embed(sample["target"]).detach()

    def forward_decoder_scores(self, decoder_out: Tuple[Tensor, Dict], metric: str) -> Tensor:
        assert metric is not None
        scores = None
        model_out = decoder_out[0]

        if metric == "cosine":
            scores = self.decoder.get_cosine_similarity_scores(model_out)

        return scores

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
class ContinuousTransformerScheduleConfig(ContinuousTransformerConfig):
    freeze_after_update: int = field(
        default=-1,
        metadata={"help": "freeze target embeddings after N number of update"},
    )
    freeze_before_update: int = field(
        default=-1,
        metadata={"help": "freeze target embeddings before N number of updates"},
    )
@register_model("continuous_transformer_schedule", dataclass=ContinuousTransformerScheduleConfig)
class TransformerModelContinuousSchedule(TransformerModelContinuous):
    def __init__(self, cfg, encoder, decoder):
        super().__init__(cfg, encoder, decoder)
        self.freeze_after_update = cfg.freeze_after_update
        self.freeze_before_update = cfg.freeze_before_update
        assert not (self.freeze_after_update>0 and self.freeze_before_update>0), "freeze_after_update and freeze_before_update " \
                                                                                 "cannot be both positive"

    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):
        assert cfg.target_embed_path is not None
        logger.info(f"loading target embeddings from {cfg.target_embed_path}...")
        target_embed = cls.build_embedding(
            cfg, tgt_dict, cfg.decoder.output_dim, cfg.target_embed_path
        )
        decoder = ContinuousTransformerDecoder(
            cfg,
            tgt_dict,
            embed_tokens,
            target_embed
        )
        return decoder
    def get_targets(self, sample, net_output, **kwargs):
        if self.num_updates < self.freeze_after_update or self.num_updates > self.freeze_before_update:
            return self.decoder.target_embed(sample["target"])
        else:
            return self.decoder.target_embed(sample["target"]).detach()
    def set_num_updates(self, num_updates):
        self.num_updates = num_updates
@register_model("continuous_transformer_trained", dataclass=ContinuousTransformerConfig)
class TransformerModelContinuousTrained(TransformerModelContinuous):
    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):
        assert cfg.target_embed_path is not None
        logger.info(f"loading target embeddings from {cfg.target_embed_path}...")
        target_embed = cls.build_embedding(
            cfg, tgt_dict, cfg.decoder.output_dim, cfg.target_embed_path
        )
        decoder = ContinuousTransformerDecoder(
            cfg,
            tgt_dict,
            embed_tokens,
            target_embed
        )

        return decoder
    def get_targets(self, sample, net_output, **kwargs):
        return self.decoder.target_embed(sample["target"])

@dataclass
class ProbaTransformerConfig(ContinuousTransformerConfig):
    kappa: str = field(default="norm", metadata={"help": "kappa to use"})
@register_model("proba_transformer", dataclass=ProbaTransformerConfig)
class TransformerModelProba(TransformerModelContinuous):
    def forward_kappa(self, decoder_out: Tuple[Tensor, Dict]) -> Tensor:
        return self.decoder.forward_kappa(decoder_out)

    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):
        assert cfg.target_embed_path is not None
        logger.info(f"loading target embeddings from {cfg.target_embed_path}...")
        target_embed = cls.build_embedding(
            cfg, tgt_dict, cfg.decoder.output_dim, cfg.target_embed_path
        )
        decoder = ProbaTransformerDecoder(
            cfg,
            tgt_dict,
            embed_tokens,
            target_embed
        )

        decoder.target_embed.weight.requires_grad = False
        return decoder

    def forward_decoder_scores(self, decoder_out: Tuple[Tensor, Dict], metric: str) -> Tensor:
        assert metric is not None
        scores = None
        model_out = decoder_out[0]

        scores = self.decoder.get_ps_likelihood_scores(model_out)
        return scores


@register_model("pseudo_cont_transformer", dataclass=ContinuousTransformerConfig)
class PseudoContTransformer(TransformerModelContinuous):
    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):
        assert cfg.target_embed_path is not None
        logger.info(f"loading target embeddings from {cfg.target_embed_path}...")
        target_embed = cls.build_embedding(
            cfg, tgt_dict, cfg.decoder.output_dim, cfg.target_embed_path
        )
        decoder = PseudoContinuousTransformerDecoder( cfg,
            tgt_dict,
            embed_tokens,
            target_embed)

        decoder.target_embed.weight.requires_grad = False
        return decoder

@dataclass
class MixtureContinuousTransformerConfig(ContinuousTransformerConfig):
    num_mixture_components: int = field(
        default=2,
        metadata={"help": "number of mixture components for the distribution"},
    )
    mixture_dropout: float = field(
        default=0.1,
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
    add_latent_z: Optional[bool] = field(default=False,
                                         metadata={"help": "add z_i to the output of decoder"})
    uniform_prior: Optional[bool] = field(default=False,
                                          metadata={"help": "use uniform prior for mixture components"})


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

    def forward_samples(self, decoder_out: Tuple[Tensor, Dict],
                        mixtures_out: Optional[Tensor] = None, n_samples=1, ) -> Tensor:
        return self.decoder.sample_output_tokens(decoder_out, mixtures_out, n_samples)

    def get_normalized_probs(
            self,
            net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
            log_probs: bool,
            sample: Optional[Dict[str, Tensor]] = None,
            metric: Optional[str] = "cosine",
            reduce_strategy: Optional[str] = "max",
    ):
        return self.decoder.get_normalized_probs(net_output, log_probs, sample,
                                                 metric,
                                                 reduce_strategy)
