import logging
from dataclasses import dataclass, field

from cdgm_textgen.models import register_model
from cdgm_textgen.models.transformer.decoder_hybrid import HybridTransformerDecoder
from cdgm_textgen.models.transformer.transformer_continuous import ContinuousTransformerConfig, \
    TransformerModelContinuous

logger = logging.getLogger("cdgm_textgen.models.transformer.transformer_hybrid")


@dataclass
class HybridTransformerConfig(ContinuousTransformerConfig):
    stop_grads_to_target_emb: bool = field( default=True,
        metadata={"help": "stop gradients to target embeddings"},
    )

@register_model("hybrid_transformer", dataclass=HybridTransformerConfig)
class TransformerModelHybrid(TransformerModelContinuous):
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):
        assert cfg.target_embed_path is not None
        logger.info(f"loading target embeddings from {cfg.target_embed_path}...")
        target_embed = cls.build_embedding(
            cfg, tgt_dict, cfg.decoder.output_dim, cfg.decoder.target_embed_path
        )
        decoder = HybridTransformerDecoder(
            cfg,
            tgt_dict,
            embed_tokens,
            target_embed
        )
        return decoder

    def get_targets_grad(self, sample, net_output, **kwargs):
        return self.decoder.target_embed(sample["target"])