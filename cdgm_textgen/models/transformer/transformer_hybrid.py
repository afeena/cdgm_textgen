import logging
from dataclasses import dataclass, field

from cdgm_textgen.models import register_model
from cdgm_textgen.models.transformer.decoder_hybrid import HybridTransformerDecoder
from cdgm_textgen.models.transformer.transformer_continuous import ContinuousTransformerConfig, \
    TransformerModelContinuous

logger = logging.getLogger("cdgm_textgen.models.transformer.transformer_hybrid")


@register_model("hybrid_transformer", dataclass=ContinuousTransformerConfig)
class TransformerModelHybrid(TransformerModelContinuous):
    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):
        assert cfg.target_embed_path is not None
        logger.info(f"loading target embeddings from {cfg.target_embed_path}...")
        target_embed = cls.build_embedding(
            cfg, tgt_dict, cfg.decoder.output_dim, cfg.target_embed_path
        )
        decoder = HybridTransformerDecoder(
            cfg,
            tgt_dict,
            embed_tokens,
            target_embed=target_embed
        )
        return decoder
    @classmethod
    def build_model(cls, cfg, task):
        model = super().build_model(cfg, task)
        #here set the target_embed to not the optimizer group of riemannian adam
        for pn, p in model.named_parameters():
            if pn.endswith("target_embed") or "target_embed" in pn:
                p.param_group = "riemannian"
            else:
                p.param_group = "default"
        return model
    def get_targets(self, sample, net_output, **kwargs):
        return self.decoder.target_embed(sample["target"])