from dataclasses import field, dataclass
from typing import Optional

from fairseq.models.transformer import TransformerConfig

import logging
from cdgm_textgen.models import register_model
from cdgm_textgen.models.transformer import TransformerModelBase


logger = logging.getLogger("cdgm_textgen.models.transformer.riemann_transformer")
@dataclass
class RiemannianTransformerConfig(TransformerConfig):
    riemannian_enc_inp: Optional[bool] = field(
        default=True,
        metadata={"help": "use RiemannianAdam for encoder input embeddings"},
    )
    riemannian_dec_inp: Optional[bool] = field(
        default=True,
        metadata={"help": "use RiemannianAdam for decoder input embeddings"},
    )
    riemannian_dec_out: Optional[bool] = field(
        default=True,
        metadata={"help": "use RiemannianAdam for decoder output embeddings"},
    )


@register_model("riemannian_transformer", dataclass=RiemannianTransformerConfig)
class RiemannianTransformerModel(TransformerModelBase):
    def __init__(self, cfg, encoder, decoder):
        super().__init__(cfg, encoder, decoder)


    @classmethod
    def build_model(cls, cfg, task):
        model = super().build_model(cfg, task)
        if cfg.share_all_embeddings:
            assert cfg.riemannian_enc_inp==cfg.riemannian_dec_inp==cfg.riemannian_dec_out,\
                "if share_all_embeddings, then all embeddings must use RiemannianAdam"
        if cfg.share_decoder_input_output_embed:
            assert cfg.riemannian_dec_inp==cfg.riemannian_dec_out, "if share_decoer_input_output_embed, " \
                                                                   "then both embeddings must use RiemannianAdam"
        #here set the embed to the optimizer group of riemannian adam
        for pn, p in model.named_parameters():
            if cfg.riemannian_enc_inp and (pn.endswith("encoder.embed_tokens") or "encoder.embed_tokens" in pn):
                p.param_group = "riemannian"
            if cfg.riemannian_dec_inp and (pn.endswith("decoder.embed_tokens") or "decoder.embed_tokens" in pn):
                p.param_group = "riemannian"
            if cfg.riemannian_dec_out and (pn.endswith("output_projection") or "output_projection" in pn):
                p.param_group = "riemannian"
            else:
                p.param_group = "default"
        return model