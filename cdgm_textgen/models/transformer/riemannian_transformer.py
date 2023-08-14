from dataclasses import field, dataclass
from typing import Optional

from fairseq.models.transformer import TransformerConfig
from fairseq import utils
import logging
from cdgm_textgen.models import register_model
from cdgm_textgen.models.transformer import TransformerModelBase
from geoopt import SphereExact, ManifoldParameter
import torch.nn as nn

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
    manifold: Optional[bool] = field(
        default='sphere',
        metadata={"help": "use RiemannianAdam for decoder output embeddings"},
    )


@register_model("riemannian_emb_transformer", dataclass=RiemannianTransformerConfig)
class RiemannianEmbeddingsTransformerModel(TransformerModelBase):
    def __init__(self, cfg, encoder, decoder):
        super().__init__(cfg, encoder, decoder)
        if cfg.manifold=="sphere":
            self.manifold = SphereExact()
        else:
            raise NotImplementedError(f"manifold {cfg.manifold} not implemented")

    @classmethod
    def build_model(cls, cfg, task):
        """Build a new model instance."""
        model = super().build_model(cfg, task)
        #TODO:put parameters on manifold
        #here set the embed to the optimizer group of riemannian adam
        for pn, p in model.named_parameters():
            if cfg.all_riemannian:
                p = ManifoldParameter(p, manifold=model.manifold)
            else:
                if cfg.riemannian_enc_inp and (pn.endswith("encoder.embed_tokens") or "encoder.embed_tokens" in pn):
                    p.param_group = "riemannian"
                    p = ManifoldParameter(p, manifold=model.manifold)
                if cfg.riemannian_dec_inp and (pn.endswith("decoder.embed_tokens") or "decoder.embed_tokens" in pn):
                    p.param_group = "riemannian"
                    p = ManifoldParameter(p, manifold=model.manifold)
                if cfg.riemannian_dec_out and (pn.endswith("decoder.output_projection") or "decoder.output_projection" in pn):
                    p.param_group = "riemannian"
                    p = ManifoldParameter(p, manifold=model.manifold)
                else:
                    p.param_group = "default"
        return model