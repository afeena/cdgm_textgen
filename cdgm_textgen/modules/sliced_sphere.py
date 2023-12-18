import math

import torch
import torch.nn.functional as F


class SlicedSphereDispersionFunction(torch.autograd.Function):
    """
    calculates forward and backward pass for sliced sphere despersion regularized
    """

    @staticmethod
    def forward(ctx, weight):

        ctx.save_for_backward(weight)
        emb_dim = weight.size(-1)
        dtype = weight.dtype
        device = weight.device

        PQ = F.normalize(torch.randn(2, emb_dim,
                                     dtype=dtype,
                                     device=device),
                         dim=-1)
        PQ, _ = torch.linalg.qr(PQ.T)
        p, q = PQ[:, 0], PQ[:, 1]

        ctx.p = p
        ctx.q = q
        thetas = torch.arctan2(weight @ p, weight @ q)

        vocab_size = weight.size(0)
        ix = torch.argsort(thetas)
        invix = torch.empty_like(ix)
        invix[ix] = torch.arange(vocab_size, device=device)

        phis = -math.pi - math.pi / vocab_size + (2 * math.pi *
                                                  torch.arange(1, vocab_size + 1,
                                                               device=device)
                                                  ) / vocab_size
        phis = phis[invix]

        # riemannian projection:
        return 0.5 * torch.mean(torch.pow(thetas - phis, 2))

    @staticmethod
    def backward(ctx, grad_output):
        """
        """
        weight = ctx.saved_tensors[-1]
        p = ctx.p
        q = ctx.q

        return grad_output * (
            ((weight @ p) * q - (weight @ q) * p)/ ((weight @ p)**2+(weight @ q)**2)
        )

compute_sliced_sphere_dispersion = SlicedSphereDispersionFunction.apply