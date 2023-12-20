import math

import torch
import torch.nn.functional as F


class SlicedSphereDispersionFunction(torch.autograd.Function):
    """
    calculates forward and backward pass for sliced sphere despersion regularized
    """
    @staticmethod
    def forward(ctx, X, p, q):
        N = X.size(0)
        device = X.device

        Xp = X @ p
        Xq = X @ q

        thetas = torch.arctan2(Xq, Xp)

        ix = torch.argsort(thetas)
        invix = torch.empty_like(ix)
        invix[ix] = torch.arange(N, device=device)

        phis = -math.pi - math.pi / N + (2 * math.pi *
                                                  torch.arange(1, N + 1,
                                                               device=device)
                                                  ) / N


        phis = phis[invix]
        thetas_star = torch.mean(thetas)+phis

        ctx.save_for_backward(Xp, Xq, p, q, thetas-thetas_star)

        return 0.5*torch.mean(torch.pow(thetas-thetas_star, 2))

    @staticmethod
    def backward(ctx, grad_output):
        """
        """
        Xp, Xq, p, q, theta_minus_thetastr = ctx.saved_tensors

        grad = (
                (Xp * q.unsqueeze(-1) - Xq * p.unsqueeze(-1)) / (Xp ** 2 + Xq ** 2)
        ) * theta_minus_thetastr

        return grad_output * grad.T, None, None

compute_sliced_sphere_dispersion = SlicedSphereDispersionFunction.apply