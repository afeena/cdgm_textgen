import math

import torch
from ledoh_torch import forward_sliced_dispersion, sliced_dispersion_gradient


class SlicedSphereDispersionFunction(torch.autograd.Function):
    """
    calculates forward and backward pass for sliced sphere despersion regularized
    """
    @staticmethod
    def forward(ctx, X, p, q):

        dist, hs = forward_sliced_dispersion(X, p, q, return_hidden_states=True)

        ctx.save_for_backward(hs["xp"], hs["xq"], p, q, hs["theta_diff"])

        return dist

    @staticmethod
    def backward(ctx, grad_output):
        """
        """
        Xp, Xq, p, q, theta_minus_thetastr = ctx.saved_tensors

        grad = sliced_dispersion_gradient(Xp, Xq, p, q, theta_minus_thetastr, grad_output)

        return grad, None, None

compute_sliced_sphere_dispersion = SlicedSphereDispersionFunction.apply