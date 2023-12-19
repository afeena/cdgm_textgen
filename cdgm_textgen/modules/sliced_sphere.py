import math

import torch
import torch.nn.functional as F


class SlicedSphereDispersionFunction(torch.autograd.Function):
    """
    calculates forward and backward pass for sliced sphere despersion regularized
    """
    @staticmethod
    def forward(ctx, X):

        d = X.size(-1)
        N = X.size(0)
        dtype = X.dtype
        device = X.device

        PQ = torch.randn(d, 2, requires_grad=False,
                                     dtype=dtype,
                                     device=device)

        PQ, R = torch.linalg.qr(PQ)
        PQ *= R.diagonal().sign()
        p, q = PQ[:, 0], PQ[:, 1]

        ctx.p = p
        ctx.q = q
        Xp = X @ p
        Xq = X @ q

        ctx.save_for_backward(Xp, Xq)

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

        ctx.thetas_star = thetas_star
        ctx.thetas = thetas

        return 0.5*torch.mean(torch.pow(thetas-thetas_star, 2))

    @staticmethod
    def backward(ctx, grad_output):
        """
        """
        #check this part!
        Xp, Xq = ctx.saved_tensors
        p = ctx.p
        q = ctx.q

        thetas_star = ctx.thetas_star
        thetas = ctx.thetas

        grad = (
                (Xp * q.unsqueeze(-1) - Xq * p.unsqueeze(-1)) / (Xp ** 2 + Xq ** 2)
        ) * (thetas - thetas_star)

        return grad_output * grad.T

compute_sliced_sphere_dispersion = SlicedSphereDispersionFunction.apply

# def main():
#     X = torch.randn(10, 2, requires_grad=True)
#     y = F.normalize(X, dim=-1)
#     res = compute_sliced_sphere_dispersion(y)
#     res.backward()
#     print(X.grad)
#
# if __name__ == "__main__":
#     main()