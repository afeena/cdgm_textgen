import torch
from hyperspherical_vae.ops.ive import ive_fraction_approx2
import numpy as np

def _log_iv_approx(m, k):
    """
    m - dimension of the vector, torch.tensor
    k - concentration parameter, torch.tensor
    """
    eta = (m+0.5)/(2*(m+1))

    k_le_m = (m*torch.log(k)
             +(eta*k)
             -(eta+m)*np.log(2.0)
             -torch.lgamma(m+1.0)
             )

    k_geq_m = (k
             -0.5*torch.log(k)
             -0.5*np.log(2.0*np.pi)
             )
    res = torch.where(k < m, k_le_m, k_geq_m)
    

    return torch.where(k < m, k_le_m, k_geq_m)

def _neglogcmk(m, k):
    logcmk = (- (0.5*m - 1.0) * torch.log(k)
            + (0.5*m) * np.log(2.0 * np.pi)
            + _log_iv_approx(0.5*m-1.0  , k)
            )
    return logcmk



class NegLogCmkApproxFunction(torch.autograd.Function):
    """
    forward and backward approximation for C_m(k)
    """
    @staticmethod
    def forward(ctx, m, k):
        """
        forward pass returns the approximation of the log of the bessel function
        according to the https://arxiv.org/pdf/1902.02603.pdf
        """
        ctx.save_for_backward(k)
        ctx.m = m

        return _neglogcmk(m,k)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Note: ive_fraction_approx2 works with I_m(k)
        we need grad of neg log of C_m(k) which is
        - (I_m/2(k)/(I_m/2-1)
        to make it consistent with the forward function, ctx.m is multiplied with 0.5
        """
        k = ctx.saved_tensors[-1]
        grad_input = grad_output.clone()
        neg_log_ive_grad = ive_fraction_approx2(0.5*ctx.m, k)*-1.0
        return (
            None,
            grad_input * neg_log_ive_grad )

neg_log_cmk_approx = NegLogCmkApproxFunction.apply
