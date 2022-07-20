import torch
from power_spherical import PowerSpherical

from fairseq.criterions.continuous_loss import \
    PowerSphericalRevKLTaylorApproxCriterion

def main():

    # torch.set_default_dtype(torch.float64)

    b = 2
    d = 512
    mu_p = torch.randn(b, d)
    mu_q = torch.randn(b, d)

    mu_p = torch.nn.functional.normalize(mu_p, dim=-1)
    mu_q = torch.nn.functional.normalize(mu_q, dim=-1)

    kappa_p = 100.0 * torch.ones(b)
    kappa_q = 1300.0 * torch.ones(b)

    p = PowerSpherical(mu_p, kappa_p, validate_args=True)
    q = PowerSpherical(mu_q, kappa_q, validate_args=True)

    crit = PowerSphericalRevKLTaylorApproxCriterion(
        task=None,
        loss_type=None,
        kappa_lambda=None,
        n_samples=0,
        kldirection="forward",
        target_kappa_scale=1
    )

    Xq = q.rsample((1000,))
    # print(Xq.shape)
    # 1000 x 2 x 512

    # approximate KL
    kl_pointwise = q.log_prob(Xq) - p.log_prob(Xq)
    print(kl_pointwise.mean(dim=0))
    print(crit.kl(p, q))


if __name__ == '__main__':
    main()

