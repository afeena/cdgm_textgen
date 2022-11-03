import torch
import torch.nn.functional as F
from fairseq.criterions import register_criterion, FairseqCriterion


from cdgm_textgen.criterions import ARPowerSphericalRevKLCriterionConfig, ARPowerSphericalRevKLCriterion

@register_criterion("power_spherical_klmc_mixture_loss", dataclass=ARPowerSphericalRevKLCriterionConfig)
class MixturePowerSphericalRevKLCriterion(ARPowerSphericalRevKLCriterion):
    def compute_loss(self, model, net_output, sample, reduce=True):
        target = model.get_targets(sample, net_output)
        mask = sample["target"].ne(self.padding_idx)
        out = net_output[0]
        mixture_out = net_output[1]["mixtures_out"]

        loss, logging = self.compute_kl(out, target, mask, mixture_out=mixture_out)

        return loss, logging

    def compute_kl(self, out, target, mask, mixture_out=None):

        kappa = torch.norm(out, dim=-1)
        out = F.normalize(out, dim=-1)

        #apply padding mask
        target = target[mask]
        kappa = kappa[mask]
        out = out[mask]

        p, q = self.set_dist(out, target, kappa)
        z_samples = q.rsample()

        modes_out = mixture_out[mask]
        modes_probs = F.softmax(modes_out, dim=-1)

        p_out = p.log_prob(z_samples.transpose(0,1))
        q_out = q.log_prob(z_samples)
        loss = torch.sum(modes_probs*(q_out-p_out.t()), dim=-1)

        loss = loss.mean(dim=-1)

        return loss, {"kappa": kappa.mean()}