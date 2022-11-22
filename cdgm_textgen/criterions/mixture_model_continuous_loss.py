import power_spherical
import torch
import torch.nn.functional as F
from fairseq.criterions import register_criterion, FairseqCriterion

from cdgm_textgen.criterions import ARPowerSphericalRevKLCriterionConfig, ARPowerSphericalRevKLCriterion, \
    ARPowerSphericalCriterion, ARPowerSphericalCriterionConfig


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

        # apply padding mask
        target = target[mask]
        kappa = kappa[mask]
        out = out[mask]

        p, q = self.set_dist(out, target, kappa)
        z_samples = q.rsample()

        modes_out = mixture_out[mask]
        modes_probs = F.softmax(modes_out, dim=-1) #prior
        #uniform_probs=torch.full(modes_out.size(), 1.0/modes_out.shape[-1], device=modes_out.device)

        p_out = p.log_prob(z_samples.transpose(0, 1))
        q_out = q.log_prob(z_samples)
        loss = torch.sum(modes_probs * (q_out - p_out.t()), dim=-1)

        #calculate posterior
        with torch.no_grad():
            lmodes_probs = torch.log(modes_probs)+q_out
            lmodes_probs = lmodes_probs - torch.logsumexp(lmodes_probs, dim=-1)
            print(lmodes_probs)

        loss = loss.mean(dim=-1)

        return loss, {"kappa": kappa.mean().detach()}


@register_criterion("ar_mixture_ps_loss", dataclass=ARPowerSphericalCriterionConfig)
class MixtureModelCriterion(ARPowerSphericalCriterion):
    def compute_loss(self, model, net_output, sample, reduce=True):
        target = model.get_targets(sample, net_output)
        mask = sample["target"].ne(self.padding_idx)

        modes_out = net_output[1]["modes_out"]

        modes_probs = F.log_softmax(modes_out, dim=-1)

        # here out is BxTxD*num_modes, reshape it to BxTxMxD
        out = net_output[0]
        bsz = out.size(0)
        dim = model.decoder.output_embed_dim
        num_modes = model.decoder.num_mixture_components
        out = out.view(bsz, -1, num_modes, dim)

        # need to extend target to match output shape
        target = target.repeat(1, 1, num_modes).view(bsz, -1, num_modes, dim)

        kappa = torch.norm(out, dim=-1)
        out = F.normalize(out, dim=-1)

        # apply padding mask
        target = target[mask]
        kappa = kappa[mask]
        out = out[mask]
        modes_probs = modes_probs[mask]

        # get PS probs for each component
        d_x = power_spherical.PowerSpherical(out, kappa)

        p_out = d_x.log_prob(target)

        # sum over components
        loss = torch.logsumexp(modes_probs + p_out, dim=-1)
        loss = -loss.mean() + (self.kappa_lambda * torch.pow(kappa, 2)).mean()
        logging = {"kappa": kappa.mean().detach()}
        return loss, logging
