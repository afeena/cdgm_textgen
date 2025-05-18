import argparse
import torch
import numpy as np
from power_spherical import PowerSpherical
import torch.nn.functional as F


argp = argparse.ArgumentParser()
argp.add_argument("-n",dest="num_samples", type=int, default=1000000, help="Number of samples to generate")
argp.add_argument("-d",dest="dim", type=int, default=768, help="Dimension of the samples")
argp.add_argument("--n_mixtures",dest="n_mixtures", type=int, default=1, help="Number of mixtures to use")
argp.add_argument("--ps_kappa",dest="ps_kappa", type=float, default=0, help="Kappa parameter for the power spherical distribution")
argp.add_argument("--output")

args = argp.parse_args()


synth_features = np.memmap(f"{args.output}_{args.num_samples}_synth_features_{args.dist}_d{args.dim}_kappa{args.ps_kappa}.npy", dtype=np.float32, mode="w+", shape=(args.num_samples, args.dim))
synth_labels = np.memmap(f"{args.output}_{args.num_samples}_synth_labels_{args.dist}_d{args.dim}_kappa{args.ps_kappa}.npy", dtype=np.int32, mode="w+", shape=(args.num_samples,))

n_mixtures = args.n_mixtures
dim = args.dim
ps_kappa = args.ps_kappa

mix = torch.distributions.Categorical(torch.ones(n_mixtures,))
locs = F.normalize(torch.randn(n_mixtures,dim).float(), dim=-1)
scales = torch.zeros(n_mixtures,).float() + ps_kappa
ps = PowerSpherical(loc = locs, scale=scales)

with torch.no_grad():
    for i in range(args.num_samples):
        comp = mix.sample().item()
        sample = ps.sample()[comp] * torch.FloatTensor(1).uniform_(1,100).item()
        synth_features[i] = sample.cpu().numpy()
        synth_labels[i] = i


