import argparse
import torch
import numpy as np
import power_spherical
from power_spherical import PowerSpherical, HypersphericalUniform
import torch.nn.functional as F

argp = argparse.ArgumentParser()
argp.add_argument("-n",dest="num_samples", type=int, default=1000000, help="Number of samples to generate")
argp.add_argument("-d",dest="dim", type=int, default=768, help="Dimension of the samples")
argp.add_argument("--dist", type=str, default="normal", choices=["uniform", "ps"], help="Type of distribution to use")
argp.add_argument("--ps_kappa",dest="ps_kappa", type=float, default=0, help="Kappa parameter for the power spherical distribution")
argp.add_argument("--output")

args = argp.parse_args()


synth_features = np.memmap(f"{args.output}_{args.num_samples}_synth_features_{args.dist}_d{args.dim}_kappa{args.ps_kappa}.npy", dtype=np.float32, mode="w+", shape=(args.num_samples, args.dim))
synth_labels = np.memmap(f"{args.output}_{args.num_samples}_synth_labels_{args.dist}_d{args.dim}_kappa{args.ps_kappa}.npy", dtype=np.int32, mode="w+", shape=(args.num_samples,))

if args.dist == "ps":
    dist = PowerSpherical(loc=F.normalize(torch.randn(args.dim).float(), dim=-1),
                             scale=torch.tensor([args.ps_kappa]))

elif args.dist == "uniform":
    dist = HypersphericalUniform(args.dim)
else:
    raise ValueError(f"Unknown distribution: {args.dist}")


with torch.no_grad():
    for i in range(args.num_samples):
        sample = dist.sample()
        synth_features[i] = sample.cpu().numpy()
        synth_labels[i] = i


