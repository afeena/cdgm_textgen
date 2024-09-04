import argparse
import json

import torch
from cdgm_textgen.models.transformer.roberta import RobertaSphEmbeddingsModel
from geoopt.optim import RiemannianAdam

from ledoh_torch import (
    KernelSphereDispersion,
    KernelSphereSemibatchDispersion,
    LloydSphereDispersion,
    AxisAlignedBatchSphereDispersion,
    SlicedSphereDispersion,
    MMADispersion,
    minimum_acos_distance as minimum_acos_distance,
    circular_variance
)

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--model-path", type=str, required=True)
arg_parser.add_argument("--lr", type=float, default=0.1)
arg_parser.add_argument("--niters", type=int, default=1000)
arg_parser.add_argument("--dispersion", type=str, default=None)
arg_parser.add_argument("--dispersion-params", type=str)

args = arg_parser.parse_args()

dispersion_params = json.loads(args.dispersion_params) if args.dispersion_params is not None else {}
dispersion_func = {
    "mmd": KernelSphereSemibatchDispersion(**dispersion_params),
    "lloyd": LloydSphereDispersion(**dispersion_params),
    "axis": AxisAlignedBatchSphereDispersion(**dispersion_params),
    "sliced": SlicedSphereDispersion(**dispersion_params),
}

loss_fn = dispersion_func[args.dispersion]

device = "cuda" if torch.cuda.is_available() else "cpu"
model = RobertaSphEmbeddingsModel.from_pretrained(args.model_path).to(device)

for pn, p in model.named_parameters():
    if "embed" not in pn:
        print(pn)
        p.requires_grad = False
    else:
        print(f"Parameter {pn} is added for optimization, {p.requires_grad}")

opt = RiemannianAdam([p for p in model.parameters() if p.requires_grad==True], lr=args.lr)
for i in range(args.niters):
    opt.zero_grad()
    loss = loss_fn(mo)
    loss.backward()
    opt.step()

    minimum_acos_distance(model, model)
    print(f"iter {i} loss {loss.item()}")
