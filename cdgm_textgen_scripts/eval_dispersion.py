import argparse
import torch
from ledoh_torch import minimum_cosine_distance, circular_variance

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path")

    args = parser.parse_args()

    model = torch.load(args.model_path)
    print(model.keys())

    embed = model["model"]["decoder.embed_tokens.weight"]

    md = minimum_cosine_distance(embed)
    cv = circular_variance(embed)
    print(f"min dist: {md}, circular variance: {cv}")