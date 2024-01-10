import argparse
import torch
from ledoh_torch import minimum_acos_distance, circular_variance

from cdgm_textgen_scripts import txt_to_pt

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, type=str, help="path to model checkpoint")
    parser.add_argument("--txt", default=None, type=str, help="path to txt file with sentences")

    args = parser.parse_args()

    if args.model is not None:
        model = torch.load(args.model_path)
        print(model.keys())

        embed = model["model"]["decoder.embed_tokens.weight"]
    elif args.txt is not None:
        vocab, embed = txt_to_pt(args.txt)
    else:
        print("Provide either model or txt")
        exit(-1)
    md = minimum_acos_distance(embed)
    cv = circular_variance(embed)
    print(f"min dist: {md}, circular variance: {cv}")