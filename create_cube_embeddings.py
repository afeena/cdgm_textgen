import argparse
import torch


def build_cube_embed(num_emb, embed_dim, permute=False, stretch=False):
    if permute:
        d = torch.randint(0, 2 ** (embed_dim - 1), (num_emb * 2,)).unique()[:num_emb]
    else:
        step = 1
        if stretch:
            # stretch embeddings in the cube
            step = (2 ** (embed_dim - 1)) // num_emb
        d = torch.arange(0, num_emb) * step
    mask = 2 ** torch.arange(embed_dim - 1, -1, -1).to(d.device, d.dtype)
    d = d.unsqueeze(-1).bitwise_and(mask).ne(0).float()
    d[d == 0] = -1

    return d


def save_cube_embed(cube, fname, dictionary):
    # cdgm_textgen reuires this format as an input for pre-trained embeddings
    with open(f"{fname}.txt", "w") as out:
        out.write("{} {}".format(cube.shape[0], cube.shape[1]))
        out.write("\n")

        for tok, ind in dictionary.items():
            vals = " ".join([str(x) for x in cube[ind, :].cpu().data.numpy()])
            out.write(f"{tok} {vals}\n")


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("-d", "--dim", type=int, required=True)
    permute = argp.add_argument("-r", "--rand", action="store_true", default=False)
    stretch = argp.add_argument("--stretch", action="store_true", default=False)
    dict = argp.add_argument("-v", "--vocab", required=True)
    save_fn = argp.add_argument("-s", "--save", required=True)

    args = argp.parse_args()
    print(args)

    special_tokens = ["<s>", "<pad>", "</s>", "<unk>"]

    vocab = {v: i for i, v in enumerate(special_tokens)}

    with open(args.vocab, 'r', encoding="utf-8") as d:
        lines = d.readlines()
        for line in lines:
            token, field = line.rstrip().rsplit(" ", 1)
            if token in special_tokens:
                continue
            vocab[token] = len(vocab)
    num_emb = len(vocab)
    cube = build_cube_embed(num_emb, args.dim, permute=args.permute)
    save_cube_embed(cube, args.save, vocab)
