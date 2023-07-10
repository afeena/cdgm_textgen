import argparse
import torch

def project_to_cube(source, target_dim, reduce_sum=False):
    if reduce_sum:
        tgt = source.view(source.shape[0], target_dim, -1).sum(dim=-1)
    else:
        tgt = source[:, :target_dim]
    tgt = torch.where(tgt > 0, 1.0, -1.0)
    print(tgt.unique(dim=0).shape == source.shape[0], tgt.unique(dim=0).shape, source.shape[0])
    return tgt


def save_cube_embed(cube, fname, dictionary):
    # cdgm_textgen reuires this format as an input for pre-trained embeddings
    with open(f"{fname}.txt", "w") as out:
        out.write("{} {}".format(cube.shape[0], cube.shape[1]))
        out.write("\n")

        for tok, ind in dictionary.items():
            vals = " ".join([str(x) for x in cube[ind, :].cpu().data.numpy()])
            out.write(f"{tok} {vals}\n")

def txt_to_pt(embd_path):
    embd_tmp = []
    vocab = {}
    i = 0
    with open(embd_path, encoding='utf-8') as embd_f:
        next(embd_f)
        for line in embd_f:
            pieces = line.rstrip().split(" ")
            vocab[pieces[0]] = i
            vals = torch.tensor([
                [float(weight) for weight in pieces[1:]]]
            )
            embd_tmp.append(vals)
            i += 1

    orig_embed = torch.cat(embd_tmp, 0)
    return vocab, orig_embed

if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("-s", "--source", type=str, required=True)
    argp.add_argument("-d", "--dim", type=int, required=True)
    argp.add_argument("-r", "--reducesum", action="store_true", default=False)
    dict = argp.add_argument("-v", "--vocab", required=True)
    save_fn = argp.add_argument("-t", "--target", required=True)

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
    parsed_voc, parsed_embd = txt_to_pt(args.source)
    assert parsed_voc == vocab
    cube = project_to_cube(source=parsed_embd, target_dim=args.dim, reduce_sum=args.reducesum)
    save_cube_embed(cube, args.target, vocab)
