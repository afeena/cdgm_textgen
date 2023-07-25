import torch

def txt_to_pt(embd_path):
    """
        load embeddings from file
        (emb are in a format that can be read by fairseq
        <num_of_embeddings> <embedding_dim>
        <token> <embedding>)
        """
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

def save_embed(emb, fname, dictionary):
    """
    save embeddings to file in a format that can be read by fairseq
    <num_of_embeddings> <embedding_dim>
    <token> <embedding>
    """
    with open(f"{fname}.txt", "w") as out:
        out.write("{} {}".format(emb.shape[0], emb.shape[1]))
        out.write("\n")

        for tok, ind in dictionary.items():
            vals = " ".join([str(x) for x in emb[ind, :].cpu().data.numpy()])
            out.write(f"{tok} {vals}\n")


def read_mt_vocab(vocab_fn, special_tokens=()):
    """
    read fairseq vocabulary file
    add special tokens for translation task
    """
    if not special_tokens:
        special_tokens = ["<s>", "<pad>", "</s>", "<unk>"]

    vocab = {v: i for i, v in enumerate(special_tokens)}

    with open(vocab_fn, 'r', encoding="utf-8") as d:
        lines = d.readlines()
        for line in lines:
            token, field = line.rstrip().rsplit(" ", 1)
            if token in special_tokens:
                continue
            vocab[token] = len(vocab)
    return vocab