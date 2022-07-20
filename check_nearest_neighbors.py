# coding: utf8

import torch
import sys
def txt_to_pt(embd_path):
    embd_tmp = []
    vocab = {}
    i=0
    with open(embd_path, encoding='utf-8') as embd_f:
        next(embd_f)
        for line in embd_f:
            pieces = line.rstrip().split(" ")
            vocab[pieces[0]]=i
            vals = torch.tensor([
                [float(weight) for weight in pieces[1:]]]
            )
            embd_tmp.append(vals)
            i+=1

    orig_embed = torch.cat(embd_tmp,0)
    return vocab, orig_embed

def main():
    #special_tokens = []
    #vocab = {v: i for i, v in enumerate(special_tokens)}
#    inv_vocab = {i: v for i, v in enumerate(special_tokens)}
#    with open(sys.argv[1]) as f:
#        for k, w in enumerate(f):
#            w = w.strip().rsplit()[0]
#            if w not in vocab:
#                vocab[w] = len(vocab)
#                inv_vocab[vocab[w]] = w

    vocab, X_orig = txt_to_pt(sys.argv[1])
    inv_vocab = {ind : w for w, ind in vocab.items() }
    

    def print_ix(topk):
        for i, w in zip(topk.indices, topk.values):
            i = i.item()
            print(f"{w:02.2f} {inv_vocab[i]}")

    def knns(ix, k=15):

        # euclidean KNN original

        eucl_orig = ((X_orig - X_orig[ix].unsqueeze(0))
            .norm(dim=-1)
            .topk(k, largest=False))

        print("Euclidean original")
        print_ix(eucl_orig)

        cos_orig = (torch.cosine_similarity(X_orig, X_orig[ix].unsqueeze(0))
            .topk(k, largest=True))

        print("Cosine original")
        print_ix(cos_orig)


    knns(vocab[sys.argv[2]])




if __name__ == '__main__':
    main()

