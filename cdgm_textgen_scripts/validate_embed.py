import sys

from utils import txt_to_pt

vocab, emb = txt_to_pt(sys.argv[1])

print(emb.norm(dim=-1))
print(emb.shape)

