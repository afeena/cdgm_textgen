import sys

from cdgm_textgen_scripts import txt_to_pt

vocab, emb = txt_to_pt(sys.argv[1])

print(emb.norm())
print(emb.shape)

