from pathlib import Path
from typing import Tuple

import numpy as np


paths = {
    "en": "/ivi/ilps/projects/cnmt/data/ro-en/mbart-pp/fairseq-bin-spm-disjoint/dict.en.txt",
} # separated by space

def read_mt_vocab_with_freq(vocab_fn, special_tokens=()):
    """
    read fairseq vocabulary file
    add special tokens for translation task
    """
    if not special_tokens:
        special_tokens = ["<s>", "<pad>", "</s>", "<unk>"]

    vocab = {v: i for i, v in enumerate(special_tokens)}
    idx2word = {i: v for i, v in enumerate(special_tokens)}
    frequencies = np.array([1 for _ in special_tokens])

    with open(vocab_fn, 'r', encoding="utf-8") as d:
        lines = d.readlines()
        for line in lines:
            token, field = line.rstrip().rsplit(" ", 1)
            if token in special_tokens:
                continue
            vocab[token] = len(vocab)
            idx2word[len(idx2word)] = token
            frequencies = np.append(frequencies, np.array([int(field)]))

    for i in range(len(special_tokens)):
        frequencies[i] = max(frequencies)

    return vocab, idx2word, frequencies


def load_en():
    return read_mt_vocab_with_freq(
        Path(paths["en"])
    )