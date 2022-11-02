# DGMs for text generation

## Installation
This repository contains the extension of [fairseq](https://github.com/facebookresearch/fairseq) for training continuous seq2seq models.
Please follow the installation instruvtion of `fairseq`

## Supported models:
- Continuous Transformer Model
- Continuous CMLM Transformer Model

## Supported losses:
- NLL with Von Mises-Fisher (vMF) Distribution
- NLL with Power Spherical (PS) Distribution
- KL divergence with PS distribution

## Decoding

Decoding is based on gready search with cosine similarity


## Run models using Hydra
This library supports running `training` through hydra:

```bash
python  ~/develop/cdgm_textgen/train.py --config-dir <config_dir> \
--config-name <config_name> <any additional hydra parameters>

```
`decoding` can be done as:
```bash
python decode.py path_to_data_folder --task name_task --source-lang src_lang --target-lang tgt_lang \
--path path_to_the_model_checkpoint

```

## Citations

[On Target Representation in Continuous-output Neural Machine Translation](https://aclanthology.org/2022.repl4nlp-1.24) (Tokarchuk & Niculae, RepL4NLP 2022)

