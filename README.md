# DGMs for text generation

## Quick Start
This repository contains the extension of [fairseq](https://github.com/facebookresearch/fairseq) for training continuous seq2seq models.
Please follow the installation instruction of `fairseq`.

You do not need to install any additional libraries. Just clone the repository

```bash

git clone https://github.com/afeena/cdgm_textgen/
cd cdgm_textgen

#train the model
python  ~/develop/cdgm_textgen/train.py --config-dir <config_dir> \
--config-name <config_name> <any additional hydra parameters>

#decode and obtain results

python decode.py path_to_data_folder --task name_task --source-lang src_lang --target-lang tgt_lang \
--path path_to_the_model_checkpoint

```

## Supported Models:
Setup the model name in hydra config as 
`model._name=model_name`. 

Avaliable continuous models:
- Continuous Transformer Model `continuous_transformer`
- Continuous Transformer Mixture model `continuous_transformer_mixture`
- Continuous CMLM Transformer Model `cont_cmlm_transformer`

## Supported losses:
- NLL with Von Mises-Fisher (vMF) Distribution
- NLL with Power Spherical (PS) Distribution
- KL divergence with PS distribution


## Citations

[On Target Representation in Continuous-output Neural Machine Translation](https://aclanthology.org/2022.repl4nlp-1.24) (Tokarchuk & Niculae, RepL4NLP 2022)

