# DGMs for text generation

##Requirements
This repository contains the extension of [fairseq](https://github.com/facebookresearch/fairseq) for training continuous
seq2seq models.
Please follow the installation instruction of `fairseq`.

Mixture model uses `functorch` for efficient computations, but it's only compatible with torch>=1.13

## Quick Start

```bash

git clone https://github.com/ltl-uva/cdgm_textgen/
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
- Continuous CMLM Transformer Model `cont_cmlm_transformer` (deprecated, old model)

## Supported losses:

- NLL with Von Mises-Fisher (vMF) Distribution
- NLL with Power Spherical (PS) Distribution
- KL divergence with PS distribution
- KL divergence between mixture of PS and PS distribution

## Continuous Generator

`task._name = translation_cont` allows for generation with cosine similarity and ps likelihood.
`ContinuousSequenceGenerator` class implements

- beam search
- beam search for mixture models with several option for mixture selection `reduce_strategy`:
    - `sum`: get likelihood as a sum over all mixture components
    - `max`: choose mixture with the highest probability
    - `None`, compatible with `mixture_beam` only

## Citations

[On Target Representation in Continuous-output Neural Machine Translation](https://aclanthology.org/2022.repl4nlp-1.24) (
Tokarchuk & Niculae, RepL4NLP 2022)

