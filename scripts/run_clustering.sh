#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=0-3

GEOMSTATS_BACKEND=pytorch python cluster_embeddings.py /ivi/ilps/projects/cnmt/tgt_training_emb/ro-en/mttransfer.spm.00062.d128.txt 128 256
