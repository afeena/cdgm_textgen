#!/bin/bash
#SBATCH -c 2
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH --mem-per-gpu=64GB
#SBATCH -t 1:00:00
#SBATCH -J run_metrics
#SBATCH -o logs/run_metrics_%j.out
#SBATCH -e logs/run_metrics_%j.err

# Run your Python script with the model directory as argument
folder=${1:-"00019_de_en_transformer_base"}
cd /home/stroshin/cdgm_textgen
/projects/gus20642/evgeniia/conda-torch-py3.9/bin/python sergey/eacl/run_subfolder.py ~/cdgm_textgen/measure_collapse.py /projects/gus20642/sergey/models/$folder
