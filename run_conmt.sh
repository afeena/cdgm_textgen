#!/bin/bash
#SBATCH -c 2
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH --mem-per-gpu=64GB
#SBATCH -t 1:00:00
#SBATCH -J run_metrics
#SBATCH -o logs/run_metrics_conmt_%j.out
#SBATCH -e logs/run_metrics_conmt_%j.err

# Run your Python script with the model directory as argument
folder=${1:-"00017_de_en_continuous_transformer"}
cd /home/stroshin/cdgm_textgen
/projects/gus20642/evgeniia/conda-torch-py3.9/bin/python sergey/eacl/run_subfolder.py measure_collapse_conmt.py /projects/gus20642/sergey/models/$folder
