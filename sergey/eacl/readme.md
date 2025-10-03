# env and run
srun -c 2 --partition gpu_h100 --gres=gpu:1  -n 1 --mem-per-gpu 64GB -t 1:00:00 --pty bash
cd /home/stroshin/cdgm_textgen
conda activate /projects/gus20642/evgeniia/conda-torch-py3.9
bash sergey/eacl/run_baseline.sh

# copy the data
rsync -rva /projects/gus20642/evgeniia/models/ /projects/gus20642/sergey/models/