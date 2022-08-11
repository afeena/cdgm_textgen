import json
import os
import yaml
import argparse
import stat
import subprocess
import datetime
import time

batch_conf = """#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --exclude=ilps-cn111,ilps-cn108
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=3-12
"""

env = """export PATH=/home/etokarc/develop/torch-env/bin:$PATH
source /home/etokarc/develop/torch-env/bin/activate
"""


def get_experiment_number(models_path):
    import os
    import re

    occupied = sorted([int(x.split("_")[-1]) for x in os.listdir(models_path)
                       if re.match("^\d+$", x.split("_")[-1])])
    occupied = set(occupied)

    full = set([x for x in range(10000)])

    avaliable = full.difference(occupied)
    return avaliable.pop()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config", "-c", help="config file for the model")
    argparser.add_argument("--excl-node", default=None, type=str, help="exclude specific node")
    argparser.add_argument("--model-dir", default="/ivi/ilps/personal/etokarc/models/translation", type=str)
    argparser.add_argument("--train-dir", default=None, type=str, help="setup dir")
    argparser.add_argument("--exp", type=int, default=0, help="experiment number")
    argparser.add_argument("--extra", type=str, default=None, help="extra params to pass")

    args = argparser.parse_args()

    with open(args.config) as cfg:
        model_config = yaml.load(cfg, Loader=yaml.FullLoader)

    arch = model_config["model"]["_name"]
    src = model_config["task"]["source_lang"]
    trg = model_config["task"]["target_lang"]

    dir_name = args.train_dir
    if dir_name is None:
        exp_number = get_experiment_number(args.model_dir)
        dir_name = f"{args.model_dir}/{src}_{trg}_{arch}_{format(exp_number, '04d')}"
        try:
            os.makedirs(dir_name)
        except FileExistsError as e:
            print("Model dir exists, check your counter state!")
    else:
        try:
            exp_number = int(dir_name.rsplit("/")[-1].split("_")[-1])
        except ValueError:
            exp_number = args.exp

    out_file = f"#SBATCH -o {dir_name}/{src}_{trg}_{arch}-%j.log"
    err_file = f"#SBATCH -e {dir_name}/{src}_{trg}_{arch}-%j.err"
    job_name = f"#SBATCH -J {src}_{trg}_{arch}_{format(exp_number, '04d')}"

    config_dir = os.path.dirname(os.path.abspath(args.config))
    extra_params_str = ""
    if args.extra is not None:
        extra_params = json.loads(args.extra)
        for k, v in extra_params.items():
            extra_params_str += f" {k}={v}"

    fairseq_command = f"python ~/cdgm_textgen/train.py --config-dir {config_dir}  --config-name {args.config} checkpoint.save_dir={dir_name} {extra_params_str}\n"

    batch_conf = f"{batch_conf}\n{out_file}\n{err_file}\n{job_name}\n{env}\n"

    with open("tmp_train_batch_script.sh", "w") as scr:
        scr.write(batch_conf)
        if args.excl_node is not None:
            scr.write(f"#SBATCH --exclude={args.excl_node}\n")
        scr.write("\n")
        scr.write(fairseq_command)

    st = os.stat('tmp_train_batch_script.sh')
    os.chmod('tmp_train_batch_script.sh', st.st_mode | stat.S_IEXEC)

    bashCommand = "sbatch tmp_train_batch_script.sh"

    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
