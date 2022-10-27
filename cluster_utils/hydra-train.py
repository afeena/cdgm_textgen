import json
import os
import yaml
import argparse
import stat
import subprocess

def get_experiment_number(models_path):
    import os
    import re

    occupied = sorted([int(x.split("_")[0]) for x in os.listdir(models_path)
                       if re.match("^\d+$", x.split("_")[0])])
    occupied = set(occupied)

    full = set([x for x in range(10000)])

    avaliable = full.difference(occupied)
    return avaliable.pop()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--fairseq-config", "-c", help="config file for the model")
    argparser.add_argument("--slurm-config", "-s", default="slurm_conf.json", help="config file for the slurm")
    argparser.add_argument("--train-dir", default=None, type=str, help="setup dir")
    argparser.add_argument("--extra", type=str, default=None, help="extra params to pass")

    args = argparser.parse_args()


    with open(args.slurm_config) as sc:
       slurm_config = json.load(sc)
       batch_conf = "\n".join(slurm_config["batch"])
       env = "\n".join(slurm_config["env"])
       models_dir = slurm_config["model_dir"]


    with open(args.fairseq_config) as cfg:
        model_config = yaml.load(cfg, Loader=yaml.FullLoader)

    arch = model_config["model"]["_name"]
    src = model_config["task"]["source_lang"]
    trg = model_config["task"]["target_lang"]

    saving_dir_name = args.train_dir
    if saving_dir_name is None:
        exp_number = get_experiment_number(models_dir)
        saving_dir_name = f"{models_dir}/{format(exp_number, '05d')}_{src}_{trg}_{arch}"
    else:
        exp_number = int(saving_dir_name.rsplit("/")[-1].split("_")[-1])

    try:
        os.makedirs(saving_dir_name)
    except FileExistsError as e:
        print("Directory exists. Delete or reset counter")
        exit(-1)

    out_file = f"#SBATCH -o {saving_dir_name}/%j_{src}_{trg}_{arch}.log"
    err_file = f"#SBATCH -e {saving_dir_name}/%j_{src}_{trg}_{arch}.err"
    job_name = f"#SBATCH -J {format(exp_number, '05d')}_{src}_{trg}_{arch}"


    config_dir = os.path.dirname(os.path.abspath(args.fairseq_config))
    extra_params_str = ""
    if args.extra is not None:
        extra_params = json.loads(args.extra)
        for k, v in extra_params.items():
            extra_params_str += f" {k}={v}"

    batch_sh_commands = f"{batch_conf}\n{out_file}\n{err_file}\n{job_name}\n{env}\n"

    fairseq_command = f"python ~/develop/cdgm_textgen/train.py \
    --config-dir {config_dir}  \
    --config-name {args.fairseq_config} \
    checkpoint.save_dir={saving_dir_name} \
    {extra_params_str}\n"

    sbatch_script = f"{saving_dir_name}/{exp_number}_train_batch_script.sh"
    with open(sbatch_script, "w") as scr:
        scr.write(batch_conf)
        scr.write("\n")
        scr.write(batch_sh_commands)
        scr.write("\n")
        scr.write(fairseq_command)

    st = os.stat(sbatch_script)
    os.chmod(sbatch_script, st.st_mode | stat.S_IEXEC)

    run_training_command = f"sbatch {sbatch_script}"

    process = subprocess.Popen(run_training_command.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    job_id = output.decode("utf-8").rsplit()[-1]
    print(f"Dir: {saving_dir_name}\nJob id: {job_id}")
