import glob
import os
import yaml
import argparse
import stat
import subprocess

batch_conf = """#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=05:00:00
#SBATCH --exclude=ilps-cn108
"""
env = """export PATH=/home/etokarc/develop/torch-env/bin:$PATH
source /home/etokarc/develop/torch-env/bin/activate
"""
interpreter = "/home/etokarc/develop/torch-env/bin/python3"
decode_script = "/home/etokarc/develop/fairseq-cont-nat/fairseq_cli/generate.py"

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config", "-c", help="config file for the model")
    argparser.add_argument("--exp", type=int, help="experiment number")
    argparser.add_argument("--excl-node", default=None, type=str, help="exclude specific node")
    argparser.add_argument("--tag", default=None, help="additional name tag")

    args = argparser.parse_args()

    with open(args.config) as cfg:
        model_config = yaml.load(cfg, Loader=yaml.FullLoader)

    model_dir = "/ivi/ilps/personal/etokarc/models/translation"
    exp_id = format(args.exp, '04d')
    dir_name = glob.glob(f"{model_dir}/*{exp_id}")
    print(dir_name)

    if len(dir_name) == 0:
        print("The experiment folder is not found!")
        exit(-1)
    else:
        dir_name = dir_name[0]
    eval_dir = f"{dir_name}/eval/"
    os.makedirs(eval_dir, exist_ok=True)

    src = model_config["decode_params"]["source-lang"]
    trg = model_config["decode_params"]["target-lang"]
    checkpoint_name = model_config["checkpoint"]
    out_file = f"#SBATCH -o {eval_dir}/{src}_{trg}_{checkpoint_name}-%j.log"
    err_file = f"#SBATCH -e {eval_dir}/{src}_{trg}_{checkpoint_name}-%j.err"
    job_name = f"#SBATCH -J {src}_{trg}_decode_{exp_id}_{checkpoint_name}"
    kv_params = ""
    for k, v in model_config["decode_params"].items():
        if k != "data":
            kv_params += f"--{k} {v} "

    " ".join(kv_params.split())

    data = model_config["data"]
    checkpoint_path = f"{dir_name}/{checkpoint_name}.pt"
    task = model_config["decode_params"]["task"]
    beam = model_config["decode_params"]["beam"]
    iters = ""
    decoding_m = ""
    tag = f".{args.tag}" if args.tag is not None else ""
    if "iter-decode-max-iter" in model_config["decode_params"]:
        it = model_config["decode_params"]["iter-decode-max-iter"]
        iters = f".it{it}"
    if "decoding-measure" in model_config["decode_params"]:
        decoding_m = ".dm-" + model_config["decode_params"]["decoding-measure"]

    batch_conf = f"{batch_conf}\n{out_file}\n{err_file}\n{job_name}"

    with open("tmp_decode_batch_script.sh", "w") as scr:
        scr.write(batch_conf)
        if args.excl_node is not None:
            scr.write(f"#SBATCH --exclude={args.excl_node}\n")
        scr.write("\n")
        scr.write(env)
        scr.write("\n")
        scr.write("MOSES=\"/home/etokarc/develop/mosesdecoder/\"\n")

    for k, v in model_config["inputs"].items():
        with open("tmp_decode_batch_script.sh", "a") as scr:
            current_set = f"--input {v}"
            hyp_file = f"{eval_dir}/{src}_{trg}_{checkpoint_name}_{k}_{task}_beam{beam}{iters}{decoding_m}{tag}.hyp"
            fairseq_command = f"python ~/develop/cdgm_textgen/decode.py {data} {kv_params} --path {checkpoint_path} {current_set} > {hyp_file}\n"
            scr.write(fairseq_command)
            eval_ready_file = f"{eval_dir}/{src}_{trg}_{checkpoint_name}_{k}_{task}_beam{beam}{iters}{decoding_m}{tag}.eval"
            eval_command = f"cat {hyp_file} | grep -P \"^H\" | sort -V | cut -f3- | $MOSES/scripts/recaser/detruecase.perl | $MOSES/scripts/tokenizer/detokenizer.perl -l {trg} > {eval_ready_file}\n"
            scr.write(eval_command)
            current_ref = model_config["refs"][k]
            blue_score_file = f"{eval_dir}/{src}_{trg}_{checkpoint_name}_{k}_{task}_beam{beam}{iters}{decoding_m}{tag}.sacrebleu"
            blue_score_command = f"cat {eval_ready_file} | sacrebleu {current_ref} > {blue_score_file}\n"
            bert_score_file = f"{eval_dir}/{src}_{trg}_{checkpoint_name}_{k}_{task}_beam{beam}{iters}{decoding_m}{tag}.bert"
            bert_score_command = f"bert-score -r {current_ref} -c {eval_ready_file}  --lang {trg} --rescale_with_baseline > {bert_score_file}\n"
            scr.write(blue_score_command)
            scr.write(bert_score_command)
            scr.write("\n")

    st = os.stat('tmp_decode_batch_script.sh')
    os.chmod('tmp_decode_batch_script.sh', st.st_mode | stat.S_IEXEC)

    bashCommand = "sbatch tmp_decode_batch_script.sh"

    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

    print(output.decode("utf-8").split()[-1], error)
