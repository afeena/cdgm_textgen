datadir=/projects/gus20642/evgeniia/wmt19-deen/fairseq-bin/
script=$1  # ~/cdgm_textgen/measure_collapse.py
basedir=$2
checkpoint=$3

mkdir -p $basedir/$checkpoint.collapse
command="/projects/gus20642/evgeniia/conda-torch-py3.9/bin/python $script $datadir --task translation --path $basedir/$checkpoint --max-tokens 4096 --valid-subset train --results-path $basedir/$checkpoint.collapse"
echo $command
$command
echo "Saved to $basedir/$checkpoint.collapse"ssss