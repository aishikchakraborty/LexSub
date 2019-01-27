#!/bin/bash
#SBATCH --account=rrg-dprecup
#SBATCH --ntasks=8
#SBATCH --mem=30000M
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kushal.arora@mail.mcgill.ca
#SBATCH --time=23:00:00
#SBATCH --gres=gpu:1
###########################

set -ex

cmd="python -u main.py --cuda --save-emb ${output_dir} --save ${output_dir}"
if [ -n "$lr" ]; then
    cmd+=" --lr $lr"
fi

if [ -n "$epoch" ]; then
    cmd+=" --epoch $epoch"
fi

if [ -n "$bptt" ]; then
    cmd+=" --bptt $bptt"
fi

if [ -n "$bsize" ]; then
    cmd+=" --batch-size $bsize"
fi

if [ -n "$patience" ]; then
    cmd+=" --patience $patience"
fi

if [ -n "$mdl" ]; then
    cmd+=" --mdl $mdl"
fi

if [ -n "$emb_size" ]; then
    cmd+=" --emsize $emb_size"
fi

if [ -n "$nhid" ]; then
    cmd+=" --nhid $nhid"
fi

if [ -n "$wnhid" ]; then
    cmd+=" --wn_hid $wnhid"
fi

if [ -n "$data" ]; then
    cmd+=" --data $data"
fi

if [ -n "$optim" ]; then
    cmd+=" --optim $optim"
fi

if [ -n "$distance" ]; then
    cmd+=" --distance $distance"
fi

if [ -n "$reg" ]; then
    cmd+=" --reg"
fi

if [ -n "$tied" ]; then
    cmd+=" --tied"
fi

if [ -n "$adaptive" ]; then
    cmd+=" --adaptive"
fi

$cmd

emb_filename=emb_${data}_${mdl}_${emb_size}_${nhid}_${wnhid}_${distance}

cd analogy_tasks;
python main.py  --sim-task --emb ../${output_dir}/${emb_filename}.pkl --vocab ../vocab_${data}.pkl

cd -;
export emb_filetxt=${output_dir}/${emb_filename}.txt

for task in sst esim # bidaf
do
    task_file=$(mktemp ${output_dir}/${task}-${emb_filename}.XXXXXX)
    envsubst < ./extrinsic_tasks/local/${task}_template.jsonnet > ${task_file}
    allennlp train ${task_file} -s ${output_dir}/${task}/
    rm ${task_file}
done