#!/bin/bash
#SBATCH --account=rrg-dprecup
#SBATCH --ntasks=1
#SBATCH --mem=30000M
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kushal.arora@mail.mcgill.ca
#SBATCH --time=23:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
###########################

set -ex

source activate lm_wn

export emb_size="${emb_size:=300}"
export wnhid="${wnhid:=100}"
export nhid="${nhid:=300}"
export distance="${distance:=pairwise}"
export task_emb_size=${emb_size}
export log_interval=${log_interval:=200}

cmd="python -u main.py --cuda --save-emb ${output_dir} --save ${output_dir} "

if [ -n "$log_interval" ]; then
   cmd+=" --log-interval ${log_interval} "
fi

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

if [ -n "$extend_wn" ]; then
    cmd+=" --extend_wn"
fi

if [ -n "$syn" ]; then
    cmd+=" -l syn"
    if [ -n "$extend_wn" ]; then
        task_emb_size=`expr ${task_emb_size} + ${wnhid}`
    fi
fi

if [ -n "$hyp" ]; then
    cmd+=" -l hyp"
    if [ -n "$extend_wn" ]; then
        task_emb_size=`expr ${task_emb_size} + ${wnhid}`
    fi
fi

if [ -n "$mer" ]; then
    cmd+=" -l mer"
    if [ -n "$extend_wn" ]; then
        task_emb_size=`expr ${task_emb_size} + ${wnhid}`
    fi
fi

if [ -n "$seg" ]; then
    cmd+=" --seg"
fi

if [ -n "$fixed_wn" ]; then
    cmd+=" --fixed_wn"
fi

if [ -n "$random_wn" ]; then
    cmd+=" --random_wn"
fi

if [ -n "$lower" ]; then
    cmd+=" --lower"
fi

if [ -n "$mdl" ]; then
    cmd+=" --model $mdl"
fi

$cmd

emb_filename=emb_${data}_${mdl}_${lexs}_${emb_size}_${nhid}_${wnhid}_${distance}

cd analogy_tasks;
python main.py  --sim-task --emb ../${output_dir}/${emb_filename}.pkl --vocab ../${output_dir}/vocab_${data}.pkl
python main.py  --analogy-task --emb ../${output_dir}/${emb_filename}.pkl --vocab ../${output_dir}/vocab_${data}.pkl

cd -;
export emb_filetxt=${output_dir}/${emb_filename}.txt
export bidaf_input_size=$(expr ${task_emb_size} + 100)
export ner_input_size=$(expr ${task_emb_size} + 128)
for task in ner sst esim bidaf
do
    task_file=$(mktemp ${output_dir}/${task}-${emb_filename}.XXXXXX)
    envsubst < ./extrinsic_tasks/local/${task}_template.jsonnet > ${task_file}
    allennlp train ${task_file} -s ${output_dir}/${task}/
    rm ${task_file}
done
