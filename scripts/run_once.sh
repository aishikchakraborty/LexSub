#!/bin/bash

set -ex

export 	LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

export mdl=${mdl:="rnn"}

if [ "${data}" == "wikitext2" ]; then
    export data="wikitext-2"
    export nhid="${nhid:=300}"
    export mem="${mem:=30000}"

    if [ "${mdl}" == "skipgram" ]; then
        time="${time:=1-00:00:00}"
        export epoch="${epoch:=40}"
    elif [ "${mdl}" == "rnn" ]; then
        export epoch="${epoch:=40}"
        export bptt="${bptt:=70}"
        export bsize="${bsize:=40}"
        export data="wikitext-2"
        export rnn_type="${rnn_type:=QRNN}"
        export data_version="${data_version:=2}"
        time="${time:=3:00:00}"
    fi

fi

if [ "${data}" == "wikitext103" ]; then
    export data="wikitext-103"
    export nhid="${nhid:=1200}"
    export adaptive=true

    if [ "${mdl}" == "skipgram" ]; then
        export time="${time:=2-00:00:00}"
        export epoch="${epoch:=10}"
        export mem="${mem:=257000M}"
    elif [ "${mdl}" == "rnn" ]; then
        export epoch="${epoch:=10}"
        export bptt="${bptt:=140}"
        export bsize="${bsize:=30}"
        export mem="${mem:=50000M}"
        export rnn_type="${rnn_type:=QRNN}"
        export vocab_size="${vocab_size:=100000}"
        export data_version=${data_version:=2}
        export time="${time:=1-00:00:00}"
    fi
fi

if [ "${mdl}" == "retro" ]; then
    export epoch="${epoch:=40}"
    export bptt="${bptt:=1}"
    if [ "${data}" == "wikitext103" ]; then
        export data="wikitext-103"
    elif [ "${data}" == "glove" ]; then
        export data="glove"
    else
        export data=${data:=glove}
    fi

    export bsize=${bsize:=5000}
    export lr=${lr:=2}
    export optim="${optim:=adagrad}"
    export time="${time:=3:00:00}"
    export mem="${mem:=30000}"
    export data_version=${data_version:=2}
    export log_interval=${log_interval:=10}
fi

export lr="${lr:=20}"
export optim="${optim:=sgd}"
export bsize="${bsize:=20}"
export lower=true

lexs_arr=()
if [ -n "$syn" ]; then
    lexs_arr+=("syn")
fi

if [ -n "$hyp" ]; then
    lexs_arr+=("hyp")
fi

if [ -n "$mer" ]; then
    lexs_arr+=("mer")
fi

lexs_tmp=$(IFS=_; echo "${lexs_arr[*]}")

export lexs=${lexs_tmp}
if [ -n "$vanilla" ] || [ "$lexs" == "" ]; then
    export lexs="vanilla"
fi


job_name="${data}_${mdl}_${lexs}_${syn_ratio}_${hyp_ratio}_${mer_ratio}"
job_name=${job_name}"$([[ $reg ]] && echo _reg || echo '')"
job_name=${job_name}"$([[ $fixed_wn ]] && echo _fixed || echo '')"
job_name=${job_name}"$([[ $random_wn ]] && echo _radom || echo '')"
job_name=${job_name}"$([[ $seg ]] && echo _seg || echo '')"
job_name=${job_name}"$([[ $lower ]] && echo _lower || echo '')"
job_name=${job_name}"$([[ $extend_wn ]] && echo _extend || echo '')"
job_name=${job_name}"$([[ $data_version ]] && echo "_wn_v${data_version}" || echo '')"
export output_dir_prefix=${output_dir_prefix:="output"}
dir="${output_dir_prefix}/${job_name}/""${date_suffix:=$(date '+%Y_%m_%d_%H_%M')}"

export output_dir=${output_dir:=$dir}
export account="${account:=rpp-bengioy}"
# export account="${account:=rrg-dprecup}"
export mode="${mode:=slurm}"

mkdir -p ${output_dir}

if [ "${mode}" == "slurm" ] && [ "${data}" == "wikitext-103" ] && [ "${mdl}" == "skipgram" ]; then
    sbatch -J "${job_name}" -A ${account} -t ${time} -e ${output_dir}/std.out -o ${output_dir}/std.out --nodes=1 --gres=gpu:1 --mem 0 scripts/launcher_wn.sh
elif [ "${mode}" == "slurm" ]; then
    sbatch -J "${job_name}" -A ${account} -t ${time} -e ${output_dir}/std.out -o ${output_dir}/std.out --mem ${mem} --gres=gpu:1 scripts/launcher_wn.sh
else
    ./scripts/launcher_wn.sh
fi
