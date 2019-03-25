#!/bin/bash

set -ex


export mdl=${mdl:="rnn"}

if [ "${data}" == "wikitext2" ]; then
    export epoch="${epoch:=40}"
    export bptt="${bptt:=35}"
    export data="wikitext-2"
    export nhid="${nhid:=300}"
    time="${time:=23:00:00}"
    export mem="${mem:=30000}"
fi

if [ "${data}" == "wikitext103" ]; then
    export epoch="${epoch:=10}"
    export bptt="${bptt:=50}"
    export data="wikitext-103"
    export nhid="${nhid:=1200}"
    export adaptive=true
    # export nce=true
    export time="${time:=1-10:00:00}"
    export mem="${mem:=257000M}"
fi

if [ "${mdl}" == "retro" ]; then
    export epoch="${epoch:=40}"
    export bptt="${bptt:=1}"
    export data=${data:=glove}
    export bsize=${bsize:=6000}
    export lr=${lr:=2}
    export optim="${optim:=adagrad}"
    export time="${time:=3:00:00}"
    export mem="${mem:=30000}"
    export data_version=${data_version:=2}
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


job_name="${data}_${mdl}_${lexs}"
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
# account="${account:=rpp-bengioy}"
export account="${account:=rrg-dprecup}"
export mode="${mode:=slurm}"

mkdir -p ${output_dir}

if [ "${mode}" == "slurm" -a "${data}" == "wikitext-103" ]; then
    sbatch -J "${job_name}" -A ${account} -t ${time} -e ${output_dir}/std.out -o ${output_dir}/std.out --nodes=1 --gres=gpu:1 --mem 0 scripts/launcher_wn.sh
elif [ "${mode}" == "slurm" ]; then
    sbatch -J "${job_name}" -A ${account} -t ${time} -e ${output_dir}/std.out -o ${output_dir}/std.out --mem ${mem} --gres=gpu:1 scripts/launcher_wn.sh
else
    ./scripts/launcher_wn.sh
fi
