#!/bin/bash

set -ex


export mdl=${mdl:="rnn"}

if [ "${data}" == "wikitext2" ]; then
    export epoch="${epoch:=70}"
    export bptt="${bptt:=35}"
    export data="wikitext-2"
    export nhid="${nhid:=300}"
    time="${time:=4:00:00}"
    export mem="${mem:=30000}"
fi

if [ "${data}" == "wikitext103" ]; then
    export epoch="${epoch:=9}"
    export bptt="${bptt:=50}"
    export data="wikitext-103"
    export nhid="${nhid:=1200}"
    export adaptive=true
    export time="${time:=2-23:00:00}"
    export mem="${mem:=90000}"
fi

if [ "${mdl}" == "retro" ]; then
    export epoch="${epoch:=40}"
    export bptt="${bptt:=1}"
    export data=${data:=glove}
    export bsize=${bsize:=512}
    export lr=${lr:=1}
    export optim="${optim:=adagrad}"
    export time="${time:=23:00:00}"
    export mem="${mem:=30000}"
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


dir="output/${data}_${mdl}_${lexs}"
dir=${dir}"$([[ $reg ]] && echo _reg || echo '')"
dir=${dir}"$([[ $fixed_wn ]] && echo _fixed || echo '')"
dir=${dir}"$([[ $random_wn ]] && echo _radom || echo '')"
dir=${dir}"$([[ $seg ]] && echo _seg || echo '')"
dir=${dir}"$([[ $lower ]] && echo _lower || echo '')"
dir=${dir}"$([[ $extend_wn ]] && echo _extend || echo '')"
dir=${dir}"/$(date '+%Y_%m_%d_%H_%M')"

export output_dir=${output_dir:=$dir}
#account="${account:=rpp-bengioy}"
account="${account:=rrg-dprecup}"

mkdir -p ${output_dir}
sbatch -A ${account} -t ${time} -e ${output_dir}/std.out -o ${output_dir}/std.out --mem ${mem} scripts/launcher_wn.sh
# ./scripts/launcher_wn.sh
