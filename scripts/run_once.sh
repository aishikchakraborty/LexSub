#!/bin/bash

set -ex


export mdl=${mdl:="rnn"}

if [ "${data}" == "wikitext2" ]; then
    export epoch="${epoch:=100}"
    export bptt="${bptt:=35}"
    export data=wikitext-2
    export nhid="${nhid:=300}"
    time=${time:=23:00:00}
    mem=${mem:=30000}
fi

if [ "${data}" == "wikitext103" ]; then
    export epoch="${epoch:=9}"
    export bptt="${bptt:=50}"
    export data=wikitext-103
    export nhid="${nhid:=1200}"
    time=${time:=2-23:00:00}
    mem=${mem:=120000}
    export adaptive=true
fi

if [ "${mdl}" == "retro" ]; then
    export epoch="${epoch:=40}"
    export bptt="${bptt:=1}"
    export data=${data:=glove}
    export bsize=${bsize:=512}
    export optim=${optim:="adam"}
    export lr=${lr:=0.01}
    time=${time:=24:00:00}
    mem=${mem:=30000}
fi

export lr="${lr:=20}"
export optim="${optim:=sgd}"
export bsize="${bsize:=20}"
export lower=true

export lmdl=""
if [ -n "$syn" ]; then
    export lmdl="syn"
fi

if [ -n "$hyp" ]; then
    export lmdl=${lmdl}"_hyp"
fi

if [ -n "$mer" ]; then
    export lmdl=${lmdl}"_mer"
fi

if [ -n "$vanilla" ] || [ "$lmdl" == "" ]; then
    export lmdl="vanilla"
fi


dir="output/${data}_${mdl}_${lmdl}"
dir=${dir}"$([[ $reg ]] && echo _reg || echo '')"
dir=${dir}"$([[ $fixed_wn ]] && echo _fixed || echo '')"
dir=${dir}"$([[ $random_wn ]] && echo _radom || echo '')"
dir=${dir}"$([[ $seg ]] && echo _seg || echo '')"
dir=${dir}"$([[ $lower ]] && echo _lower || echo '')"
dir=${dir}"$([[ $extend_wn ]] && echo _extend || echo '')"
dir=${dir}"/$(date '+%Y_%m_%d_%H_%M')"

export output_dir=${output_dir:=$dir}
account="${account:=rpp-bengioy}"

mkdir -p ${output_dir}
sbatch -A ${account} -t ${time} -e ${output_dir}/std.out -o ${output_dir}/std.out --mem ${mem} scripts/launcher_wn.sh
#./scripts/launcher_wn.sh
