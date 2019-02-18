#!/bin/bash

set -ex

export lr="${lr:=20}"
export optim="${optim:=sgd}"
export epoch="${epoch:=100}"
export bptt="${bptt:=35}"
export bsize="${bsize:=20}"
export data=wikitext-2
export nhid="${nhid:=300}"
export seg=True
export output_dir=${output_dir:="output/${data}_${mdl}""$([[ $reg ]] && echo _reg || echo '')""$([[ $fixed_wn ]] && echo _fixed || echo '')""$([[ $random_wn ]] && echo _radom || echo '')""$([[ $seg ]] && echo _seg || echo '')""$([[ $lower ]] && echo _lower || echo '')""/$(date '+%Y_%m_%d_%H_%M')"}
time=${time:=23:00:00}

mkdir -p ${output_dir}
sbatch -t ${time} -e ${output_dir}/std.out -o ${output_dir}/std.out --mem 30000M scripts/launcher_wn.sh
#./scripts/launcher_wn.sh
