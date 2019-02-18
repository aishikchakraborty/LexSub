#!/bin/bash

set -ex

export lr="${lr:=20}"
export optim="${optim:=sgd}"
export epoch="${epoch:=9}"
export bptt="${bptt:=50}"
export bsize="${bsize:=20}"
export data=wikitext-103
export nhid="${nhid:=1200}"
export adaptive=true
export seg=True
export output_dir=${output_dir:="output/${data}_${mdl}""$([[ $reg ]] && echo _reg || echo '')""$([[ $fixed_wn ]] && echo _fixed || echo '')""$([[ $random_wn ]] && echo _radom || echo '')""$([[ $seg ]] && echo _seg || echo '')""$([[ $lower ]] && echo _lower || echo '')""/$(date '+%Y_%m_%d_%H_%M')"}

mkdir -p ${output_dir}
time=${time:=2-23:00:00}
mem=${mem:=120000}
sbatch -t ${time} -e ${output_dir}/std.out -o ${output_dir}/std.out --mem ${mem} scripts/launcher_wn.sh
#./scripts/launcher_wn.sh
