#!/bin/bash

set -ex

export lr=20
export optim=sgd
export epoch=100
export bptt=35
export bsize=20
export data=wikitext-2
export emb_size=300
export nhid=300
export wnhid=100
export distance=pairwise
export output_dir="output/${data}_${mdl}""$([[ $reg ]] && echo _reg || echo '')""$([[ $fixed_wn ]] && echo _fixed || echo '')""$([[ $random_wn ]] && echo _radom || echo '')""$([[ $seg ]] && echo _seg || echo '')""/$(date '+%Y_%m_%d_%H_%M')"

mkdir -p ${output_dir}
sbatch -t 23:00:00 -e ${output_dir}/std.out -o ${output_dir}/std.out scripts/launcher_wn.sh
#./scripts/launcher_wn.sh
