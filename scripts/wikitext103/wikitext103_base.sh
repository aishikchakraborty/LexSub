#!/bin/bash

set -ex

export lr=20
export optim=sgd
export epoch=9
export bptt=50
export bsize=20
export data=wikitext-103
export emb_size=300
export nhid=1200
export wnhid=100
export distance=pairwise
export output_dir="output/${data}_${mdl}""$([[ $reg ]] && echo _reg || echo '')""/$(date '+%Y_%m_%d_%H_%M')"
export adaptive=true

mkdir -p ${output_dir}
sbatch -t 2-23:00:00 -e ${output_dir}/std.out -o ${output_dir}/std.out --mem 90000M scripts/launcher_wn.sh
#./scripts/launcher_wn.sh
