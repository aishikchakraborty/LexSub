#!/bin/bash
#SBATCH --account=def-jcheung
#SBATCH --ntasks=1
#SBATCH --mem=30G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=chakraba@mila.quebec
#SBATCH --time=10:00:00
###########################

python -u main.py --data ../data/wikitext-103 --model skipgram --batch-size 4000 --bptt 35 --lower --ss_t 1e-3
