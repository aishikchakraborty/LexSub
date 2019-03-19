#!/bin/bash
#SBATCH --account=rrg-dprecup
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=chakraba@mila.quebec
#SBATCH --time=10:00:00
###########################

python -u main.py --data ../data/wikitext-2 --model skipgram --batch-size 4000 --bptt 35 --lower --max-pair 15 --ss_t 5e-6 --version 2
python -u main.py --data ../data/wikitext-103 --model skipgram --batch-size 4000 --bptt 35 --lower --max-pair 15 --ss_t 5e-6 --version 2
