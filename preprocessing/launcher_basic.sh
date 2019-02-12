#!/bin/bash
#SBATCH --account=def-dprecup
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120000M
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kushal.arora@mail.mcgill.ca
#SBATCH --time=23:00:00
###########################

set -eux
$@
