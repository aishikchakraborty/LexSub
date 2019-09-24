#!/bin/bash
#SBATCH --account=rrg-dprecup
#SBATCH --ntasks=1
#SBATCH --mem=30000M
#SBATCH --mail-type=ALL
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:1
###########################

set -eux
echo $(date '+%Y_%m_%d_%H_%M') - $SLURM_JOB_NAME - $SLURM_JOBID - `hostname` - ${output_dir} >> ./lm_wn_task_machine_assignments.txt
$@
