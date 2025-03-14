#!/bin/bash
#
## usage:  sbatch ./job_array_vsc4.sh
#
#SBATCH -J array
#SBATCH -N 1
#SBATCH -p skylake_0096
#SBATCH --qos=skylake_0096
#SBATCH --array=1-10

echo "Hi, this is array job number"  $SLURM_ARRAY_TASK_ID

sleep  $SLURM_ARRAY_TASK_ID


