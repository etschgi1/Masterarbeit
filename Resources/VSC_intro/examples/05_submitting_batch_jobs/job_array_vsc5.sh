#!/bin/bash
# 
## usage:  sbatch ./job_array_vsc5.sh
#
#SBATCH -J array
#SBATCH -N 1
#SBATCH -p zen3_0512
#SBATCH --qos=zen3_0512
#SBATCH --array=1-20:5

echo "Hi, this is array job number"  $SLURM_ARRAY_TASK_ID

sleep  $SLURM_ARRAY_TASK_ID

