#!/bin/bash
#
## usage:  sbatch ./combined_array_multiple_jobs_vsc5.sh  
#
#SBATCH -J combi 
#SBATCH -N 1
#SBATCH -p zen3_0512
#SBATCH --qos=zen3_0512
#SBATCH --array=1-384:128

j=$SLURM_ARRAY_TASK_ID
((j+=127))

for ((i=$SLURM_ARRAY_TASK_ID; i<=$j; i++))
do
   stress --cpu 1 --timeout $i  &
done
wait  


