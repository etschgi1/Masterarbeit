#!/bin/bash
#
## usage:  sbatch ./combined_array_multiple_jobs_vsc4.sh  
#
#SBATCH -J combi 
#SBATCH -N 1
#SBATCH -p skylake_0096
#SBATCH --qos=skylake_0096
#SBATCH --array=1-144:48 

j=$SLURM_ARRAY_TASK_ID
((j+=47))

for ((i=$SLURM_ARRAY_TASK_ID; i<=$j; i++))
do
   stress --cpu 1 --timeout $i  &
done
wait  


