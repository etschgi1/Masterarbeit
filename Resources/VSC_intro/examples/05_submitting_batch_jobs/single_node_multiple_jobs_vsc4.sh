#!/bin/bash
#
## usage:  sbatch ./single_node_multiple_jobs_vsc4.sh
#
#SBATCH -J snglcre
#SBATCH -N 1
#SBATCH -p skylake_0096
#SBATCH --qos=skylake_0096


for ((i=1; i<=48; i++))
do
   stress --cpu 1 --timeout $i  &
done
wait  


