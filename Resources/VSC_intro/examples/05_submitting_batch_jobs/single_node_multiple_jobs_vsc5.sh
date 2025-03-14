#!/bin/bash
#
## usage:  sbatch ./single_node_multiple_jobs_vsc5.sh
#
#SBATCH -J snglcre
#SBATCH -N 1
#SBATCH -p zen3_0512
#SBATCH --qos=zen3_0512


for ((i=1; i<=128; i+=5))
do
   echo "Hello from"$HOSTNAME
   stress --cpu 1 --timeout $i  &
done
wait  


