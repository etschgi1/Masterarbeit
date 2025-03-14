#!/bin/bash

#SBATCH -J Testjob
#SBATCH -c 4                  # Use 4 CPU cores
#SBATCH -N 1                  # Ensure it runs on a single node
#SBATCH --partition=local     # use partition that fits to the qos
#SBATCH --mem=4G

module purge
module load python/3.11.9-gcc-13.2.0-eqozuli miniconda3/latest
eval "$(conda shell.bash hook)"
conda activate scf_dev
"using python: $( python --version ) from $( which python )"

echo 
echo 'Hello from node: '$HOSTNAME
echo 'Number of nodes: '$SLURM_JOB_NUM_NODES
echo 'Tasks per node:  '$SLURM_TASKS_PER_NODE
echo 'Partition used:  '$SLURM_JOB_PARTITION
# echo 'QOS used:        '$SLURM_JOB_QOS
echo 'Using the nodes: '$SLURM_JOB_NODELIST
echo 

python test.py