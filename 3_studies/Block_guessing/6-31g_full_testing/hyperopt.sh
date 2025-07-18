#!/bin/bash

#SBATCH -J hyperopt_full
#SBATCH -c 16
#SBATCH -N 1                  # Ensure it runs on a single node
#SBATCH --partition=local     # use partition that fits to the qos
#SBATCH --mem=48G

# clean /tmp
find /tmp -maxdepth 1 -type f -user "$(whoami)" -delete

cd "/home/ewachmann/REPOS/Masterarbeit/3_studies/Block_guessing/6-31g_full_testing"

module purge
module load python/3.11.9-gcc-13.2.0-eqozuli miniconda3/latest
eval "$(conda shell.bash hook)"
conda activate scf_tools_311
echo "using python: $( python --version ) from $( which python )"

export PYSCF_TMPDIR=~/scratch/pyscf_tmp
mkdir -p $PYSCF_TMPDIR
echo 'Setting PYSCF_TMPDIR to '$PYSCF_TMPDIR

echo 
echo 'Hello from node: '$HOSTNAME
echo 'Number of nodes: '$SLURM_JOB_NUM_NODES
echo 'Tasks per node:  '$SLURM_TASKS_PER_NODE
echo 'Partition used:  '$SLURM_JOB_PARTITION
# echo 'QOS used:        '$SLURM_JOB_QOS
echo 'Using the nodes: '$SLURM_JOB_NODELIST
echo 

python hyperopt.py hyp_small_full_full_mat_loss.py