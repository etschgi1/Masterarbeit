#!/bin/bash
#SBATCH -J cpu_stress_test      # Job name
#SBATCH -c 16                   # Number of CPU cores to use
#SBATCH -N 1                    # Run on a single node
#SBATCH --partition=local        # Adjust partition as needed
#SBATCH --mem=4G                 # Allocate 4GB memory
#SBATCH --time=10:00             # Set a time limit (10 minutes)

# Load necessary modules (if needed)
module purge

# Print CPU info for debugging
echo "Running on node: $HOSTNAME"
echo "Allocated cores: $SLURM_CPUS_PER_TASK"

# Run stress test for n cores
stress-ng --cpu 16 --timeout 60 --metrics-brief


echo "CPU Stress test completed!"
