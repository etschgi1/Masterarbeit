#!/bin/bash

# Exit script on error
set -e

# Define environment name
ENV_NAME="scf_dev"

# Check if the conda environment exists and remove it
if conda env list | grep -q "^$ENV_NAME\s"; then
    echo "Removing existing conda environment: $ENV_NAME"
    conda env remove --name $ENV_NAME
fi

# Create a new Conda environment
echo "Creating new conda environment: $ENV_NAME"
conda create --name $ENV_NAME python=3.11 -y

# Activate the environment
echo "Activating conda environment: $ENV_NAME"
source activate $ENV_NAME

# Install conda-build
echo "Installing conda-build"
conda install conda-build -y

# Build the package
echo "Building package from recipe"
conda build --channel conda-forge --channel pyscf recipe

# Install the package from local build
echo "Installing package into $ENV_NAME"
conda install --channel conda-forge --channel pyscf --use-local scf_guess_tools -y

echo "Setup complete!"
