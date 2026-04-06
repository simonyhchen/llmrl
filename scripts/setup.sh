#!/bin/bash

# Setup script for LLMRL project
# This script clones AutoCkt, creates the conda environment, and installs additional dependencies

echo "Setting up LLMRL project..."

# Clone AutoCkt if not exists
if [ ! -d "autockt/AutoCkt" ]; then
    echo "Cloning AutoCkt..."
    git clone https://github.com/ksettaluri6/AutoCkt autockt/AutoCkt
else
    echo "AutoCkt already exists, skipping clone."
fi

# Create conda environment
echo "Creating conda environment..."
conda env create -f autockt/AutoCkt/environment.yml

# Activate environment and install additional requirements
echo "Activating environment and installing additional packages..."
conda activate autockt
pip install -r requirements.txt

echo "Setup complete! Activate the environment with 'conda activate autockt'."