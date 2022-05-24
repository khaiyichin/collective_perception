#!/bin/bash
#SBATCH -N 1
#SBATCH -n 20
#SBATCH --mem=128G
#SBATCH -p short
#SBATCH -o log%x%j.out
#SBATCH -e log%x%j.err
#SBATCH -t 24:00:00
#SBATCH --mail-user=kchin@wpi.edu
#SBATCH --mail-type=all

# This sbatch script should be run in the directory containing the .sif file
# $1: argument for path to the scripts/ directory
# $2: argument for the type of communication network to simulate: "full", "ring", "line", "scale-free"

# Load required modules
module load singularity/3.6.2

# Copy required files
cp $1/examples/param/param_multi_agent_sim.yaml .
cp $1/python/multi_agent_sim.py .
cp $1/python/sim_modules.py .
cp $1/bash/hpc_execute_multi_agent_sim.sh

# Run simulation
./hpc_execute_multi_agent_sim.sh $2