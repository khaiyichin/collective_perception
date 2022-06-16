#!/bin/bash
#SBATCH -N 1
#SBATCH -n 20
#SBATCH --mem=128G
#SBATCH -p short
#SBATCH -o log_%x_%j.out
#SBATCH -e log_%x_%j.err
#SBATCH -t 24:00:00
#SBATCH --mail-user=kchin@wpi.edu
#SBATCH --mail-type=all

# This sbatch script should be run in the directory containing the .sif file
# $1: argument for path to the scripts/ directory

# Load required modules
module load singularity/3.6.2

# Copy required files
cp $1/examples/param/param_multi_agent_sim_dynamic.argos .
cp $1/bash/hpc_execute_multi_agent_sim_dynamic.sh .

# Run simulation
./hpc_execute_multi_agent_sim_dynamic.sh