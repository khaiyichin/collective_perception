#!/bin/bash
#SBATCH -N 1
#SBATCH -n 40
#SBATCH --mem=128G
#SBATCH -p short
#SBATCH -o log%j.out
#SBATCH -e log%j.err
#SBATCH -t 02:00:00
#SBATCH --mail-user=kchin@wpi.edu
#SBATCH --mail-type=all

# This sbatch script should be run in the directory containing the .sif file.

# $1: argument for path to the directory containing all the python scripts
# $2: argument for top-level directory containing all the sim data for a type of communication network
# $3: argument for VisualizationDataGroup save location

# Load required modules
module load singularity/3.6.2

# Copy required files
cp $1/convert_exp_data_to_viz_data_group.py .
cp $1/sim_modules.py .
cp $1/viz_modules.py .

# Run conversion
singularity exec multi_agent_sim_static.sif python3 convert_exp_data_to_viz_data_group.py $2 -s $3