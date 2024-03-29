#!/bin/bash
#SBATCH -N 1
#SBATCH -n 30
#SBATCH --mem=256G
#SBATCH -p short
#SBATCH -o log_%x_%j.out
#SBATCH -e log_%x_%j.err
#SBATCH -t 02:30:00
#SBATCH --mail-user=kchin@wpi.edu
#SBATCH --mail-type=fail,end

# $1: argument for path to working directory to change to, relative to the current directory
# $2: argument for path to the top-level directory containing all the sim data to combine into one VisualizationDataGroup object, relative to $1
# $3: argument for the absolute path to the multi_agent_sim_dynamic_no_qt.sif file
# $4: argument for path to the converted VisualizationDataGroup save location and name, relative to $1

# Load required modules
module load singularity

# Change to working directory (this is so that the log files are in the correct folders)
cd $1

# Run conversion
singularity exec $3 convert_exp_data_to_viz_data_group.py $2 -s $4