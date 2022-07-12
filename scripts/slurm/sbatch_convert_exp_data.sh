#!/bin/bash
#SBATCH -N 1
#SBATCH -n 30
#SBATCH --mem=512G
#SBATCH -p short
#SBATCH -o log_%x_%j.out
#SBATCH -e log_%x_%j.err
#SBATCH -t 02:30:00
#SBATCH --mail-user=kchin@wpi.edu
#SBATCH --mail-type=all

# $1: argument for path to working directory to change to, relative to the current directory
# $2: argument for the absolute path to the directory containing all the python scripts
# $3: argument for path to the top-level directory containing all the sim data to combine into one VisualizationDataGroup object, relative to $1
# $4: argument for the absolute path to the multi_agent_sim_dynamic_no_qt.sif file
# $5: argument for path to the converted VisualizationDataGroup save location and name, relative to $1

# Load required modules
module load singularity/3.6.2

# Change to working directory (this is so that the log files are in the correct folders)
cd $1

# Copy required files
cp $2/convert_exp_data_to_viz_data_group.py .
cp $2/sim_modules.py .
cp $2/viz_modules.py .
cp $2/simulation_set_pb2.py .
cp $2/util_pb2.py .

# Run conversion
singularity exec $4 python3 convert_exp_data_to_viz_data_group.py $3 -s $5