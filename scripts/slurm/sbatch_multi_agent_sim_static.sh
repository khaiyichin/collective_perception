#!/bin/bash
#SBATCH -N 1
#SBATCH -n 24
#SBATCH --mem=256G
#SBATCH -p short
#SBATCH -o log_%x_%j.out
#SBATCH -e log_%x_%j.err
#SBATCH -t 24:00:00
#SBATCH --mail-user=kchin@wpi.edu
#SBATCH --mail-type=all

# To run this you need `param_multi_agent_sim_static.yaml` in the working directory.

# $1: argument for path to working directory to change to, relative to the current directory
# $2: argument for path to the directory containing hpc_execute_multi_agent_sim_dynamic.sh, relative to $1
# $3: argument for the absolute path to the multi_agent_sim_static.sif file
# $4: argument for the type of communication network to simulate: "full", "ring", "line", "scale-free"

# Load required modules
module load singularity/3.6.2

# Change to working directory (this is so that the log files are in the correct folders)
cd $1

# Run simulation
./$2/hpc_execute_multi_agent_sim_static.sh $2/param_multi_agent_sim_static.yaml $3 $4