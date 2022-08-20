#!/bin/bash
#SBATCH -N 1
#SBATCH -n 40
#SBATCH --mem=128G
#SBATCH -p short
#SBATCH -o log_%x_%j.out
#SBATCH -e log_%x_%j.err
#SBATCH -t 24:00:00
#SBATCH --mail-user=kchin@wpi.edu
#SBATCH --mail-type=fail,end

# To run this you need `param_multi_agent_sim_dynamic.argos` in the sane directory as `hpc_execute_multi_agents_sim_dynamic.sh`.

# $1: argument for path to working directory to change to, relative to the current directory
# $2: argument for path to the directory containing hpc_execute_multi_agent_sim_dynamic.sh, relative to $1
# $3: argument for the absolute path to the multi_agent_sim_full_no_qt.sif file
# $4: argument for path to the directory for writing output data, relative to $1

# Load required modules
module load singularity

# Change to working directory (this is so that the log files are in the correct folders)
cd $1

# Run simulation
$2/hpc_execute_multi_agent_sim_dynamic.sh $2/param_multi_agent_sim_dynamic.argos $3 $4