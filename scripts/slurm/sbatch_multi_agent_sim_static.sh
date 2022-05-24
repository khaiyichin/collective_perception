#!/bin/bash
#SBATCH -N 1
#SBATCH -n 20
#SBATCH --mem=128G
#SBATCH -p short
#SBATCH -o log%j.out
#SBATCH -e log%j.err
#SBATCH -t 24:00:00
#SBATCH --mail-user=kchin@wpi.edu
#SBATCH --mail-type=all

# Load required modules
module load singularity/3.6.2

# This sbatch script should be run in the directory containing the .sif file
# $1: argument for path to the directory containing all the bash scripts
# $2: argument for the type of communication network to simulate: "full", "ring", "line", "scale-free"

# Run simulation
DIR=$(pwd)
cd $1
./hpc_execute_multi_agent_sim.sh $DIR $2