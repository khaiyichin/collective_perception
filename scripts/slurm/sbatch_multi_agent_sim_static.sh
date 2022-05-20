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
singularity/3.6.2

./hpc_execute_multi_agent_sim.sh $1 $2