#!/bin/bash

# Script to execute static multi agent simulation (Python-based)
# on the HPC cluster, which means that the Python script is run
# in an Apptainer container. The apptainer name is assumed to be
# multi_agent_sim_static.sif.

# Verify that arguments are provided
if [ $# == 0 ] || [ $# == 1 ]; then
    echo "Not enough arguments provided!"
    exit 1
fi

if [[ ! -d $1 ]]; then
    echo "1st argument should be a working directory that exists!"
    exit 1
fi

s1="full"
s2="ring"
s3="line"
s4="scale-free"

if [ $2 != $s1 ] && [ $2 != $s2 ] && [ $2 != $s3 ] && [ $2 != $s4 ]; then
    echo "2nd argument should be one of the following: \"full\", \"ring\", \"line\", \"scale-free\"!"
    exit 1
fi

# Switch to working directory
CURR_DIR=$(pwd)
echo "Changing to working directory: $1";
pushd $1

# Copy scripts
cp $CURR_DIR/../examples/param/param_multi_agent_sim.yaml .
cp $CURR_DIR/../python/multi_agent_sim.py .
cp $CURR_DIR/../python/sim_modules.py .

# Iterate and run scripts for different param cases
# COMM=("full" "ring" "line" "scale-free")
COMM=$2 # when run in the cluster different communication graph parameters are run with separate nodes for increased efficiency
PERIOD=(1 2 5 10)
AGENTS=(5 10 50 100 200)

MIN=(0.05)
MAX=(0.95)
INC=(19)

# Set fixed parameters
sed -i "s/numExperiments:.*/numExperiments: 5/g" param_multi_agent_sim.yaml
sed -i "s/numObs:.*/numObs: 1000/g" param_multi_agent_sim.yaml

# Run simulations
{
    START_TIME=$(date +%m/%d/%Y-%H:%M:%S)

    echo -e "\n################################### EXECUTION BEGIN ###################################"
    echo -e "################################# ${START_TIME} #################################\n"

    sed -i "s/type:.*/type: \"$COMM\"/g" param_multi_agent_sim.yaml # communication network graph type

    for (( b = 0; b <= 3; b++ )) # comms period
    do
        period=$(echo ${PERIOD[b]})
        sed -i "s/commsPeriod:.*/commsPeriod: $period/" param_multi_agent_sim.yaml

        for (( c = 0; c <= 4; c++ )) # agent number
        do
            agents=$(echo ${AGENTS[c]})
            sed -i "s/numAgents:.*/numAgents: $agents/" param_multi_agent_sim.yaml

            for (( d = 0; d <= 0; d++ )) # fill ratios
            do
                min=$(echo ${MIN[d]})
                max=$(echo ${MAX[d]})
                inc=$(echo ${INC[d]})
                sed -i "/desFillRatios:/{n;N;N;d}" param_multi_agent_sim.yaml
                sed -i "s/desFillRatios:/desFillRatios:\n  min: $min\n  max: $max\n  incSteps: $inc/g" param_multi_agent_sim.yaml
                apptainer run multi_agent_sim_static.sif
            done

            # Copy and move the data
            folder=${COMM}_${period}_${agents} # concatenate string and numbers as folder name
            mkdir $folder
            mv data/* $folder
        done
    done

    END_TIME=$(date +%m/%d/%Y-%H:%M:%S)
    echo -e "\n################################### EXECUTION END ###################################"
    echo -e "################################# ${END_TIME} #################################\n"
}