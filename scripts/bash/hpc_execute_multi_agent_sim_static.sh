#!/bin/bash

# Script to execute static multi agent simulation (Python-based)
# on the HPC cluster, which means that the Python script is run
# in an Apptainer container. The apptainer name is assumed to be
# multi_agent_sim_static.sif.

# Verify that arguments are provided
if [ $# != 3 ]; then
    echo "Not enough arguments provided!"
    exit 1
fi

PARAMFILE=$1
SIFFILE=$2

s1="full"
s2="ring"
s3="line"
s4="scale-free"

if [ $3 != $s1 ] && [ $3 != $s2 ] && [ $3 != $s3 ] && [ $3 != $s4 ]; then
    echo "1st argument should be one of the following: \"full\", \"ring\", \"line\", \"scale-free\"!"
    exit 1
fi

# Iterate and run scripts for different param cases
COMM=$3 # when run in the cluster different communication graph parameters are run with separate nodes for increased efficiency
PERIOD=(1 2 5 10)
AGENTS=(10 20 50 100 200)

MIN=(0.05)
MAX=(0.95)
INC=(19)

# Set fixed parameters
sed -i "s/numTrials:.*/numTrials: 5/g" $PARAMFILE
sed -i "s/numSteps:.*/numSteps: 2000/g" $PARAMFILE

# Run simulations
{
    START_TIME=$(date +%m/%d/%Y-%H:%M:%S)

    echo -e "\n################################### EXECUTION BEGIN ###################################"
    echo -e "################################# ${START_TIME} #################################\n"

    sed -i "s/type:.*/type: \"$COMM\"/g" $PARAMFILE # communication network graph type

    for (( b = 0; b < ${#PERIOD[@]}; b++ )) # comms period
    do
        period=$(echo ${PERIOD[b]})
        sed -i "s/commsPeriod:.*/commsPeriod: $period/" $PARAMFILE

        for (( c = 0; c < ${#AGENTS[@]}; c++ )) # agent number
        do
            agents=$(echo ${AGENTS[c]})
            sed -i "s/numAgents:.*/numAgents: $agents/" $PARAMFILE

            for (( d = 0; d < 1; d++ )) # fill ratios
            do
                min=$(echo ${MIN[d]})
                max=$(echo ${MAX[d]})
                inc=$(echo ${INC[d]})
                sed -i "/targFillRatios:/{n;N;N;d}" $PARAMFILE # remove the line and 2 lines after 'targFillRatios'
                sed -i "s/targFillRatios:/targFillRatios:\n  min: $min\n  max: $max\n  incSteps: $inc/g" $PARAMFILE
                singularity exec $SIFFILE multi_agent_sim_static.py -p
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
