#!/bin/bash

# Script to execute static multi agent simulation (Python-based)
# on the HPC cluster, which means that the Python script is run
# in an Apptainer container. The apptainer name is assumed to be
# multi_agent_sim_static.sif.

# Verify that arguments are provided
if [ $# != 3 ]; then
    echo "Incorrect number of arguments provided!"
    exit 1
fi

PARAMFILE=$1
SIFFILE=$2

s1="full"
s2="ring"
s3="line"
s4="scale-free"

if [ $3 != $s1 ] && [ $3 != $s2 ] && [ $3 != $s3 ] && [ $3 != $s4 ]; then
    echo "3rd argument should be one of the following: \"full\", \"ring\", \"line\", \"scale-free\"!"
    exit 1
fi

# Iterate and run scripts for different param cases
COMM=$3 # when run in the cluster different communication graph parameters are run with separate nodes for increased efficiency
PERIOD=(1 2 5 10)
AGENTS=(10 20 50 100 200)

# Set fixed parameters
TFR_RANGE=(0.05 0.95 19)
SP_RANGE=(0.525 0.975 19)
TRIALS=5
STEPS=20000
LEGACY="False"

sed -i "/targFillRatios:/{n;N;N;d}" $PARAMFILE # remove 3 lines after 'targFillRatios'
sed -i "s/targFillRatios:/targFillRatios:\n  min: ${TFR_RANGE[0]}\n  max: ${TFR_RANGE[1]}\n  incSteps: ${TFR_RANGE[2]}/g" $PARAMFILE
sed -i "/sensorProb:/{n;N;N;d}" $PARAMFILE # remove 3 lines after 'sensorProb'
sed -i "s/sensorProb:/sensorProb:\n  min: ${SP_RANGE[0]}\n  max: ${SP_RANGE[1]}\n  incSteps: ${SP_RANGE[2]}/g" $PARAMFILE
sed -i "s/numTrials:.*/numTrials: $TRIALS/g" $PARAMFILE
sed -i "s/numSteps:.*/numSteps: $STEPS/g" $PARAMFILE
sed -i "s/legacy:.*/legacy: $LEGACY/g" $PARAMFILE

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

            singularity exec $SIFFILE multi_agent_sim_static.py $PARAMFILE -p

            # Copy and move the data
            folder=${COMM}_${period}_${agents} # concatenate string and numbers as folder name
            mkdir -p $folder
            mv data/* $folder
        done
    done

    END_TIME=$(date +%m/%d/%Y-%H:%M:%S)
    echo -e "\n################################### EXECUTION END ###################################"
    echo -e "################################# ${END_TIME} #################################\n"
}
