#!/bin/bash

# Switch to working directory
CURR_DIR=$(pwd)
echo "Changing to working directory: $1";
pushd $1

# Activate python virtual environment (assuming it's called .venv)
source .venv/bin/activate

# Copy scripts
cp $CURR_DIR/../examples/param/param_multi_agent_sim.yaml .
cp $CURR_DIR/../python/multi_agent_sim.py .
cp $CURR_DIR/../python/sim_modules.py .

# Iterate and run scripts for different param cases
COMM=("full" "ring" "line" "scale-free")
PERIOD=(1 2 5 10)
AGENTS=(5 10 50 100 200)

MIN=(0.05 0.25 0.45 0.65 0.85)
MAX=(0.20 0.4 0.6 0.8 0.95)
INC=(4 4 4 4 3)

# Set fixed parameters
sed -i "s/numExperiments:.*/numExperiments: 5/g" param_multi_agent_sim.yaml
sed -i "s/numObs:.*/numObs: 1000/g" param_multi_agent_sim.yaml

# Run simulations
{
    START_TIME=$(date +%m/%d/%Y-%H:%M:%S)

    echo -e "\n################################### EXECUTION BEGIN ###################################"
    echo -e "################################# ${START_TIME} #################################\n"
    for (( a = 0; a <= 3; a++ )) # comms type
    do
        comm=$(echo ${COMM[a]})
        sed -i "s/type:.*/type: \"$comm\"/g" param_multi_agent_sim.yaml

        for (( b = 0; b <= 3; b++ )) # comms period
        do
            period=$(echo ${PERIOD[b]})
            sed -i "s/commsPeriod:.*/commsPeriod: $period/" param_multi_agent_sim.yaml

            for (( c = 0; c <= 4; c++ )) # agent number
            do
                agents=$(echo ${AGENTS[c]})
                sed -i "s/numAgents:.*/numAgents: $agents/" param_multi_agent_sim.yaml

                for (( d = 0; d <= 4; d++ )) # fill ratios
                do
                    min=$(echo ${MIN[d]})
                    max=$(echo ${MAX[d]})
                    inc=$(echo ${INC[d]})
                    sed -i "/desFillRatios:/{n;N;N;d}" param_multi_agent_sim.yaml
                    sed -i "s/desFillRatios:/desFillRatios:\n  min: $min\n  max: $max\n  incSteps: $inc/g" param_multi_agent_sim.yaml
                    python3 multi_agent_sim.py -p # run parallel
                done

                # Copy and move the data
                folder=${comm}_${period}_${agents} # concatenate string and numbers as folder name
                mkdir $folder
                mv data/* $folder
            done
        done
    done
    END_TIME=$(date +%m/%d/%Y-%H:%M:%S)
    echo -e "\n################################### EXECUTION END ###################################"
    echo -e "################################# ${END_TIME} #################################\n"
} > execute_multi_agent_sim.out 2> execute_multi_agent_sim.err # collect logs