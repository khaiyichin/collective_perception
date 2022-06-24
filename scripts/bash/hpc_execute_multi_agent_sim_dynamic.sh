#!/bin/bash

# Script to execute static multi agent simulation (ARGoS-based)
# on the HPC cluster, which means that the C++ script is run
# in an Apptainer container. The apptainer name is assumed to be
# multi_agent_sim_dynamic_no_qt.sif.

# Verify that arguments are provided
if [ $# != 3 ]; then
    echo "Not enough arguments provided!"
    exit 1
fi

ARGOSFILE=$1
SIFFILE=$2
OUTPUTDIR=$3

# Define varying parameters
SPEED=(10.0 15.0 20.0 25.0) # cm/s
AGENTS=(10 20 50 100)
THREADS=20

# Set fixed parameters
TFR_RANGE=(0.05 0.95 19)
SP_RANGE=(0.525 0.975 19)
TRIALS=5
TILES=500
STATSPATH="multi_agent_sim_dynamic_stats.pbs"
AGENTDATAPATH="multi_agent_sim_dynamic_agent_data.pbad"

sed -i "s/<fill_ratio_range.*/<fill_ratio_range min=\"${TFR_RANGE[0]}\" max=\"${TFR_RANGE[1]}\" steps=\"${TFR_RANGE[2]}\" \/>/" $ARGOSFILE # fill ratio range
sed -i "s/<sensor_probability_range.*/<sensor_probability_range min=\"${SP_RANGE[0]}\" max=\"${SP_RANGE[1]}\" steps=\"${SP_RANGE[2]}\" \/>/" $ARGOSFILE # sensor prob range
sed -i "s/<num_trials.*/<num_trials value=\"$TRIALS\" \/>/" $ARGOSFILE # number of trials
sed -i "s/<arena_tiles.*/<arena_tiles tile_count_x=\"$TILES\" tile_count_y=\"$TILES\" \/>/" $ARGOSFILE # tile count
sed -i "s/<path.*/<path folder=\"$OUTPUTDIR\" stats=\"$STATSPATH\" agent_data=\"$AGENTDATAPATH\"  include_datetime=\"true\" \/>/" $ARGOSFILE # output path
sed -i "s/<verbosity.*/<verbosity level=\"full\" \/>/" $ARGOSFILE # verbosity
sed -i "s/<experiment.*/<experiment length=\"200\" ticks_per_second=\"10\" random_seed=\"0\" \/>/" $ARGOSFILE # experiment length

# Run simulations
{
    START_TIME=$(date +%m/%d/%Y-%H:%M:%S)

    echo -e "\n################################### EXECUTION BEGIN ###################################"
    echo -e "################################# ${START_TIME} #################################\n"

    for (( i = 0; i <= 3; i++ )) # robot speeds
    do
        # Modify robot speeds
        speed=$(echo ${SPEED[i]})
        sed -i "s/<speed.*/<speed value=\"$speed\" \/>/" $ARGOSFILE

        for (( j = 0; j <= 3; j++ )) # agent number
        do
            agents=$(echo ${AGENTS[j]})

            # Modify number of threads
            if [ $agents -ge 100 ]; then
                sed -i "s/<system threads=.*/<system threads=\"$THREADS\" \/>/" $ARGOSFILE
            else
                sed -i "s/<system threads=.*/<system threads=\"0\" \/>/" $ARGOSFILE
            fi

            # Modify number of agents
            sed -i "s/<entity.*/<entity quantity=\"$agents\" max_trials=\"100\" base_num=\"0\">/" $ARGOSFILE

            singularity exec $SIFFILE /collective_perception_cpp/build/src/run_dynamic_simulations -l /dev/num -c $ARGOSFILE

            # Copy and move the data
            folder="spd${speed}_agt${agents}" # concatenate string and numbers as folder name
            mkdir $folder
            mv $OUTPUTDIR/* $folder
        done
    done

    END_TIME=$(date +%m/%d/%Y-%H:%M:%S)
    echo -e "\n################################### EXECUTION END ###################################"
    echo -e "################################# ${END_TIME} #################################\n"
}
