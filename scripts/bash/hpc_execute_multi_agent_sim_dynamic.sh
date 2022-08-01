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
POSITION=(4.436599480604251 3.1517942390846527 2.011746925739275 1.4371645541621039) # for D = (1 2 5 10) with number of agents = 50
DENSITY=(1 2 5 10)
THREADS=40

# Set fixed parameters
TFR_RANGE=(0.05 0.95 19)
SP_RANGE=(0.525 0.975 19)
TRIALS=5
STEPS=50
TILES=1000
STATSPATH="multi_agent_sim_dynamic_stats.pbs"
AGENTDATAPATH="multi_agent_sim_dynamic_agent_data.pbad"
AGENTS=50
WALL_THICKNESS=0.1
ARENA_LEN=10
LEGACY="false"

sed -i "s/<fill_ratio_range.*/<fill_ratio_range min=\"${TFR_RANGE[0]}\" max=\"${TFR_RANGE[1]}\" steps=\"${TFR_RANGE[2]}\" \/>/" $ARGOSFILE # fill ratio range
sed -i "s/<sensor_probability_range.*/<sensor_probability_range min=\"${SP_RANGE[0]}\" max=\"${SP_RANGE[1]}\" steps=\"${SP_RANGE[2]}\" \/>/" $ARGOSFILE # sensor prob range
sed -i "s/<num_trials.*/<num_trials value=\"$TRIALS\" \/>/" $ARGOSFILE # number of trials
sed -i "s/<arena_tiles.*/<arena_tiles tile_count_x=\"$TILES\" tile_count_y=\"$TILES\" \/>/" $ARGOSFILE # tile count
sed -i "s/<path.*/<path folder=\"$OUTPUTDIR\" stats=\"$STATSPATH\" agent_data=\"$AGENTDATAPATH\"  include_datetime=\"true\" \/>/" $ARGOSFILE # output path
sed -i "s/<verbosity.*/<verbosity level=\"full\" \/>/" $ARGOSFILE # verbosity
sed -i "s/<legacy.*/<legacy bool=\"$LEGACY\" \/>/" $ARGOSFILE # legacy equations
sed -i "s/<experiment.*/<experiment length=\"$STEPS\" ticks_per_second=\"10\" random_seed=\"0\" \/>/" $ARGOSFILE # experiment length
sed -i "s/<entity.*/<entity quantity=\"$AGENTS\" max_trials=\"100\" base_num=\"0\">/" $ARGOSFILE
if [ $AGENTS -ge 100 ]; then # thread number
    sed -i "s/<system threads=.*/<system threads=\"$THREADS\" \/>/" $ARGOSFILE
else
    sed -i "s/<system threads=.*/<system threads=\"0\" \/>/" $ARGOSFILE
fi
sed -i "s/<arena size.*/<arena size=\"$ARENA_LEN, $ARENA_LEN, 1\" center=\"0,0,0.5\">/" $ARGOSFILE # arena size
sed -i "s/<box id=\"wall_north\".*/<box id=\"wall_north\" size=\"10,$WALL_THICKNESS,0.5\" movable=\"false\">/" $ARGOSFILE # north wall size
sed -i "s/<box id=\"wall_south\".*/<box id=\"wall_south\" size=\"10,$WALL_THICKNESS,0.5\" movable=\"false\">/" $ARGOSFILE # south wall size
sed -i "s/<box id=\"wall_east\".*/<box id=\"wall_east\" size=\"$WALL_THICKNESS,10,0.5\" movable=\"false\">/" $ARGOSFILE # east wall size
sed -i "s/<box id=\"wall_west\".*/<box id=\"wall_west\" size=\"$WALL_THICKNESS,10,0.5\" movable=\"false\">/" $ARGOSFILE # west wall size

# Run simulations
{
    START_TIME=$(date +%m/%d/%Y-%H:%M:%S)

    echo -e "\n################################### EXECUTION BEGIN ###################################"
    echo -e "################################# ${START_TIME} #################################\n"

    for (( i = 0; i < ${#SPEED[@]}; i++ )) # robot speeds
    do
        # Modify robot speeds
        speed=$(echo ${SPEED[i]})
        sed -i "s/<speed.*/<speed value=\"$speed\" \/>/" $ARGOSFILE

        for (( j = 0; j < ${#POSITION[@]}; j++ )) # wall positions
        do
            # Modify wall positions
            pos=$(echo ${POSITION[j]})
            sed -i "/<box id=\"wall_north\".*/{n;d}" $ARGOSFILE # remove the line after "wall_north"
            sed -i "/<box id=\"wall_south\".*/{n;d}" $ARGOSFILE # remove the line after "wall_south"
            sed -i "/<box id=\"wall_east\".*/{n;d}" $ARGOSFILE # remove the line after "wall_east"
            sed -i "/<box id=\"wall_west\".*/{n;d}" $ARGOSFILE # remove the line after "wall_west"
            sed -i "s/<box id=\"wall_north\".*/<box id=\"wall_north\" size=\"10,$WALL_THICKNESS,0.5\" movable=\"false\">\n            <body position=\"0,$pos,0\" orientation=\"0,0,0\" \/>/" $ARGOSFILE
            sed -i "s/<box id=\"wall_south\".*/<box id=\"wall_south\" size=\"10,$WALL_THICKNESS,0.5\" movable=\"false\">\n            <body position=\"0,-$pos,0\" orientation=\"0,0,0\" \/>/" $ARGOSFILE
            sed -i "s/<box id=\"wall_east\".*/<box id=\"wall_east\" size=\"$WALL_THICKNESS,10,0.5\" movable=\"false\">\n            <body position=\"$pos,0,0\" orientation=\"0,0,0\" \/>/" $ARGOSFILE
            sed -i "s/<box id=\"wall_west\".*/<box id=\"wall_west\" size=\"$WALL_THICKNESS,10,0.5\" movable=\"false\">\n            <body position=\"-$pos,0,0\" orientation=\"0,0,0\" \/>/" $ARGOSFILE
            sed -i "s/<position method=\"uniform\".*/<position method=\"uniform\" min=\"-$pos,-$pos,0\" max=\"$pos,$pos,0\" \/>/" $ARGOSFILE

            singularity exec $SIFFILE /collective_perception/collective_perception_dynamic/build/src/run_dynamic_simulations -l /dev/null -c $ARGOSFILE

            # Copy and move the data
            folder="spd${speed}_den${DENSITY[j]}" # concatenate string and numbers as folder name
            mkdir -p $folder
            mv $OUTPUTDIR/* $folder
        done
    done

    END_TIME=$(date +%m/%d/%Y-%H:%M:%S)
    echo -e "\n################################### EXECUTION END ###################################"
    echo -e "################################# ${END_TIME} #################################\n"
}
