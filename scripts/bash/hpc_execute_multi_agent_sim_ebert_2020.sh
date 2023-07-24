#!/bin/bash

# Script to execute static multi agent simulation (ARGoS-based)
# on the HPC cluster, which means that the C++ script is run
# in an Apptainer container. 

# Verify that arguments are provided
if [ $# != 3 ]; then
    echo "Not enough arguments provided!"
    exit 1
fi

ARGOSFILE=$1
SIFFILE=$2
OUTPUTDIR=$3

# Define outer group parameters
SPEED=(14.14) # cm/s
POSITION=(3.1517942390846527) # for D = 1 with number of agents = 25
DENSITY=(1)
PRIOR=10
THRSH=0.99
POSFB="true"

# Set fixed parameters
TFR_RANGE=(0.55 0.95 2)
SP_RANGE=(0.525 0.975 4)
TRIALS=30
STEPS=1000
TILES=1000
DATAPATH="data.json"
AGENTS=25
WALL_THICKNESS=0.1
ARENA_LEN=10

sed -i "s/<fill_ratio_range.*/<fill_ratio_range min=\"${TFR_RANGE[0]}\" max=\"${TFR_RANGE[1]}\" steps=\"${TFR_RANGE[2]}\" \/>/" $ARGOSFILE # fill ratio range
sed -i "s/<sensor_probability_range.*/<sensor_probability_range min=\"${SP_RANGE[0]}\" max=\"${SP_RANGE[1]}\" steps=\"${SP_RANGE[2]}\" \/>/" $ARGOSFILE # sensor prob range
sed -i "s/<positive_feedback.*/<positive_feedback bool=\"${POSFB}\" \/>/" $ARGOSFILE # positive feedback
sed -i "s/<prior.*/<prior value=\"${PRIOR}\" \/>/" $ARGOSFILE # beta prior parameter
sed -i "s/<credible_threshold.*/<credible_threshold value=\"${THRSH}\" \/>/" $ARGOSFILE # credible threshold
sed -i "s/<num_trials.*/<num_trials value=\"$TRIALS\" \/>/" $ARGOSFILE # number of trials
sed -i "s/<arena_tiles.*/<arena_tiles tile_count_x=\"$TILES\" tile_count_y=\"$TILES\" \/>/" $ARGOSFILE # tile count
sed -i "s/<path.*/<path folder=\"$OUTPUTDIR\" name=\"$DATAPATH\" include_datetime=\"true\" \/>/" $ARGOSFILE # output path
sed -i "s/<verbosity.*/<verbosity level=\"full\" \/>/" $ARGOSFILE # verbosity
sed -i "s/<experiment.*/<experiment length=\"$STEPS\" ticks_per_second=\"10\" random_seed=\"0\" \/>/" $ARGOSFILE # experiment length
sed -i "s/<entity.*/<entity quantity=\"$AGENTS\" max_trials=\"100\" base_num=\"0\">/" $ARGOSFILE # number of robots
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

            apptainer exec $SIFFILE /collective_perception/collective_perception_dynamic/build/src/run_dynamic_simulations -c $ARGOSFILE
        done
    done

    END_TIME=$(date +%m/%d/%Y-%H:%M:%S)
    echo -e "\n################################### EXECUTION END ###################################"
    echo -e "################################# ${END_TIME} #################################\n"
}
