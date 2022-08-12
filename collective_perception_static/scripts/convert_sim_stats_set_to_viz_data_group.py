#!/usr/bin/env python3
"""Load and convert SimulationStatsSet protobuf files into a VisualizationDataGroupDynamic object.
"""

import collective_perception_py.viz_modules as vm
import argparse

def main():

    parser = argparse.ArgumentParser(description="Load and convert SimulationStatsSet protobuf files into a VisualizationDataGroupDynamic object. \
        \nThis script will convert and store SimulationStatsSet protobuf files with the same: \
            \n\t- target fill ratio range, \
            \n\t- sensor probability range, \
            \n\t- number of steps, and \
            \n\t- number of trials, \
        \nand with varying: \
            \n\t- robot speed, and \
            \n\t- swarm density.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("FOLDER", type=str, help="path to the top level directory containing the serialized SimulationStatsSet protobuf files")
    parser.add_argument("-s", type=str, help="path to store the pickled VisualizationDataGroupDynamic object")
    args = parser.parse_args()

    # Initialize default values
    if not args.s: args.s = None

    # Load data
    v = vm.VisualizationDataGroupDynamic(args.FOLDER)

    # Pickle data
    v.save(args.s)

if __name__ == "__main__":
    main()