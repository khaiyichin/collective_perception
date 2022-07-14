#!/usr/bin/env python3
"""Load and convert ExperimentData objects into a VisualizationDataGroupStatic object.
"""

import collective_perception_py.viz_modules as vm
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Load and convert ExperimentData objects into a VisualizationDataGroupStatic object. \
        \nThis script will convert and store ExperimentData objects with the same: \
            \n\t- communication network type, and \
            \n\t- target fill ratio range, \
            \n\t- sensor probability range, \
            \n\t- number of steps, and \
            \n\t- number of trials, \
        \nand with varying: \
            \n\t- number of agents, \
            \n\t- communication period, \
            \n\t- communication probability.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("FOLDER", type=str, help="path to the top level directory containing the serialized ExperimentData files")
    parser.add_argument("-s", type=str, help="path to store the pickled VisualizationDataGroupStatic object")
    args = parser.parse_args()

    # Initialize default values
    if not args.s: args.s = None

    # Load data
    v = vm.VisualizationDataGroupStatic(args.FOLDER)

    # Pickle data
    v.save(args.s)