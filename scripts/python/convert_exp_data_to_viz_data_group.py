"""Load and convert ExperimentData objects into a VisualizationDataGroup object.
"""

import viz_modules as vm
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Load and convert ExperimentData objects into a VisualizationDataGroup object. \
        \nThis script will convert and store ExperimentData objects with the same: \
            \n\t- communication network type, and \
            \n\t- number of experiments, \
        \nand with varying: \
            \n\t- number of agents, \
            \n\t- communication period, \
            \n\t- communication probability.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("FOLDER", type=str, help="path to the top level directory containing the serialized ExperimentData files")
    parser.add_argument("-s", type=str, help="path to store the pickled VisualizationDataGroup object")
    args = parser.parse_args()

    # Initialize default values
    if not args.s: args.s = None

    # Load data
    v = vm.VisualizationDataGroup(args.FOLDER)

    # Pickle data
    v.save(args.s)