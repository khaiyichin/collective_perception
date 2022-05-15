"""Display information of the serialized data. 
"""

from sim_modules import ExperimentData
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Display information of the serialized data.")
    parser.add_argument("DATA", type=str, help="Filename to the data file.")
    args = parser.parse_args()

    # Load data
    try:
        data = ExperimentData.load(args.DATA, False)
        data_type = "MultiAgentSimData"

    except Exception as e:
        # for future loading of other types of data
        pass

    print("\nSerialized Data Properties:\n")
    
    print("\tType:\t\t\t\t\t{0}".format(data_type))
    print("\tNumber of agents:\t\t\t{0}".format(data.num_agents))
    print("\tNumber of experiments:\t\t\t{0}".format(data.num_exp))
    print("\tNumber of observations:\t\t\t{0}".format(data.num_obs))
    print("\tSimulated target fill ratios:\t\t{0}".format(data.dfr_range))
    print("\tSimulated sensor probabilities:\t\t{0}".format(data.sp_range))
    print("\tCommunication properties:")
    print("\t\tGraph type:\t\t\t{0}".format(data.graph_type))
    print("\t\tCommunication period:\t\t{0}".format(data.comms_period))
    print("\t\tCommunication probability:\t{0}".format(data.comms_prob))