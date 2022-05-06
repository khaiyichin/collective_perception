from sim_modules import MultiAgentSim, MultiAgentSimData, SimParam

import numpy as np
import os
import yaml
from datetime import datetime


"""
Want to test/break the following assumptions:
- full communication with all agents (graph comm probability =/= 1.0)
- same sensor quality among all agents
- communicate after each observation
"""

# TODO:
# pickling files vs protobuf vs csv? (you can serialize then write to csv also)

def create_data_folder():
    """Create a folder named data and use it as the working directory.
    """

    # Create the `data` folder if it doesn't exist
    try:
        os.mkdir("data")
        os.chdir("data")
        
    except FileExistsError:

        # Change to `data` directory
        try:
            os.chdir("data")
        except Exception:
            pass

def create_simulation_folder(suffix, curr_time):
    """Create a simulation folder within the `data` directory.
    """

    # Create data folder and change working directory
    create_data_folder()

    # Define folder name
    folder_name = curr_time + suffix

    # Create folder to store simulation data based on current datetime if doesn't exist (shouldn't exist, really)
    try:
        os.mkdir(folder_name)
        os.chdir(folder_name)

    except FileExistsError:
        print("folder {} already exists".format(folder_name))

        try:
            os.chdir(folder_name)
        except Exception:
            pass

def parse_yaml_param_file(yaml_filepath):
    """Parse the YAML parameter file.
    """

    with open(yaml_filepath) as fopen:
        try:
            config = yaml.safe_load(fopen)
        except yaml.YAMLError as exception:
            print(exception)
    
        # Displayed processed arguments
        print('\n\t' + '='*15 + ' Processed Parameters ' + '='*15 + '\n')
        print('\t\t', end='')
        for line in yaml.dump(config, indent=4, default_flow_style=False):        
            if line == '\n':
                print(line, '\t\t',  end='')
            else:
                print(line, end='')
        print('\r', end='') # reset the cursor for print
        print('\n\t' + '='*13 + ' End Processed Parameters ' + '='*13  + '\n')

    param_obj = SimParam(config)

    return param_obj

if __name__ == "__main__":

    # 'num_agents', 'num_exp', 'num_obs', 'des_fill_ratio', 'b_prob', 'w_prob', 'comms_period', and 'main_f_suffix'
    # sample_init_params = [4, 2, 10, 0.75, 0.9, 0.9, 1, 'dummy-debug']

    # Obtain timestamp
    curr_time = datetime.now().strftime("%m%d%y_%H%M%S")

    # Parse simulation parameters
    param_obj = parse_yaml_param_file("param_multi_agent_sim.yaml")

    # Create a folder to store simulation data
    create_simulation_folder(param_obj.full_suffix, curr_time)

    # Create a HeatmapData object to process heatmap data
    data = MultiAgentSimData(param_obj)

    for f in param_obj.dfr_range: # iterate through each desired fill ratio

        print("\nRunning cases with fill ratio = " + str(f))
        
        for p in param_obj.sp_range: # iterate through each sensor probabilities
            print("\tRunning case with probability ratio = " + str(p) + "... ", end="")

            s = MultiAgentSim(param_obj, f, p)
            s.run() # run the multi agent simulation

            # hr.populate(s) # populate one heatmap row for both f_hat and fisher_inv
            data.insert_sim_obj(f, p, s) # store completed simulation object into

            print("Done!")

    data.save(curr_time) # serialize and store data

    print("\nSimulation complete!")