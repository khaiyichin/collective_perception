from sim_modules import MultiAgentSim, MultiAgentSimData, SimParam

import numpy as np
import os
import yaml
from datetime import datetime
from joblib import Parallel, delayed
import timeit
import argparse

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

def run_sim_parallel(param_obj, target_fill_ratio):
    print("\nRunning cases with fill ratio = " + str(target_fill_ratio))

    outputs = []

    for p in param_obj.sp_range: # iterate through each sensor probabilities
        s = MultiAgentSim(param_obj, target_fill_ratio, p)
        s.run() # run the multi agent simulation

        outputs.append((target_fill_ratio, p, s))

    return outputs

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Visualize multi-agent simulation data")
    parser.add_argument("-p", action="store_true", help="flag to use cores to run simulations in parallel.")
    args = parser.parse_args()

    # Obtain timestamp
    curr_time = datetime.now().strftime("%m%d%y_%H%M%S")

    # Parse simulation parameters
    param_obj = parse_yaml_param_file("param_multi_agent_sim.yaml")

    # Create a folder to store simulation data
    create_simulation_folder(param_obj.full_suffix, curr_time)

    # Create a HeatmapData object to process heatmap data
    data = MultiAgentSimData(param_obj)

    if args.p:
        lst_of_lst_outputs = Parallel(n_jobs=-1, verbose=100)(delayed(run_sim_parallel)(param_obj, f) for f in param_obj.dfr_range)
        outputs = [output_tup for target_fill_ratio_outputs_lst in lst_of_lst_outputs
                   for output_tup in target_fill_ratio_outputs_lst]

        # Insert the simulation object into the data object
        [data.insert_sim_obj(o[0], o[1], o[2]) for o in outputs]

    else:
        start = timeit.default_timer()
        for f in param_obj.dfr_range: # iterate through each desired fill ratio

            print("\nRunning cases with fill ratio = " + str(f))

            for p in param_obj.sp_range: # iterate through each sensor probabilities
                print("\tRunning case with probability ratio = " + str(p) + "... ", end="")

                s = MultiAgentSim(param_obj, f, p)
                s.run() # run the multi agent simulation

                data.insert_sim_obj(f, p, s) # store completed simulation object into

                print("Done!")

        end = timeit.default_timer()

        print('Elapsed time:', end-start)

    data.save(curr_time) # serialize and store data

    print("\nSimulation complete!")