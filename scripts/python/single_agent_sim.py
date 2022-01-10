from sim_modules import SingleAgentSim, HeatmapData, HeatmapRow, SimParam

import numpy as np
import os
import yaml
from datetime import datetime

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

def create_simulation_folder(suffix):
    """Create a simulation folder within the `data` directory.
    """

    # Create data folder and change working directory
    create_data_folder()

    # Define folder name
    curr_time = datetime.now().strftime("%m%d%Y_%H%M%S")
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

    # Parse simulation parameters
    param_obj = parse_yaml_param_file("param_single_agent_sim.yaml")

    # Create a folder to store simulation data
    create_simulation_folder(param_obj.full_suffix)

    # Create a HeatmapData object to process heatmap data
    hm = HeatmapData(param_obj)

    # Run simulations
    for f in param_obj.dfr_range: # iterate through each desired fill ratio

        # Create HeatmapRow object
        hr = HeatmapRow()

        print("\nRunning cases with fill ratio = " + str(f))
        
        for p in param_obj.sp_range: # iterate through each sensor probabilities
            print("\tRunning case with probability ratio = " + str(p) + "... ", end="")

            s = SingleAgentSim(param_obj.num_cycle, param_obj.num_obs, f, p, p, param_obj.filename_suffix_1)
            s.run(param_obj.write_all) # run the single agent simulation

            hr.populate(s) # populate one heatmap row for both f_hat and fisher_inv

            print("Done!")
            
        hm.compile_data(hr)

    # Write completed heatmap data to CSV files
    hm.write_data_to_csv()

    print("\nSimulation complete!")