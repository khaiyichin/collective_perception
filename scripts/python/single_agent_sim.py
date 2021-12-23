from sim_modules import SingleAgentSim, HeatmapData, HeatmapRow

import numpy as np
import os
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

if __name__ == "__main__":
    """
    SIM PARAM BEGIN
    """
    sensor_prob = np.round(np.linspace(0.55, 1.0, 19), 3)
    des_fill_ratios = np.round(np.linspace(0.05, 0.95, 19), 3)
    
    num_cycle = 200
    num_obs = 500

    """
    SIM PARAM END
    """

    # Compute increment step sizes
    sensor_prob_inc = sensor_prob[1] - sensor_prob[0]
    fill_ratio_inc = des_fill_ratios[1] - des_fill_ratios[0]
    
    # Define filename descriptors
    min_sensor_prob, max_sensor_prob = int(sensor_prob[0]*1e2), int(sensor_prob[-1]*1e2)
    min_des_fill_ratio, max_des_fill_ratio = int(des_fill_ratios[0]*1e2), int(des_fill_ratios[-1]*1e2)
    p_inc, f_inc = int(sensor_prob_inc*1e2), int(fill_ratio_inc*1e2)

    filename_suffix_1 = "_c" +str(num_cycle) + "_o" + str(num_obs) # describing number of cycles and observations

    prob_suffix = "_p" + str(min_sensor_prob) + "-" + str(p_inc) + "-" + str(max_sensor_prob)
    f_suffix = "_f" + str(min_des_fill_ratio) + "-" + str(f_inc) + "-" + str(max_des_fill_ratio)
    filename_suffix_2 = prob_suffix + f_suffix # describing the probabilites and fill ratios

    # Create a folder to store simulation data
    create_simulation_folder(filename_suffix_1+filename_suffix_2)

    # Create a HeatmapData object to process heatmap data
    hm = HeatmapData(num_cycle, num_obs, sensor_prob, des_fill_ratios, filename_suffix_1+filename_suffix_2)

    # Run simulations
    for f in des_fill_ratios: # iterate through each desired fill ratio

        # Create HeatmapRow object
        hr = HeatmapRow()

        print("\nRunning cases with fill ratio = " + str(f))
        
        for p in sensor_prob: # iterate through each sensor probabilities
            print("\tRunning case with probability ratio = " + str(p) + "... ", end="")

            s = SingleAgentSim(num_cycle, num_obs, f, p, p, filename_suffix_1)
            s.run() # run the single agent simulation

            hr.populate(s) # populate one heatmap row for both f_hat and fisher_inv

            print("Done!")
            
        hm.compile_data(hr)

    hm.write_data_to_csv()