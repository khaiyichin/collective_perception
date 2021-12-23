from sim_modules import SingleAgentSim, HeatmapData, HeatmapRow

import numpy as np
import csv
import os
from datetime import datetime

def create_data_folder():
    try:
        os.mkdir("data")
        os.chdir("data")
    except FileExistsError:
        print("folder {} already exists".format(folder_name))
    pass

if __name__ == "__main__":
    
    # Define simulation parameters
    """
    SIM PARAM BEGIN
    """
    sensor_prob = np.round(np.linspace(0.55, 1.0, 4), 3)
    des_fill_ratios = np.round(np.linspace(0.05, 0.95, 5), 3)
    
    num_cycle = 200
    num_obs = 1000

    """
    SIM PARAM END
    """
    raw_data = {}
    hm = HeatmapData(num_cycle, num_obs, sensor_prob, des_fill_ratios)
    # f_hat_heatmap_data = {"mean": [], "min": [], "max": []}
    # fisher_inv_heatmap_data = {"mean": [], "min": [], "max": []}
    
    curr_time = datetime.now().strftime("%m%d%Y_%H%M%S")
    suffix = "_p" + str(int(sensor_prob[0]*100)) + "-" + str(int(sensor_prob[-1]*100)) + "_f" + str(int(des_fill_ratios[0]*100)) + "-" + str(int(des_fill_ratios[-1]*100))
    folder_name = os.path.join("./data/", curr_time + "_" + str(num_cycle) + "_" + str(num_obs) + suffix)

    for f in des_fill_ratios:

        hr = HeatmapRow()

        print("\nRunning cases with fill ratio = " + str(f))
        
        # f_hat_mean_data = []
        # f_hat_min_data = []
        # f_hat_max_data = []
        # fisher_inv_mean_data = []
        # fisher_inv_min_data = []
        # fisher_inv_max_data = []

        for p in sensor_prob:
            print("\tRunning case with probability ratio = " + str(p) + "... ", end="")

            s = SingleAgentSim(num_cycle, num_obs, f, p, p)

            s.run()

            raw_data[f] = {
                p: {
                    "f_hat_mean": s.f_hat_sample_mean,
                    "f_hat_std": s.f_hat_sample_std,
                    "fisher_mean": s.fisher_inv_sample_mean,
                    "fisher_std": s.fisher_inv_sample_std
                    }
                }

            hr.populate(s)

            # f_hat_mean_data.append( s.f_hat_sample_mean[-1] )
            # f_hat_min_data.append( s.f_hat_sample_min[-1] )
            # f_hat_max_data.append( s.f_hat_sample_max[-1] )
            # fisher_inv_mean_data.append( s.fisher_inv_sample_mean[-1] )
            # fisher_inv_min_data.append( s.fisher_inv_sample_min[-1] )
            # fisher_inv_max_data.append( s.fisher_inv_sample_max[-1] )

            print("Done!")
            
        # f_hat_heatmap_data["mean"].append(f_hat_mean_data)
        # f_hat_heatmap_data["min"].append(f_hat_min_data)
        # f_hat_heatmap_data["max"].append(f_hat_max_data)            
        # fisher_inv_heatmap_data["mean"].append(fisher_inv_mean_data)
        # fisher_inv_heatmap_data["min"].append(fisher_inv_min_data)
        # fisher_inv_heatmap_data["max"].append(fisher_inv_max_data)

        hm.compile_data(hr)

    # Create folder if doesn't exist (shouldn't exist, really)
    try:
        os.mkdir(folder_name)
        os.chdir(folder_name)
    except FileExistsError:
        print("folder {} already exists".format(folder_name))

    hm.write_data_to_csv()

    # Write heatmap data
    # f_hat_mean_filename = "f_hat_heatmap_mean" + curr_time + "_" + str(num_cycle) + "_" + str(num_obs) + suffix + ".csv"
    # f_hat_min_filename = "f_hat_heatmap_min" + curr_time + "_" + str(num_cycle) + "_" + str(num_obs) + suffix + ".csv"    
    # f_hat_max_filename = "f_hat_heatmap_max" + curr_time + "_" + str(num_cycle) + "_" + str(num_obs) + suffix + ".csv"
    # fisher_inv_mean_filename = "fisher_inv_heatmap_mean" + curr_time + "_" + str(num_cycle) + "_" + str(num_obs) + suffix + ".csv"
    # fisher_inv_min_filename = "fisher_inv_heatmap_min" + curr_time + "_" + str(num_cycle) + "_" + str(num_obs) + suffix + ".csv"    
    # fisher_inv_max_filename = "fisher_inv_heatmap_max" + curr_time + "_" + str(num_cycle) + "_" + str(num_obs) + suffix + ".csv"

    # with open(f_hat_mean_filename, "w", encoding="UTF8", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerows(f_hat_heatmap_data["mean"])

    # with open(f_hat_min_filename, "w", encoding="UTF8", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerows(f_hat_heatmap_data["min"])

    # with open(f_hat_max_filename, "w", encoding="UTF8", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerows(f_hat_heatmap_data["max"])

    # with open(fisher_inv_mean_filename, "w", encoding="UTF8", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerows(fisher_inv_heatmap_data["mean"])

    # with open(fisher_inv_min_filename, "w", encoding="UTF8", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerows(fisher_inv_heatmap_data["min"])

    # with open(fisher_inv_max_filename, "w", encoding="UTF8", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerows(fisher_inv_heatmap_data["max"])