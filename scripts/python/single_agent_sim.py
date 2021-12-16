from sim_modules import SingleAgentSim

import numpy as np
import csv
from datetime import datetime

if __name__ == "__main__":
    
    # Define simulation parameters
    # sensor_prob = np.linspace(0.05, 0.95, 10)
    sensor_prob = [0.05]
    # des_fill_ratios = np.linspace(0.05, 0.95, 10)
    des_fill_ratios = [0.05]
    num_cycle = 50
    num_obs = 1000
    raw_data = {}
    heatmap_data = {"mean": [], "min": [], "max": []}

    for f in des_fill_ratios:

        print("Running cases with fill ratio = " + str(f))
        
        mean_data = []
        min_data = []
        max_data = []

        for p in sensor_prob:
            print("\tRunning case with probability ratio = " + str(p) + "... ", end="")

            s = SingleAgentSim(num_cycle, num_obs, f, p, p)

            s.run(True)

            raw_data[f] = {
                p: {
                    "f_hat_mean": s.f_hat_sample_mean,
                    "f_hat_std": s.f_hat_sample_std,
                    "fisher_mean": s.fisher_info_sample_mean,
                    "fisher_std": s.fisher_info_sample_std
                    }
                }

            mean_data.append( s.fisher_info_sample_mean[-1] )
            min_data.append( s.fisher_info_sample_min[-1] )
            max_data.append( s.fisher_info_sample_max[-1] )

            print("Done!")
            
        heatmap_data["mean"].append(mean_data)
        heatmap_data["min"].append(min_data)
        heatmap_data["max"].append(max_data)

    # Write heatmap data    
    curr_time = datetime.now().strftime("%d%m%Y_%H%M%S")
    mean_filename = "./data/heatmap_mean" + curr_time + "_" + str(num_cycle) + "_" + str(num_obs) + ".csv"
    min_filename = "./data/heatmap_min" + curr_time + "_" + str(num_cycle) + "_" + str(num_obs) + ".csv"    
    max_filename = "./data/heatmap_max" + curr_time + "_" + str(num_cycle) + "_" + str(num_obs) + ".csv"

    with open(mean_filename, "w", encoding="UTF8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(heatmap_data["mean"])

    with open(min_filename, "w", encoding="UTF8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(heatmap_data["min"])

    with open(max_filename, "w", encoding="UTF8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(heatmap_data["max"])