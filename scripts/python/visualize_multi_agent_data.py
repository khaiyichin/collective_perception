from sim_modules import MultiAgentSimData
import viz_modules as vm
import argparse
import numpy as np

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Visualize multi-agent simulation data")
    parser.add_argument("DATA", type=str, help="filename to the data file")
    parser.add_argument("-t", action="store_true", help="flag to plot time series data for a randomly selected target fill ratio and sensor probability")
    parser.add_argument("-m", action="store_true", help="flag to plot heatmap data; only useful if multiple target fill ratios and sensor probabilities are simulated in data")
    args = parser.parse_args()

    # Load the data
    data = MultiAgentSimData.load(args.DATA)

    # Plot time series data for a random target fill ratio and sensor probability since there
    # too many to plot
    if args.t:
        target_fill_ratio = np.random.choice(data.dfr_range)
        sensor_prob = np.random.choice(data.sp_range)
        vm.plot_timeseries(target_fill_ratio, sensor_prob, data)

    # Plot heatmap for all simulation data
    if args.m:
        pass