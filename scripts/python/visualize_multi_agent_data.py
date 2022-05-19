from sim_modules import ExperimentData
import viz_modules as vm
import argparse
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Visualize multi-agent simulation data")
    parser.add_argument("DATA", type=str, help="folder containing the data")
    parser.add_argument("-t", nargs=2, help="flag to plot time series data for a specified selected target fill ratio and sensor probability")
    parser.add_argument("-m", action="store_true", help="flag to plot heatmap data; only useful if multiple target fill ratios and sensor probabilities are simulated in data")
    parser.add_argument("-a", action="store_true", help="flag to use aggregate data instead of data from individual experiments")
    args = parser.parse_args()

    # Aggregate experiment data into one mean trajectory
    data = vm.VisualizationData(args.DATA)

    # Plot time series data for specified target fill ratio and sensor probability
    if args.t:
        target_fill_ratio = float(args.t[0])
        sensor_prob = float(args.t[1])
        vm.plot_timeseries(target_fill_ratio, sensor_prob, data, args.a)

    # Plot heatmap for all simulation data
    if args.m:
        vm.plot_heatmap(data)

    plt.show()