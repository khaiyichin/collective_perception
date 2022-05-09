from sim_modules import ExperimentData
import viz_modules as vm
import argparse
import numpy as np

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Visualize multi-agent simulation data")
    parser.add_argument("DATA", type=str, help="filename to the data file")
    parser.add_argument("-t", nargs=2, help="flag to plot time series data for a specified selected target fill ratio and sensor probability")
    parser.add_argument("-m", action="store_true", help="flag to plot heatmap data; only useful if multiple target fill ratios and sensor probabilities are simulated in data")
    args = parser.parse_args()

    # Load the data
    data = ExperimentData.load(args.DATA)

    # Compute convergence for each simulation data entry
    # for dfr in data.dfr_range:
    #     for sp in data.sp_range:
    #         smooth_traj = vm.compute_cma(data.get_sim_data(dfr, sp).x_sample_mean, 3)

    # Plot time series data for a random target fill ratio and sensor probability since there
    # too many to plot
    if args.t:
        target_fill_ratio = float(args.t[0])
        sensor_prob = float(args.t[1])
        vm.plot_timeseries(target_fill_ratio, sensor_prob, data)

    # Plot heatmap for all simulation data
    if args.m:
        pass