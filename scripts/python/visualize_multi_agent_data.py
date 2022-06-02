from sim_modules import ExperimentData
import viz_modules as vm
import argparse
import numpy as np
import matplotlib.pyplot as plt
import timeit

if __name__ == "__main__":

    start = timeit.default_timer()

    parser = argparse.ArgumentParser(description="Visualize multi-agent simulation data")
    parser.add_argument("FILE", type=str, help="path to folder containing serialized ExperimentData files or path to a VisualizationDataGroup pickle file (see the \"g\" flag")
    parser.add_argument("CONV", type=float, help="convergence threshold value (only applies if the \"a\" or \"m\" flags are raised")
    parser.add_argument("-g", action="store_true", help="flag to indicate if the path is pointing to a VisualizationDataGroup pickle file")
    parser.add_argument("-t", nargs="*", help="flag to plot time series data for a specified selected target fill ratio and sensor probability (if group data: comms period, comms probability, and number of agents required)")
    parser.add_argument("-m", action="store_true", help="flag to plot heatmap data; only useful if multiple target fill ratios and sensor probabilities are simulated in data")
    parser.add_argument("-a", action="store_true", help="flag to use aggregate data instead of data from individual experiments")
    parser.add_argument("-s", action="store_true", help="flag to show the plots")
    args = parser.parse_args()

    # Initialize values
    if args.t:
        if args.g and (len(args.t) != 5): raise ValueError("Insufficient arguments for plotting time series data using VisualizationDataGroup file!")
        elif not args.g and (len(args.t) != 2): raise ValueError("Insufficient arguments for plotting time series data using VisualizationData file!")

    # Load data
    if args.g:
        data = vm.VisualizationDataGroup.load(args.FILE)
    else:
        data = vm.VisualizationData(args.FILE)

    # Plot time series data for specified target fill ratio and sensor probability
    if args.t and not args.g:
        target_fill_ratio = float(args.t[0])
        sensor_prob = float(args.t[1])
        vm.plot_timeseries(target_fill_ratio, sensor_prob, data, args.a)
    elif args.t:
        target_fill_ratio = float(args.t[0])
        sensor_prob = float(args.t[1])
        comms_period = int(args.t[2])
        comms_prob = float(args.t[3])
        num_agents = int(args.t[4])

        v = data.get_viz_data_obj(comms_period, comms_prob, num_agents)

        vm.plot_timeseries(target_fill_ratio, sensor_prob, v, args.a, args.CONV)

    # Plot heatmap for all simulation data
    if args.m and not args.g:
        vm.plot_heatmap_vd(data)
    elif args.m:
        vm.plot_heatmap_vdg(
            data,
            [1, 2, 5, 10],
            [10, 50, 100, 200],
            ["Comms. Period = 1", "Comms. Period = 2", "Comms. Period = 5", "Comms. Period = 10"],
            ["Num. Agents = 10", "Num. Agents = 50", "Num. Agents = 100", "Num. Agents = 200"],
            args.CONV,
            comms_network_str="line"
        )

    end = timeit.default_timer()

    print('Elapsed time:', end-start)

    if args.s: plt.show()