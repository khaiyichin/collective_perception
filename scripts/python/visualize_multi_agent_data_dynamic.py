import viz_modules as vm
import argparse
import matplotlib.pyplot as plt
import timeit

if __name__ == "__main__":

    start = timeit.default_timer()

    parser = argparse.ArgumentParser(description="Visualize dynamic multi-agent simulation data")
    parser.add_argument("FILE", type=str, help="path to folder containing serialized SimulationStatsSet protobuf files or path to a VisualizationDataGroup pickle file (see the \"g\" flag")
    parser.add_argument("CONV", type=float, help="convergence threshold value (only applies if the \"a\" or \"m\" flags are raised")
    parser.add_argument("-g", action="store_true", help="flag to indicate if the path is pointing to a VisualizationDataGroup pickle file")
    parser.add_argument("-t", nargs="*", help="flag to plot time series data for a specified selected target fill ratio and sensor probability")
    parser.add_argument("-m", action="store_true", help="flag to plot heatmap data; only useful if multiple target fill ratios and sensor probabilities are simulated in data")
    parser.add_argument("-a", action="store_true", help="flag to use aggregate data instead of data from individual experiments")
    parser.add_argument("-u", nargs="*", help="flag to plot data for a specific number of agents and robot speed (only used with -g flag)")
    parser.add_argument("-s", action="store_true", help="flag to show the plots")
    args = parser.parse_args()

    # Check flag values
    if args.t and (len(args.t) != 2): raise ValueError("Insufficient arguments for \"t\" flag!")
    elif args.t:
        target_fill_ratio = float(args.t[0])
        sensor_prob = float(args.t[1])

    if args.g:
        if args.u:
            if len(args.u) != 2: raise ValueError("Insufficient arguments for \"u\" flag!")
            else:
                speed = int(args.u[0])
                num_agents = int(args.u[1])

        try:
            data = vm.VisualizationDataGroupDynamic.load(args.FILE)
        except:
            data = vm.VisualizationDataGroup.load(args.FILE) # TODO: to support legacy class; must remove after upgrade

    else:
        data = vm.VisualizationData(args.FILE)

    # Plot time series data for specified target fill ratio and sensor probability
    if args.t and not args.g:
        vm.plot_timeseries(target_fill_ratio, sensor_prob, data, args.a)
    elif args.t:
        v = data.get_viz_data_obj({"num_agents": num_agents, "speed": speed})

        vm.plot_timeseries(target_fill_ratio, sensor_prob, v, args.a, args.CONV)

    # Plot heatmap for all simulation data
    if args.m and not args.g: # plot single heatmap from VisualizationData
        vm.plot_heatmap_vd(data)
    elif args.m and args.g and not args.u: # plot gridded heatmap
        vm.plot_heatmap_vdg(
            data,
            "speed", # row_arg_str
            [10.0, 15.0, 20.0], # row_keys
            "num_agents", # col_arg_str
            [10, 20, 50, 100], # col_keys
            ["Speed = 10", "Speed = 15", "Speed = 20"], # col_labels
            ["Num. Agents = 10", "Num. Agents = 20", "Num. Agents = 50", "Num. Agents = 100"], # row_labels
            args.CONV,
            title="Dynamic multi-agent simulation performance (convergence threshold: {0})".format(args.CONV)
        )
    elif args.m and args.g and args.u: # plot single heatmap from VisualizationDataGroupDynamic
        v = data.get_viz_data_obj({"num_agents": num_agents, "speed": speed})
        vm.plot_heatmap_vd(
            v,
            args.CONV,
            title="Dynamic multi-agent simulation performance (convergence threshold: {0})".format(args.CONV)
        )

    end = timeit.default_timer()

    print('Elapsed time:', end-start)

    if args.s: plt.show()