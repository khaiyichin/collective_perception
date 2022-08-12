import os
import collective_perception_py.viz_modules as vm
import argparse

def main():

    parser = argparse.ArgumentParser(description="Extract a single VisualizationData object from a VisualizationDataGroup object.")

    # Main arguments
    parser.add_argument("VDG", type=str, help="path to the base VisualationDataGroup pickled object")
    parser.add_argument("-s", type=str, help="path to store the output VisualizationData object")
    parser.add_argument("-tfr", nargs="+", type=float, help="target fill ratios to include in extracted file; default is to extract all")
    parser.add_argument("-sp", nargs="+", type=float, help="sensor probabilities to include in extracted file; default is to extract all")

    # Sub arguments
    data_type_subparser = parser.add_subparsers(dest='data_type', help="data type commands")
    sta_subparser = data_type_subparser.add_parser("static", help="static input data")
    dyn_subparser = data_type_subparser.add_parser("dynamic", help="dynamic input data")

    sta_subparser.add_argument("comms_period", type=int, help="specific communications period")
    sta_subparser.add_argument("comms_prob", type=float, help="specific communications probability")
    sta_subparser.add_argument("num_agents", type=int, help="specific number of agents")

    dyn_subparser.add_argument("density", type=float, help="specific density")
    dyn_subparser.add_argument("speed", type=float, help="specific speed")

    args = parser.parse_args()

    input_vdg = vm.VisualizationDataGroupBase.load(args.VDG)

    if args.data_type == "static":
        keys = {"comms_period": args.comms_period, "comms_prob": args.comms_prob, "num_agents": args.num_agents}
        suffix = "prd{0}_cprob{1}_agt{2}".format(int(args.comms_period), int(args.comms_prob), int(args.num_agents))
    else:
        keys = {"density": args.density, "speed": args.speed}
        suffix = "spd{0}_den{1}".format(int(args.density), int(args.speed))

    output_vd = input_vdg.get_viz_data_obj(keys)

    # Truncate stats object dict
    if args.tfr:
        truncated_dict = {}
        truncated_tfr_range = []

        for t in args.tfr:
            truncated_dict[float(t)] = output_vd.stats_obj_dict[float(t)]
            truncated_tfr_range.append(float(t))

        output_vd.stats_obj_dict = truncated_dict
        output_vd.tfr_range = sorted(truncated_tfr_range)
        output_vd.aggregate_statistics()

    if args.sp:
        truncated_dict = {tfr: None for tfr in output_vd.stats_obj_dict.keys()}
        truncated_sp_range = []
        for s in args.sp:
            [truncated_dict[t].update(output_vd.stats_obj_dict[t][float(s)]) for t in output_vd.tfr_range]
            truncated_sp_range.append(float(s))

        output_vd.stats_obj_dict = truncated_dict
        output_vd.sp_range = sorted(truncated_sp_range)
        output_vd.aggregate_statistics()

    # Add parameter values to filename
    name, ext = os.path.splitext(args.s)

    output_vd.save(name + "_" + suffix + ext)

if __name__ == "__main__":
    main()