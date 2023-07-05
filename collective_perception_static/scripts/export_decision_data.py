#!/usr/bin/env python3
import collective_perception_py.viz_modules as vm
import collective_perception_py.viz_modules_benchmark as vmb
import argparse
import os
import json
import pprint

def main():
    # 
    parser = argparse.ArgumentParser(
        description="Export decision data from a VisualizationDataGroup object (minimalistic collective perception data) or JSON files from a single benchmark algorithm (benchmark data).\
            \nThe exported .decd file contains a dict of dicts in the following form:\
            \n{\
            \n\tsensor_prob_1 (float):\
            \n\t{\
            \n\t    sim_step_11 (int): decision_11 (float),\
            \n\t    sim_step_12 (int): decision_12 (float),\
            \n\t    ...\
            \n\t},\
            \n\tsensor_prob_2 (float):\
            \n\t{\
            \n\t    sim_step_21 (int): decision_21 (float),\
            \n\t    sim_step_22 (int): decision_22 (float),\
            \n\t    ...\
            \n\t},\
            \n\tsensor_prob_3 (float):\
            \n\t{\
            \n\t    sim_step_31 (int): decision_31 (float),\
            \n\t    sim_step_32 (int): decision_32 (float),\
            \n\t    ...\
            \n\t},\
            \n\t...\
            \n}",
    formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        "FILE",
        type=str,
        help="path to the .vdg file (minimalistic collective perception data) or a folder containing the .json files (benchmark data)"
    )
    parser.add_argument(
        "TFR",
        type=float,
        help="the target fill ratio for the experiment that produced the data to be extracted"
    )
    parser.add_argument(
        "PARAM_2_RANGE",
        type=float,
        nargs="+",
        help="space-delimited array of values for the 2nd parameter to extract (typically sensor probabilities)"
    )
    parser.add_argument(
        "--step_inc",
        type=int,
        default=1000,
        help="(optional) the increment in simulation steps to evaluate decisions (default: 1000)"
    )
    parser.add_argument(
        "--nopt",
        type=int,
        default=10,
        help="(optional) the number of options/bins to evaluate decisions (default: 10)"
    )
    parser.add_argument(
        "-s",
        type=str,
        help="path to store the pickle .decd object (without extension)"
    )
    parser.add_argument(
        "-u",
        type=float,
        nargs="+",
        help="(communications period, communication probability, number of agents)/(robot speed, swarm density) to use in extracting static/dynamic data"
    )

    args = parser.parse_args()

    # Load data object
    data_obj = []
    decision_data = {}
    sim_steps = 0

    # Check if data to load is a VDG file (minimalistic collective perception data) or a folder of JSON files (benchmark data)
    if os.path.isdir(args.FILE): # JSON data

        # Pick one random json file to read the `sim_type` field
        sim_type = ""

        for root, _, files in os.walk(args.FILE):
            i = 0
            found = False

            while not found:
                f = files[i]

                if os.path.splitext(f)[1] == ".json":
                    with open(os.path.join(root, f), "r") as file:
                        json_dict = json.load(file)
                        sim_type = json_dict["sim_type"]
                    found = True

                i = i + 1

            if found: break
            else: raise Exception("No .json files found!")

        args.FOLDER = args.FILE # add the .FOLDER field so that the Visualizer object can use it
        
        # Convert JSON files into the Visualizer object so that we can extract the decision data
        if sim_type == vmb.Crosscombe2017Visualizer.BENCHMARK_STR:
            args.FRR = args.PARAM_2_RANGE # add the .FRR field so that the Visualizer object can use it
            data_obj = vmb.Crosscombe2017Visualizer(args)

        elif sim_type == vmb.Ebert2020Visualizer.BENCHMARK_STR:
            args.SP = args.PARAM_2_RANGE # add the .sp field so that the Visualizer object can use it
            data_obj = vmb.Ebert2020Visualizer(args)

        else:
            raise Exception("Invalid benchmark algorithm files!")

        # Extract only the decision data based on the sim steps desired
        if data_obj.num_steps < args.step_inc: raise Exception("Step increments is too large for number of timesteps available.")
        sim_steps = [i for i in range(0, data_obj.num_steps + 1, args.step_inc)]
        decision_data = data_obj.get_decision_data(sim_steps)

    else: # VDG data
        if not args.u: raise Exception("The \"-u\" arguments must be provided for .vdg files.")

        try:
            data_obj = vm.VisualizationDataGroupStatic.load(args.FILE)
            sim_args = {"data_type": "static", "comms_period": args.u[0], "comms_prob": args.u[1], "num_agents": args.u[2]}
            
        except Exception as e:
            try:
                data_obj = vm.VisualizationDataGroupDynamic.load(args.FILE)
                sim_args = {"data_type": "dynamic", "speed": args.u[0], "density": args.u[1]}

            except Exception as e:
                print("Invalid .vdg file!")
                print(e)
                exit(-1)

        try:
            vd_obj = data_obj.get_viz_data_obj(sim_args)
        except KeyError as e:
            print("KeyError:", e)
            print("Available keys are:", data_obj.viz_data_obj_dict.keys())
            exit(-1)

        try:
            # Extract only the decision data based on the sim steps desired
            if vd_obj.num_steps < args.step_inc: raise Exception("Step increments is too large for number of timesteps available.")
            sim_steps = [i for i in range(0, vd_obj.num_steps + 1, args.step_inc)]
            decision_data = {sp: {} for sp in args.PARAM_2_RANGE}

            # Iterate through each sim step to obtain decision fraction
            for ss in sim_steps:
                sp_dec_frac_dict = vd_obj.get_decision_fractions(args.TFR, args.PARAM_2_RANGE, ss, args.nopt)
                for sp, dec_frac in sp_dec_frac_dict.items():
                    decision_data[sp][ss] = dec_frac

        except KeyError as e:
            print("Invalid key! KeyError:", e)
            print("Available target fill ratios are:", vd_obj.tfr_range)
            print("Available sensor probabilities are:", vd_obj.sp_range)
            exit(-1)

    print("\nProcessed decision data:")
    pprint.pprint(decision_data)

    vm.export_decision_data(decision_data, args.s)

if __name__ == "__main__": main()